import numpy as np
from collections import deque

from hm_sort_app import matching
from hm_sort_app.basetrack import BaseTrack, TrackState
from hm_sort_app.kalman_filter import KalmanFilter
from hm_sort_app.Strack import STrack

from collections import defaultdict


class Deep_EIoU(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = 0.8
        self.appearance_thresh = 0.3

    def adjust_dist(self, emb_dists, ious_dists):
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        raw_embed = emb_dists.copy()
        # emb_dists = emb_dists / 2.0
        emb_dists[emb_dists > self.appearance_thresh] = 1.0
        emb_dists[ious_dists_mask] = 1.0

        eps = np.finfo(float).eps
        num_fixed = np.sum(np.abs(ious_dists) <= 0) + \
            np.sum(np.abs(emb_dists) <= 0)
        safe_ious = np.where(np.abs(ious_dists) <= 0, eps, ious_dists)
        safe_embs = np.where(np.abs(emb_dists) <= 0, eps, emb_dists)

        if num_fixed > 0:
            print(f"Warning: {num_fixed} near-zero distances corrected")

        dists = (1 / safe_ious) + (1 / safe_embs)
        dists = 2 / dists
        # dists = emb_dists
        # dists[emb_dists > self.appearance_thresh] = 1
       # dists = np.minimum(ious_dists, emb_dists)
        # dists[emb_dists > self.appearance_thresh] = 1.0
        return dists

    def adjust_dist2(self, emb_dists, ious_dists):
        emb_dists[emb_dists > self.appearance_thresh] = 1.0
        return emb_dists

    def update(self, output_results, embedding):
        '''
        output_results : [x1,y1,x2,y2,score] type:ndarray
        embdding : [emb1,emb2,...] dim:512
        '''

        self.frame_id += 1
        # if (self.frame_id > 200):
        #     import pdb
        #     pdb.set_trace()
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]  # x1y1x2y2
            elif output_results.shape[1] == 7:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
            else:
                raise ValueError('Wrong detection size {}'.format(
                    output_results.shape[1]))

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]

            if self.args.with_reid:
                embedding = embedding[lowest_inds]
              #  cls = cls[lowest_inds]
                features_keep = embedding[remain_inds]
               # cls_keep = cls[remain_inds]

        else:
            bboxes = []
            scores = []
            dets = []
            scores_keep = []
            features_keep = []
            jids_keep = []
           # cls = []

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Associate with high score detection boxes
        num_iteration = 2
        # init_expand_scale = 0.7
        # expand_scale_step = 0.1
        init_expand_scale = 0.3
        expand_scale_step = 0.3

        for iteration in range(num_iteration):

            cur_expand_scale = init_expand_scale + expand_scale_step*iteration

            ious_dists = matching.eiou_distance(
                strack_pool, detections, cur_expand_scale)
            ious_dists_mask = (ious_dists > self.proximity_thresh)

            if self.args.with_reid:
                emb_dists = matching.embedding_distance(
                    strack_pool, detections)
                dists = self.adjust_dist(emb_dists, ious_dists)
            else:
                dists = ious_dists

            matches, u_track, u_detection = matching.linear_assignment(
                dists, thresh=self.args.match_thresh)

            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            strack_pool = [strack_pool[i]
                           for i in u_track if strack_pool[i].state == TrackState.Tracked]
            detections = [detections[i] for i in u_detection]

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            if self.args.with_reid:
                features_second = embedding[inds_second]
        else:
            dets_second = []
            scores_second = []
            features_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                                     (tlbr, s, f) in zip(dets_second, scores_second, features_second)]
            else:
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                     (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = strack_pool
        dists = matching.eiou_distance(
            r_tracked_stracks, detections_second, expand=0.5)
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        ious_dists = matching.eiou_distance(unconfirmed, detections, 0.5)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(
                unconfirmed, detections)

            dists = self.adjust_dist(emb_dists, ious_dists)
            # raw_emb_dists = emb_dists.copy()

        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                # track.mark_removed()
                # removed_stracks.append(track)
                pass

        """ Merge """
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks]

        return output_stracks

    def update_k(self, output_results, embedding):
        """
        output_results : [x1,y1,x2,y2,score] type: ndarray
        embedding : [emb1,emb2,...] dim: 512
        """
        self.frame_id += 1

        activated_tracks = []
        lost_tracks = []

        if len(output_results):
            if output_results.shape[1] in [5, 7]:
                bboxes = output_results[:, :4]  # x1y1x2y2
            else:
                raise ValueError(
                    f'Wrong detection size {output_results.shape[1]}')
        else:
            bboxes = []

        detections = [STrack(STrack.tlbr_to_tlwh(tlbr), 1.0, feat=f)
                      for (tlbr, f) in zip(bboxes, embedding)]  # Assigning default score 1.0

        ''' Match all tracks with all detections using expansion iterations '''
        strack_pool = self.tracked_stracks + self.lost_stracks

        num_iteration = 2
        init_expand_scale = 0.7
        expand_scale_step = 0.1

        for iteration in range(num_iteration):
            cur_expand_scale = init_expand_scale + expand_scale_step * iteration

            ious_dists = matching.eiou_distance(
                strack_pool, detections, cur_expand_scale)

            if self.args.with_reid:

                emb_dists = matching.embedding_distance(
                    strack_pool, detections)
                dists = self.adjust_dist(emb_dists, ious_dists)
            else:
                dists = ious_dists

            matches, u_track, u_detection = matching.linear_assignment(
                dists, thresh=self.args.match_thresh)

            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                track.update(det, self.frame_id)
                activated_tracks.append(track)

            strack_pool = [strack_pool[i] for i in u_track]
            detections = [detections[i] for i in u_detection]

        ''' Initialize new tracks '''
        for inew in range(len(detections)):
            track = detections[inew]
            track.activate(self.kalman_filter, self.frame_id)
            activated_tracks.append(track)

        ''' Update lost tracks '''
        for track in strack_pool:
            track.mark_lost()
            lost_tracks.append(track)

        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, activated_tracks)
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_tracks)

        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)

        return self.tracked_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
