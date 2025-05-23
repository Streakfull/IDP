
from __future__ import print_function
from skimage import io
# import matplotlib.patches as patches
# import matplotlib.pyplot as plt
import torch
from deep_sort.sk_learn_linear_assignment import linear_assignment

import os
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')


class Tracker(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def linear_assignment(self, cost_matrix):
        # x, y = linear_assignment(cost_matrix)
        # return np.array(list(zip(x, y)))\
        indices = linear_assignment(cost_matrix)
        return indices

    # def iou_batch(self, bb_test, bb_gt):
    #     """
    #     From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    #     """
    #     # bb_gt = np.expand_dims(bb_gt, 0)
    #     # bb_test = np.expand_dims(bb_test, 1)
    #     bb_gt = torch.Tensor(bb_gt).unsqueeze(0).to(bb_test.device)
    #     bb_test = bb_test.unsqueeze(1)

    #     xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    #     yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    #     xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    #     yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    #     w = np.maximum(0., xx2 - xx1)
    #     h = np.maximum(0., yy2 - yy1)
    #     wh = w * h
    #     o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    #               + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    #     return (o)
    def iou_batch(self, bb_test, bb_gt):
        """
        Computes IOU between two bboxes in the form [x1, y1, x2, y2]
        Args:
            bb_test: Tensor of bounding boxes to test, shape [N, 4]
            bb_gt: Tensor of ground truth bounding boxes, shape [M, 4]
        Returns:
            Tensor of IOU values, shape [N, M]
        """
        # Ensure bb_gt is a tensor and add a dimension
        bb_gt = torch.Tensor(bb_gt).unsqueeze(0).to(
            bb_test.device)  # Shape: [1, M, 4]
        bb_test = bb_test.unsqueeze(1)  # Shape: [N, 1, 4]

        # Calculate the intersection
        xx1 = torch.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = torch.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = torch.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = torch.minimum(bb_test[..., 3], bb_gt[..., 3])

        # Calculate width and height of intersection
        w = torch.maximum(torch.tensor(0.0, device=bb_test.device), xx2 - xx1)
        h = torch.maximum(torch.tensor(0.0, device=bb_test.device), yy2 - yy1)

        # Calculate area of intersection
        wh = w * h

        # Calculate IOU
        o = wh / (
            (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) +
            (bb_gt[..., 2] - bb_gt[..., 0]) *
            (bb_gt[..., 3] - bb_gt[..., 1]) - wh
        )

        return (o)  # Shape: [N, M]

    def convert_bbox_to_z(self, bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h  # scale is just area
        r = w / float(h)
        # import pdb
        # pdb.set_trace()
        # if(torch.cuda.is)
        return torch.Tensor([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x, score=None, classif=None):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """

        w = np.sqrt(x[2] * x[3])
        h = x[2] / w

        if (score == None and classif == None):
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., np.array([score.item()]), np.array([classif.item()])]).reshape((1, 6))

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if (len(trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        iou_matrix = self.iou_batch(detections, trackers)

        if min(iou_matrix.shape) > 0:
            # import pdb
            # pdb.set_trace()
            # a = (iou_matrix > iou_threshold).astype(np.int32)
            a = (iou_matrix > iou_threshold).to(torch.int)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a.cpu().numpy()), axis=1)
            else:
                matched_indices = self.linear_assignment(
                    -iou_matrix.cpu().numpy())
        else:
            matched_indices = np.empty(shape=(0, 2))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if (d not in matched_indices[:, 0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if (t not in matched_indices[:, 1]):
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if (iou_matrix[m[0], m[1]] < iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
