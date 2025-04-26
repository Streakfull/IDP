from prep.pre_process_contrastive_loss import PreProcessContrastiveLoss
from tqdm.notebook import tqdm
import numpy as np


class PreProcessPairwiseDistances(PreProcessContrastiveLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def write_dataset(self, m_dict, f_dict, all_random=False, all_close=False, all_far=False, clip_frames=None):
        object_ids = list(m_dict.keys())
        object_ids.sort()
        probs = [f_dict[object_id]
                 for object_id in object_ids]
        if (all_random):
            probs = None
        i = 0
        pairs = None
        with tqdm(total=self.max_pairs, desc="Processing Pairs", unit="pair") as pbar:
            while (i < self.max_pairs):
                object_id = np.random.choice(
                    object_ids, size=1, p=probs)[0]
                occ = matches = m_dict[object_id]
                occ = list(occ.keys())
                occ.sort()
                is_close = np.random.rand() <= 0.5 if all_close is False else True
                if (all_far):
                    is_close = False
                occ = np.array(occ)
                x1 = pivot = np.random.choice(occ, size=1)[0]
                occ = occ[occ != x1]

                d_arr = np.abs(occ - pivot)
                if (clip_frames is not None):
                    occ = occ[d_arr <= clip_frames]
                    d_arr = d_arr[d_arr <= clip_frames]

                if (is_close):
                    d_arr = 1/(d_arr)

                total = d_arr.sum()
                p = d_arr/total
                if (len(occ) <= 1):
                    continue
                if (all_random):
                    p = None
                x2 = np.random.choice(occ, size=1, p=p)[0]
                pair = np.array([int(x1), int(x2)])
                pairs = np.vstack(
                    (pairs, pair)) if pairs is not None else pair
                pbar.update(1)
                i += 1
        np.savetxt(
            f"{self.write_path}/pairs_eval_clipped_30.txt", pairs, fmt='%d')
        return pairs
