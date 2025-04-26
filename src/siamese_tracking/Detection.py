# vim: expandtab:ts=4:sw=4
import numpy as np
import torch


class Detection(object):

    def __init__(self, bb):
        if torch.is_tensor(bb):
            bb = bb.cpu().numpy()
        x1, y1, x2, y2, conf, cls = bb
        w, h = x2-x1, y2-y1
        self.tlwh = np.asarray([x1, y1, w, h], dtype=float)
        self.confidence = float(conf)
        # self.feature = np.asarray(feature, dtype=np.float32)
        self.feature = np.ones(128)
        self.classif = cls
        self.id = None
        self.xyxy = np.array([x1, y1, x2, y2])
        self.jid = None

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def get_conf(self):
        return self.confidence

    def get_cls(self):
        return self.classif

    def set_feature(self, feat):
        self.feature = feat.cpu().numpy()

    def get_feature(self):
        return self.feature

    def to_tlwh(self):
        return self.tlwh

    def set_id(self, id):
        self.id = id

    def get_xywh(self):
        """Returns bounding box in (x, y, w, h) format."""
        return self.tlwh

    def set_jid(self, id):
        self.jid = id

    def set_classif(self, cls):
        self.classif = cls

    def is_player(self):
        return self.classif in [0, 1, 4, 5]

    @property
    def bb(self):
        return np.concatenate((self.xyxy, [self.confidence]))
