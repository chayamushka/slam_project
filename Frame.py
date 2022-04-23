import uuid

import cv2
import numpy as np

from Constants import *
from Image import Image
from ImagePair import ImagePair


class Frame(ImagePair):
    def __init__(self, idx=0, cam1=None, cam2=None, feature_num=MAX_NUM_FEATURES):
        super().__init__(Image(idx, 0), Image(idx, 1))
        self.frame_id = idx
        self.R = np.eye(3)
        self.t = np.zeros(3)
        self.feature_descriptors(feature_num)
        self.match()
        self.tracks = []
        if cam1 is not None:
            self.triangulate(cam1, cam2)
    def set_position(self, R,t):
        self.R , self.t = R,t
    def filter_des(self):
        indx0 = []
        indx1 = []
        # full data
        kp0, kp1 = np.array(self.get_kps())
        des0, des1 = np.array(self.get_des())
        # some data
        for i, match in enumerate(self.matches):
            indx0.append(match.queryIdx)
            indx1.append(match.trainIdx)
            match.queryIdx = i
            match.trainIdx = i

        self.img0.set_kp(kp0[indx0])
        self.img0.set_des(des0[indx0])
        self.img1.set_kp(kp1[indx1])
        self.img1.set_des(des1[indx1])

    def match(self, ratio=0.8):
        self.matches = super().match()
        self.stereo_filter()
        self.filter_des()
        return self.matches

    def stereo_filter(self, pixels=2):
        kp0, kp1 = self.get_kps()
        y_dist = list(map(lambda m: abs(kp0[m.queryIdx].pt[1] - kp1[m.trainIdx].pt[1]), self.matches))
        y_dist = np.array(y_dist)
        self.matches = self.matches[y_dist <= pixels]
        return self.matches

    def triangulate(self, km1, km2):
        kp0, kp1 = self.get_kps()
        points = np.array([[kp0[m.queryIdx].pt, kp1[m.trainIdx].pt] for m in self.matches])
        self.points_cloud = cv2.triangulatePoints(km1, km2, points[:, 0].T, points[:, 1].T).T
        self.points_cloud = self.points_cloud[:, :3] / self.points_cloud[:, 3:]
        return self.points_cloud

    def get_tracks_ids(self):
        return list(map(lambda t: t.track_id, self.tracks))
