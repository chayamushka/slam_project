import cv2
import numpy as np

from Constants import *
from Image import Image
from ImagePair import ImagePair


class Frame(ImagePair):
    def __init__(self, idx=0):
        super().__init__(Image(idx, 0), Image(idx, 1))
        self.frame_id = idx
        self.R_relative = np.eye(3)
        self.t_relative = np.zeros(3)
        self.ba_R = None
        self.ba_t = None
        self.supporter_ratio = 0
        self.match()
        self.tracks = set()

    def update_tracks_ids(self, tracks):
        self.tracks.update(set(tracks))

    def get_link_size(self):
        return len(list(filter(lambda t: not t.is_last(self.frame_id), self.tracks)))

    def get_num_track(self):
        return len(self.tracks)

    def get_inlier_percentage(self):
        return self.supporter_ratio

    def set_relative_position(self, R, t,supporters):
        self.R_relative, self.t_relative = R, t
        self.supporter_ratio = sum(supporters) / len(supporters)

    def filter_des(self):
        indx0 = []
        indx1 = []
        # full data
        kp0, kp1 = self.get_kps()
        des0, des1 = self.get_des()
        # some data
        for i, match in enumerate(self.matches):
            indx0.append(match.queryIdx)
            indx1.append(match.trainIdx)
            match.queryIdx = i
            match.trainIdx = i
        indx0 = np.array(indx0)
        indx1 = np.array(indx1)
        self.img0.set_kp(kp0[indx0])
        self.img0.set_des(des0[indx0])
        self.img1.set_kp(kp1[indx1])
        self.img1.set_des(des1[indx1])

    def match(self, ratio=0.8):
        matches = super().match()
        self.matches = matches
        self.stereo_filter()
        print("num matches : ", len(matches))
        self.filter_des()
        return matches

    def stereo_filter(self, pixels=2):
        kp0, kp1 = self.get_kps()
        y_dist = list(map(lambda m: abs(kp0[m.queryIdx][1] - kp1[m.trainIdx][1]), self.matches))
        y_dist = np.array(y_dist)
        self.matches = self.matches[y_dist <= pixels]
        return self.matches

    def triangulate(self, km1, km2):
        kp0, kp1 = self.get_kps()
        points = np.array([[kp0[m.queryIdx], kp1[m.trainIdx]] for m in self.matches])
        points_cloud = cv2.triangulatePoints(km1, km2, points[:, 0].T, points[:, 1].T).T
        points_cloud = points_cloud[:, :3] / points_cloud[:, 3:]
        return points_cloud
