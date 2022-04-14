import cv2
import numpy as np

from Constants import *
from Image import Image


# from SlamEx import SlamEx


class ImagePair:

    def __init__(self, img0: Image, img1: Image):
        self.idx = img0.idx
        self.img0 = img0
        self.img1 = img1
        self.matches = None
        self.points_cloud = None
        self.R = np.eye(3)
        self.t = np.zeros(3)

    @classmethod
    def StereoPair(cls, idx=0,  cam1=None, cam2=None, feature_num=MAX_NUM_FEATURES,):
        pair = cls(Image(idx, 0), Image(idx, 1))
        pair.feature_descriptors(feature_num)
        pair.match()
        pair.stereo_filter()
        if cam1 is not None:
            pair.triangulate(cam1, cam2)
        return pair

    def apply_images(self, f):
        return f(self.img0), f(self.img1)

    def get_kps(self):
        return self.apply_images(Image.get_kp)

    def get_des(self):
        return self.apply_images(Image.get_des)

    def get_image(self, side=None):
        if side is None:
            return self.img0.image, self.img1.image
        return self.img0.image if side == 0 else self.img1.image

    def get_matches(self):
        if self.matches is None:
            self.old_match()
        return self.matches

    def feature_descriptors(self, feature_num):
        orb = cv2.ORB_create(feature_num)
        self.apply_images(lambda i: Image.detect_kp_compute_des(i, orb))

    def old_match(self):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.matches = matcher.match(*self.apply_images(Image.get_des))
        self.matches = sorted(self.matches, key=lambda m: m.distance)

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

    def match(self, ratio=0.8, update_des=True):
        matcher = cv2.BFMatcher()
        self.knn_matches = matcher.knnMatch(*self.get_des(), k=2)
        self.matches = []
        for m, n in self.knn_matches:
            if m.distance < ratio * n.distance:
                self.matches.append(m)
        self.matches = np.array(sorted(self.matches, key=lambda m: m.distance))
        if update_des:
            self.filter_des()
        return self.matches

    def stereo_filter(self, pixel_Separator=2):
        kp0, kp1 = self.get_kps()
        y_dist = list(map(lambda m: abs(kp0[m.queryIdx].pt[1] - kp1[m.trainIdx].pt[1]), self.matches))
        y_dist = np.array(y_dist)
        self.matches = self.matches[y_dist <= pixel_Separator]
        self.filter_des()
        return self.matches



    def triangulate(self, km1, km2):
        kp0, kp1 = self.get_kps()
        points = np.array([[kp0[m.queryIdx].pt, kp1[m.trainIdx].pt] for m in self.matches])
        self.points_cloud = cv2.triangulatePoints(km1, km2, points[:, 0].T, points[:, 1].T).T
        self.points_cloud = self.points_cloud[:, :3] / self.points_cloud[:, 3:]
        return self.points_cloud


