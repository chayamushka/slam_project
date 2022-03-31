import cv2
import numpy as np

from Image import Image

MAX_NUM_FEATURES = 500


class ImagePair:

    def __init__(self, idx=0):
        self.idx = idx
        self.img0 = Image(idx, 0)
        self.img1 = Image(idx, 1)
        self.matches = None

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
            self.BFMatch()
        return self.matches

    def feature_descriptors(self, feature_num=MAX_NUM_FEATURES):
        orb = cv2.ORB_create(MAX_NUM_FEATURES)
        self.apply_images(lambda i: Image.detect_kp_compute_des(i, orb))

    def BFMatch(self, matcher=None, sort=True):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.matches = matcher.match(*self.apply_images(Image.get_des))
        self.matches = sorted(self.matches, key=lambda m: m.distance) if sort else self.matches

    def significant_match(self, ratio=0.8):
        bf = cv2.BFMatcher()
        self.knn_matches = bf.knnMatch(*self.apply_images(Image.get_des), k=2)
        self.significant_matches = []
        for m, n in self.knn_matches:
            if m.distance < ratio * n.distance:
                self.significant_matches.append(m)
        self.significant_matches = sorted(self.significant_matches, key=lambda m: m.distance)
        return self.significant_matches



