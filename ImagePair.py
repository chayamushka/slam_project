import cv2
import numpy as np

from Image import Image


class ImagePair:

    def __init__(self, img0: Image, img1: Image):
        self.idx = img0.idx
        self.img0 = img0
        self.img1 = img1
        self.matches = None
        self.points_cloud = None

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
            self.match()
        return self.matches

    def feature_descriptors(self, feature_num):
        orb = cv2.ORB_create(feature_num)
        self.apply_images(lambda i: Image.detect_kp_compute_des(i, orb))

    def match(self, ratio=0.8):
        matcher = cv2.BFMatcher()
        knn_matches = matcher.knnMatch(*self.get_des(), k=2)
        self.matches = []
        for m, n in knn_matches:
            if m.distance < ratio * n.distance:
                self.matches.append(m)
        self.matches = np.array(sorted(self.matches, key=lambda m: m.distance))
        return self.matches

    @staticmethod
    def get_match_idx(matches):
        return list(map(lambda m: m.queryIdx, matches)), list(map(lambda m: m.trainIdx, matches))
