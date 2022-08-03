import cv2
import numpy as np

from Image import Image
from Constants import *


class ImagePair:

    class Match:
        def __init__(self, qu, tr):
            self.queryIdx = qu
            self.trainIdx = tr

    def __init__(self, img0: Image, img1: Image):
        self.idx = img0.idx
        self.img0 = img0
        self.img1 = img1
        self.matches = None


    def apply_images(self, f):
        return f(self.img0), f(self.img1)

    def get_kps(self):
        return self.apply_images(Image.get_kp)

    def get_des(self):
        return self.apply_images(Image.get_des)

    def get_images(self):
        return self.apply_images(Image.get_image)

    def match(self, ratio=SIGNIFICANCE_RATIO):
        matcher = cv2.BFMatcher()
        knn_matches = matcher.knnMatch(*self.get_des(), k=2)
        matches = []
        for m, n in knn_matches:
            if m.distance < ratio * n.distance:
                matches.append(m)
        matches = np.array(sorted(matches, key=lambda m: m.distance))
        self.matches = np.array([self.Match(m.queryIdx, m.trainIdx) for m in matches])
        return self.matches

    @staticmethod
    def get_match_idx(matches):
        return np.array(list(map(lambda m: m.queryIdx, matches))), np.array(list(map(lambda m: m.trainIdx, matches)))
