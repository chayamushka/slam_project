import cv2
import numpy as np
import os.path
from Constants import *
DATA_PATH = os.path.join("dataset", "sequences", "00")


class Image:

    def __init__(self, idx, side):
        img_name = '{:06d}.png'.format(idx)
        image = cv2.imread(os.path.join(DATA_PATH, f"image_{side}", f"{img_name}"), 0)
        key_points, self.des = cv2.SIFT_create(MAX_NUM_FEATURES).detectAndCompute(image, None)
        self.kp = np.array([kp.pt for kp in key_points])
        self.idx = idx
        self.side = side


    def get_image(self):
        return self.read_image(self.idx, self.side)

    def get_des(self):
        return self.des

    def get_kp(self, idx=None):
        return self.kp if idx is None else self.kp[idx]

    def get_kp_pt(self, idx):
        return self.kp[idx]

    def set_des(self, des):
        self.des = des

    def set_kp(self, kp):
        self.kp = kp

    @staticmethod
    def read_image(idx, side):
        return cv2.imread(os.path.join(DATA_PATH, f"image_{side}", f"{'{:06d}.png'.format(idx)}"), 0)
