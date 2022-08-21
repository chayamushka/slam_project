import numpy as np
import cv2
from math import log


class SlamCompute:

    @staticmethod
    def triangulate_pt(c1, c2, p1, p2):
        """
        This function computes the triangulation of 2 points, given their x,y and
        the camera matrices
        :param c1: left camera matrix
        :param c2: right camera matrix
        :param p1: left 2d point
        :param p2: right 2d point
        :return: 3d point
        """
        r1 = c1[2] * p1[0] - c1[0]
        r2 = c1[2] * p1[1] - c1[1]
        r3 = c2[2] * p2[0] - c2[0]
        r4 = c1[2] * p2[1] - c2[1]
        return SlamCompute.svd_decomposition(np.vstack((r1, r2, r3, r4)))

    @staticmethod
    def svd_decomposition(A):
        __, _, v = np.linalg.svd(A)
        X = v[-1]
        return X[:-1] / X[-1]

    @staticmethod
    def projection(pt, K, R=None, t=None):
        """
        This function computes the projection of a 3d point on the pixel image
        :param K: Intrinsic matrix
        :param R: Rotation matrix
        :param t: Transition vector
        :param pt: 3d point
        :return: 2d point
        """
        if R is None:
            R = np.eye(3)
        if t is None:
            t = [0, 0, 0]
        pt = K @ ((R @ pt) + t)
        return pt[:2] / pt[2:]

    @staticmethod
    def ransac_loop(p, supporters, num_of_point=6):
        return log(1 - p) / log(1 - pow(supporters, num_of_point))



