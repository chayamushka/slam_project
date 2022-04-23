import cv2
import numpy as np

from Constants import *
from ImagePair import ImagePair
from SlamCompute import SlamCompute


class SlamMovie:
    def __init__(self, K, stereo_dist):
        self.img_pairs = []
        self.transformations = []
        self.K, self.stereo_dist = K, stereo_dist

    @classmethod
    def Movie(cls):
        K, _, stereo_dist = SlamMovie.read_cameras()
        movie = cls(K, stereo_dist[:, 3])
        return movie

    def add_pair(self, idx=None, pair=None):
        if pair is None:
            cam1, cam2 = np.c_[np.eye(3), np.zeros(3)], np.c_[np.eye(3), np.zeros(3)]
            cam2[:, 3] += self.stereo_dist
            pair = ImagePair.StereoPair(idx, self.K @ cam1, self.K @ cam2)
        self.img_pairs.append(pair)
        return pair

    def transformation(self, idx):
        if idx >= len(self.img_pairs) - 1:
            return None
        im1, im2 = self.img_pairs[idx], self.img_pairs[idx + 1]
        pair = ImagePair(im1.img0, im2.img0)
        matches = pair.match(update_des=False)
        R, t = self.max_supporters_RANSAC(idx, matches, 100)
        if len(self.transformations):
            R = self.transformations[-1][0] @ R
            t = self.transformations[-1][0] @ t + self.transformations[-1][1]
        else:
            self.transformations.append([np.eye(3), np.zeros(3)])
        self.transformations.append([R, t])
        return R, t

    @staticmethod
    def read_cameras():
        with open(CAMERA_PATH) as f:
            l1 = f.readline().split()[1:]  # skip first token
            l2 = f.readline().split()[1:]  # skip first token
        l1 = [float(i) for i in l1]
        m1 = np.array(l1).reshape(3, 4)
        l2 = [float(i) for i in l2]
        m2 = np.array(l2).reshape(3, 4)
        k = m1[:, :3]
        m1 = np.linalg.inv(k) @ m1
        m2 = np.linalg.inv(k) @ m2
        return k, m1, m2
    def get_track(self, n):
        return np.array([t * [1, 1, -1] for r, t in self.transformations[:n]])
    def pnp(self, idx, matches, ind):
        points_3d = self.img_pairs[idx].points_cloud[list(map(lambda m: m.queryIdx, matches[ind]))]
        key_points = self.img_pairs[idx + 1].get_kps()[0][list(map(lambda m: m.trainIdx, matches[ind]))]
        pixels = np.expand_dims(cv2.KeyPoint.convert(key_points), axis=1)
        _, rvec, tvec = cv2.solvePnP(points_3d, pixels, self.K, np.zeros((4, 1), dtype=np.float64))
        R = cv2.Rodrigues(rvec)[0]
        return R, tvec.squeeze()

    def findSupporters(self, idx, matches, R, t):
        thresh = lambda ptx, px: np.all(np.abs(ptx - px) <= 2)
        supporters = np.full(len(matches), True)
        pair1, pair2 = self.img_pairs[idx], self.img_pairs[idx + 1]
        for i, match in enumerate(matches):
            point = pair1.points_cloud[match.queryIdx]
            left0 = thresh(SlamCompute.projection(point, self.K), pair1.img0.get_kp_pt(match.queryIdx))
            right0 = thresh(SlamCompute.projection(point + self.stereo_dist, self.K),
                            pair1.img1.get_kp_pt(match.queryIdx))
            left1 = thresh(SlamCompute.projection(point, self.K, R, t), pair2.img0.get_kp_pt(match.trainIdx))
            right1 = thresh(SlamCompute.projection(point + self.stereo_dist, self.K, R, t),
                            pair2.img1.get_kp_pt(match.trainIdx))
            supporters[i] = left0 and left1 and right0 and right1
        return supporters

    def max_supporters_RANSAC(self, idx, matches, max_loop=10000, min_loop=200, num_of_point=6, p=0.99):
        num_max_supporters = 0
        ransac_size = 0
        count_loop = 0
        best_R, best_t = np.eye(3), np.zeros(3)
        max_supporters = np.full(len(matches), False)
        while ransac_size + min_loop - count_loop > 0 and count_loop < max_loop:
            ind = np.random.choice(len(matches), size=num_of_point, replace=len(matches) < num_of_point)
            try:
                R, t = self.pnp(idx, matches, ind)
                supporters = self.findSupporters(idx, matches, R, t)
                if sum(supporters) > num_max_supporters:
                    num_max_supporters, max_supporters = sum(supporters), supporters
                    best_R, best_t = R, t
                    ransac_size = SlamCompute.ransac_loop(p, num_max_supporters / len(matches), num_of_point)
                    print("ransac_size: ", ransac_size)
            except:
                pass
            count_loop += 1
        print("before: ", num_max_supporters / len(matches), ": ", num_max_supporters, "/", len(matches))
        while True:
            try:
                R, t = self.pnp(idx, matches, max_supporters)
                supporters = self.findSupporters(idx, matches, R, t)
                if sum(supporters) <= num_max_supporters:
                    break
                num_max_supporters, max_supporters = sum(supporters), supporters
                best_R, best_t = R, t
            except:
                break
        print("after: ", num_max_supporters / len(matches), ": ", num_max_supporters, "/", len(matches))
        return [best_R, best_t]
