import cv2
import numpy as np

from Display import Display
from ImagePair import ImagePair
from SlamCompute import SlamCompute


class SlamEx:
    DATA_PATH = "dataset\\sequences\\00\\"
    MAX_NUM_FEATURES = 1000

    @staticmethod
    def read_cameras():
        with open(SlamEx.DATA_PATH + 'calib.txt') as f:
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

    @staticmethod
    def ex1():
        # q1
        img_pair = ImagePair.StereoPair(0)

        img_pair.feature_descriptors(SlamEx.MAX_NUM_FEATURES)
        Display.kp(img_pair.img0, "kp img1.jpg")
        Display.kp(img_pair.img1, "kp img2.jpg")

        # q2
        des0, des1 = img_pair.get_des()
        print("descriptor 0:", des0[0], "\ndescriptor 1:", des1[0])

        # q3
        # Match features.
        img_pair.old_match(sort=True)
        Display.matches(img_pair, txt="matches_pair_0.jpg")

        # q4
        img_pair.match()
        bad_matches = list(
            filter(lambda x: x not in img_pair.significant_matches, map(lambda x: x[0], img_pair.knn_matches)))

        print("good ratio", len(img_pair.significant_matches) / 1000)
        Display.matches(img_pair, txt="significant_matches.jpg", matches=img_pair.significant_matches)
        Display.matches(img_pair, matches=[bad_matches[-1]], num_matches=1)

    @staticmethod
    def ex2():
        # Intro
        img_pair = ImagePair.StereoPair(0)
        img_pair.feature_descriptors(SlamEx.MAX_NUM_FEATURES)
        img_pair.match()
        kp0, kp1 = img_pair.get_kps()
        matches = img_pair.significant_matches

        # q1
        y_dist = list(map(lambda m: abs(kp0[m.queryIdx].pt[1] - kp1[m.trainIdx].pt[1]), matches))
        h = Display.hist(y_dist, save=True)
        deviate_2 = sum(h[3:]) * 100 / sum(h)
        print("{:6.2f} %".format(deviate_2))

        # q2
        y_dist = np.array(y_dist)
        matches = np.array(matches)
        accepted, rejected = matches[y_dist <= 2], matches[y_dist > 2]
        accepted_idx0, rejected_idx0 = np.array(list(map(lambda m: m.queryIdx, accepted))), np.array(
            list(map(lambda m: m.queryIdx, rejected)))
        accepted_idx1, rejected_idx1 = np.array(list(map(lambda m: m.trainIdx, accepted))), np.array(
            list(map(lambda m: m.trainIdx, rejected)))

        kp0, kp1 = np.array(kp0), np.array(kp1)
        Display.kp_two_color(img_pair.img0, kp0[accepted_idx0], kp0[rejected_idx0], save=True)
        Display.kp_two_color(img_pair.img1, kp1[accepted_idx1], kp1[rejected_idx1], save=True)
        print("discarded: ", len(rejected))
        # Assuming the Y-coordinate of erroneous matches is distributed uniformly across the
        # image, what ratio of matches would you expect to be wrong with this rejection policy
        print("{:6.2f} %".format((1 - (3 / len(img_pair.img0.image))) * 100))

        # q3
        k, m1, m2 = SlamEx.read_cameras()
        km1, km2 = k @ m1, k @ m2
        points = np.array([[kp0[m.queryIdx].pt, kp1[m.trainIdx].pt] for m in matches])
        triangulated = np.array([SlamCompute.triangulate_pt(km1, km2, p0, p1) for p0, p1 in points])
        cv_triangulated = cv2.triangulatePoints(km1, km2, points[:, 0].T, points[:, 1].T).T
        cv_triangulated = cv_triangulated[:, :3] / cv_triangulated[:, 3:]

        Display.plot_3d(triangulated)
        Display.plot_3d(cv_triangulated)
