import cv2
import numpy as np
from SlamMovie import SlamMovie
from Display import Display
from ImagePair import ImagePair
from SlamCompute import SlamCompute
from Constants import *


class SlamEx:



    @staticmethod
    def ex1():
        # q1
        img_pair = ImagePair.StereoPair(0)

        img_pair.feature_descriptors(MAX_NUM_FEATURES)
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
        img_pair.feature_descriptors(MAX_NUM_FEATURES)
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
        k, m1, m2 = SlamMovie.read_cameras()
        km1, km2 = k @ m1, k @ m2
        points = np.array([[kp0[m.queryIdx].pt, kp1[m.trainIdx].pt] for m in matches])
        triangulated = np.array([SlamCompute.triangulate_pt(km1, km2, p0, p1) for p0, p1 in points])
        cv_triangulated = cv2.triangulatePoints(km1, km2, points[:, 0].T, points[:, 1].T).T
        cv_triangulated = cv_triangulated[:, :3] / cv_triangulated[:, 3:]

        Display.plot_3d(triangulated)
        Display.plot_3d(cv_triangulated)

    @staticmethod
    def ex3():
        # ----------------- 2.1 ------------------
        movie = SlamMovie.Movie()
        m1 = movie.stereo_dist
        im0, im1 = movie.add_pair(0), movie.add_pair(1)
        Display.plot_3d(im0.points_cloud)
        Display.plot_3d(im1.points_cloud)

        # -------------- 2.2 matches --------------
        pair = ImagePair(im0.img0, im1.img0)
        matches = pair.match(update_des=False)
        Display.matches(pair, matches=matches)
        # ---------------- 2.3 PnP ----------------
        R, t = movie.pnp(0, matches, [0, 1, 2, 3, 4, 5, 6, 7])
        inv_R = np.linalg.inv(R)
        left_0 = np.zeros(3)
        right_0 = left_0 - m1
        left_1 = (left_0 @ inv_R.T) - t
        right_1 = ((- m1) @ inv_R.T) - t
        cameras = np.array(
            [[left_0[0], left_0[2]], [right_0[0], right_0[2]], [left_1[0], left_1[2]], [right_1[0], right_1[2]]])
        Display.plot_2d(cameras)

        # ------------- 2.4 supporters -------------
        supporters = movie.findSupporters(0, matches, R, t)
        good_matches = matches[supporters]
        bad_matches = matches[np.logical_not(supporters)]
        good_kp0 = list(map(lambda m: m.queryIdx, good_matches))
        bad_kp0 = list(map(lambda m: m.queryIdx, bad_matches))
        good_kp1 = list(map(lambda m: m.trainIdx, good_matches))
        bad_kp1 = list(map(lambda m: m.trainIdx, bad_matches))

        Display.matches(pair, matches=good_matches)
        Display.matches(pair, matches=bad_matches)
        Display.kp_two_color(im0.img0, im0.img0.kp[good_kp0], im0.img0.kp[bad_kp0])
        Display.kp_two_color(im1.img0, im1.img0.kp[good_kp1], im1.img0.kp[bad_kp1])

        # ------------- 2.5 RANSAC -------------

        best_R, best_t = movie.max_supporters_RANSAC(0, matches, 100)
        cloud1 = im0.points_cloud
        cloud2 = (best_R @ im0.points_cloud.T).T + best_t
        Display.plot_3d(cloud1, cloud2)

        supporters = movie.findSupporters(0, matches, R, t)

        good_matches = matches[supporters]
        bad_matches = matches[np.logical_not(supporters)]
        good_kp0 = list(map(lambda m: m.queryIdx, good_matches))
        bad_kp0 = list(map(lambda m: m.queryIdx, bad_matches))
        good_kp1 = list(map(lambda m: m.trainIdx, good_matches))
        bad_kp1 = list(map(lambda m: m.trainIdx, bad_matches))

        Display.matches(pair, matches=good_matches)
        Display.matches(pair, matches=bad_matches)
        Display.kp_two_color(im0.img0, im0.img0.kp[good_kp0], im0.img0.kp[bad_kp0])
        Display.kp_two_color(im1.img0, im1.img0.kp[good_kp1], im1.img0.kp[bad_kp1])

        # ----------- 2.6 whole movie -----------
        num = 2760
        movie = SlamMovie.Movie()
        movie.add_pair(0)
        for i in range(1, num):
            print(i)
            movie.add_pair(i)
            movie.transformation(i - 1)

        track1 = Display.show_track(movie.transformations, num, 1)
        track2 = Display.show_track(np.loadtxt(GT_POSES), 2760, 0)

        Display.plot_2d(track1[:, [0, 2]],track2[:, [0, 2]] )
