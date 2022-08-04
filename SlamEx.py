import cv2
import matplotlib.pyplot as plt
import numpy as np

from Constants import *
from Display import Display
from Frame import Frame
from ImagePair import ImagePair
from SlamCompute import SlamCompute
from SlamMovie import SlamMovie


class SlamEx:
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

    @staticmethod
    def create_movie():
        K, _, stereo_dist = SlamEx.read_cameras()
        movie = SlamMovie(K, stereo_dist[:, 3])
        return movie

    @staticmethod
    def load_positions(num=FRAME_NUM):
        pose = np.loadtxt(GT_POSES).reshape((-1, 3, 4))[:num]
        return (np.linalg.inv(pose[:, :, :3]) @ (-pose[:, :, 3:])).squeeze()

    @staticmethod
    def load_poses(num=FRAME_NUM):
        return np.loadtxt(GT_POSES).reshape((-1, 3, 4))[:num]



    @staticmethod
    def go(num=FRAME_NUM):
        from time import time
        start = time()
        movie = SlamEx.create_movie()
        movie.run(num)
        print("time", (time() - start))
        track1 = movie.get_t_positions(num)
        track2 = SlamEx.load_positions(num)
        Display.scatter_2d(track1[:, [0, 2]], track2[:, [0, 2]])
        SlamMovie.save(movie)

    @staticmethod
    def ex1():
        # q1
        img_pair = Frame(0)

        # Display.kp(img_pair.img0, "kp img1.jpg")
        # Display.kp(img_pair.img1, "kp img2.jpg")

        # q2
        des0, des1 = img_pair.get_des()
        print("descriptor 0:", des0[0], "\ndescriptor 1:", des1[0])

        # q3
        # Match features.
        img_pair.match()
        # Display.matches(img_pair, txt="matches_pair_0.jpg")

        # q4
        img_pair.match()
        bad_matches = list(
            filter(lambda x: x not in img_pair.matches, map(lambda x: x[0], img_pair.knn_matches)))

        print("good ratio", len(img_pair.matches) / 1000)
        # Display.matches(img_pair, txt="significant_matches.jpg", matches=img_pair.matches)
        # Display.matches(img_pair, matches=[bad_matches[-1]], num_matches=1)

    @staticmethod
    def ex2():
        # Intro
        img_pair = Frame(0)

        matches = img_pair.match()
        kp0, kp1 = img_pair.get_kps()

        # q1
        y_dist = list(map(lambda m: abs(kp0[m.queryIdx][1] - kp1[m.trainIdx][1]), matches))
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
        print("{:6.2f} %".format((1 - (3 / len(img_pair.img0.get_image()))) * 100))

        # q3
        k, m1, m2 = SlamEx.read_cameras()
        km1, km2 = k @ m1, k @ m2
        points = np.array([[kp0[m.queryIdx], kp1[m.trainIdx]] for m in matches])
        triangulated = np.array([SlamCompute.triangulate_pt(km1, km2, p0, p1) for p0, p1 in points])
        cv_triangulated = cv2.triangulatePoints(km1, km2, points[:, 0].T, points[:, 1].T).T
        cv_triangulated = cv_triangulated[:, :3] / cv_triangulated[:, 3:]

        Display.plot_3d(triangulated)
        Display.plot_3d(cv_triangulated)

    @staticmethod
    def ex3():
        # ----------------- 2.1 ------------------
        movie = SlamEx.create_movie()
        m1 = movie.stereo_dist
        im0, im1 = movie.add_frame(0), movie.add_frame(1)
        cloud1,cloud2 = im0.triangulate(movie.cam1, movie.cam2), im1.triangulate(movie.cam1, movie.cam2)
        Display.plot_3d(cloud1)
        Display.plot_3d(cloud2)

        # -------------- 2.2 matches --------------
        pair = ImagePair(im0.img0, im1.img0)
        matches = pair.match()
        Display.matches(pair, matches=matches)
        # ---------------- 2.3 PnP ----------------
        R, t = movie.pnp(1, matches, [0, 1, 2, 3, 4, 5, 6, 7])
        inv_R = np.linalg.inv(R)
        left_0 = np.zeros(3)
        right_0 = left_0 - m1
        left_1 = (left_0 @ inv_R.T) - t
        right_1 = ((- m1) @ inv_R.T) - t
        cameras = np.array(
            [[left_0[0], left_0[2]], [right_0[0], right_0[2]], [left_1[0], left_1[2]], [right_1[0], right_1[2]]])
        Display.scatter_2d(cameras)

        # ------------- 2.4 supporters -------------
        supporters = movie.find_supporters(0,1, matches, R, t)
        good_matches = matches[supporters]
        bad_matches = matches[np.logical_not(supporters)]
        good_kp0, good_kp1 = ImagePair.get_match_idx(good_matches)
        bad_kp0, bad_kp1 = ImagePair.get_match_idx(bad_matches)

        # Display.matches(pair, matches=good_matches)
        # Display.matches(pair, matches=bad_matches)
        # Display.kp_two_color(im0.img0, im0.img0.kp[good_kp0], im0.img0.kp[bad_kp0])
        # Display.kp_two_color(im1.img0, im1.img0.kp[good_kp1], im1.img0.kp[bad_kp1])

        # ------------- 2.5 RANSAC -------------

        best_R, best_t = movie.max_supporters_RANSAC(0,1, matches, 100)
        cloud1 = im0.triangulate(movie.cam1, movie.cam2)
        cloud2 = (best_R @ cloud1.T).T + best_t
        Display.plot_3d(cloud1, cloud2)

        supporters = movie.find_supporters(0,1, matches, R, t)

        good_matches = matches[supporters]
        bad_matches = matches[np.logical_not(supporters)]
        good_kp0, good_kp1 = ImagePair.get_match_idx(good_matches)
        bad_kp0, bad_kp1 = ImagePair.get_match_idx(bad_matches)

        Display.matches(pair, matches=good_matches)
        Display.matches(pair, matches=bad_matches)
        Display.kp_two_color(im0.img0, im0.img0.kp[good_kp0], im0.img0.kp[bad_kp0])
        Display.kp_two_color(im1.img0, im1.img0.kp[good_kp1], im1.img0.kp[bad_kp1])

        # ----------- 2.6 whole movie - ----------
        num = 100
        movie = SlamEx.create_movie()
        movie.add_frame(0)
        for i in range(1, num):
            print(i)
            movie.add_frame(i)
            movie.transformation(i)

        track1 = movie.get_t_positions(num)
        track2 = SlamEx.load_positions(num)

        Display.scatter_2d(track1[:, [0, 2]], track2[:, [0, 2]])

    @staticmethod
    def ex4():
        frame_num = 10  # FRAME_NUM
        movie = SlamEx.create_movie()
        movie.run(frame_num)

        # ------------------------- q2 ------------------------- #
        print("Total Number Of Tracks:", movie.tracks.get_size())
        print("Total Number Of Frames:", movie.get_size())
        print("Mean track length:", np.mean(movie.tracks.get_track_lengths()))
        print("Min track length:", np.min(movie.tracks.get_track_lengths()))
        print("Max track length:", np.max(movie.tracks.get_track_lengths()))
        print("Mean number of frame links", np.mean(list(map(lambda f: f.get_num_track(), movie.frames))))

        # ------------------------- q3 ------------------------- #
        track_length = np.array(list(map(lambda t: t.get_size(), movie.tracks)))
        big_track = movie.tracks[np.random.choice(np.argwhere(track_length > 3).squeeze(), 1)[0]]
        Display.track(movie, big_track.track_id, save=True)

        # ------------------------- q4 ------------------------- #
        connectivity = np.zeros(len(movie.frames) - 1)
        for t in movie.tracks:
            connectivity[t.get_frame_ids()[:-1]] += 1
        Display.simple_plot(connectivity, 'frames', 'outing tracks', 'connectivity', save=True)

        # ------------------------- q5 ------------------------- #
        inliers = list(map(lambda f: f.get_inlier_percentage(), movie.frames))
        Display.simple_plot(inliers[1:], 'frames', 'inlier percentage', 'pnp_inliers', save=True, max_y=1)

        # ------------------------- q6 ------------------------- #
        Display.hist(track_length, "track #", "track length", "track_length_histogram")

        # ------------------------- q7 ------------------------- #
        poses = SlamEx.load_poses(frame_num)
        last_frame_id = big_track.get_frame_ids()[-1]
        last_frame = movie.frames[last_frame_id]
        real_place_l = poses[last_frame_id]  # x,y,z
        real_place_r = real_place_l
        real_place_r[:, 3] += movie.stereo_dist
        cloud = last_frame.triangulate(movie.K @ real_place_l, movie.K @ real_place_r)
        position = cloud[big_track.get_match(last_frame_id)]
        error = lambda p1, p2: np.sum(((p1 - p2) ** 2)) ** 0.5
        err0 = []
        err1 = []
        frames = movie.get_frames(big_track.get_frame_ids())
        for frame in frames:
            px_l, px_r = position, position

            x1, x2, y = movie.get_track_location(frame.frame_id, big_track.track_id)
            error_l = error(np.array([x1, y]), px_l)
            error_r = error(np.array([x2, y]), px_r)
            err0.append(error_l)
            err1.append(error_r)
        fig, ax = plt.subplots()

        ax.plot(err0, label="left camera error")
        ax.plot(err1, label="right camera error")
        ax.legend()
        plt.title("error")

        plt.show()
