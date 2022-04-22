import numpy as np

from Constants import *
from Display import Display
from ImagePair import ImagePair
from SlamMovie import SlamMovie
from SlamEx import SlamEx


def q3():
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
    num = 300
    movie = SlamMovie.Movie()
    movie.add_pair(0)
    for i in range(1, num):
        print(i)
        movie.add_pair(i)
        movie.transformation(i - 1)

    track1 = Display.show_track(movie.transformations, num)
    track2 = SlamEx.load_poses(num)

    Display.plot_2d(track1[:, [0, 2]], track2[:, [0, 2]])


if __name__ == '__main__':
    q3()
