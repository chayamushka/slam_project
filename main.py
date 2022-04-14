
import numpy as np
from SlamMovie import SlamMovie
from Constants import *
from Display import Display
from ImagePair import ImagePair


def q3():
    # ----------------- 2.1 ------------------
    k, m0, m1 = SlamMovie.read_cameras()
    km0, km1 = k @ m0, k @ m1
    im0 = ImagePair.StereoPair(0, cam1=km0, cam2=km1)
    im1 = ImagePair.StereoPair(1, cam1=km0, cam2=km1)
    Display.plot_3d(im0.points_cloud)
    Display.plot_3d(im1.points_cloud)

    # -------------- 2.2 matches --------------
    pair = ImagePair(im0.img0, im1.img0)
    matches = pair.match(update_des=False)
    Display.matches(pair, matches=matches)

    # ---------------- 2.3 PnP ----------------
    R, t = SlamMovie.pnp(im0, im1, k, matches, [0, 1, 2, 3, 4, 5, 6, 7])
    inv_R = np.linalg.inv(R)
    left_0 = np.zeros(3)
    right_0 = left_0 - m1[:, 3]
    left_1 = (left_0 @ inv_R.T) - t
    right_1 = ((- m1[:, 3]) @ inv_R.T) - t
    cameras = np.array(
        [[left_0[0], left_0[2]], [right_0[0], right_0[2]], [left_1[0], left_1[2]], [right_1[0], right_1[2]]])

    Display.plot_2d(cameras)

    # ------------- 2.4 supporters -------------
    supporters = SlamMovie.findSupporters(im0, im1, k, m1[:, 3], matches, R, t)

    good_matches = matches[supporters]
    bad_matches = matches[np.logical_not(supporters)]
    good_kp0 = list(map(lambda m: m.queryIdx, good_matches))
    bad_kp0 = list(map(lambda m: m.queryIdx, bad_matches))
    good_kp1 = list(map(lambda m: m.trainIdx, good_matches))
    bad_kp1 = list(map(lambda m: m.trainIdx, bad_matches))

    for i in range(0):
        Display.matches(pair, matches=good_matches)
        Display.matches(pair, matches=bad_matches)
        Display.kp_two_color(im0.img0, im0.img0.kp[good_kp0], im0.img0.kp[bad_kp0])
        Display.kp_two_color(im1.img0, im1.img0.kp[good_kp1], im1.img0.kp[bad_kp1])

    # ------------- 2.5 RANSAC -------------

    best_R, best_t = SlamMovie.max_supporters_RANSAC(im0, im1, k, m1[:, 3], matches, 100)
    cloud1 = im0.points_cloud
    cloud2 = (best_R @ im0.points_cloud) + best_t
    Display.plot_3d(cloud1, cloud2)

    supporters = SlamMovie.findSupporters(im0, im1, k, m1[:, 3], matches, R, t)

    good_matches = matches[supporters]
    bad_matches = matches[np.logical_not(supporters)]
    good_kp0 = list(map(lambda m: m.queryIdx, good_matches))
    bad_kp0 = list(map(lambda m: m.queryIdx, bad_matches))
    good_kp1 = list(map(lambda m: m.trainIdx, good_matches))
    bad_kp1 = list(map(lambda m: m.trainIdx, bad_matches))

    for i in range(0):
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
    track2 = Display.show_track(np.loadtxt(GT_POSES), num, 0)

    Display.plot_2d(track1, track2)


# def ex1():
#     # q1
#     img_pair = ImagePair.StereoPair(0)
#
#     img_pair.feature_descriptors(MAX_NUM_FEATURES)
#     Display.kp(img_pair.img0, "kp img1.jpg")
#     Display.kp(img_pair.img1, "kp img2.jpg")
#
#     # q2
#     des0, des1 = img_pair.get_des()
#     print("descriptor 0:", des0[0], "\ndescriptor 1:", des1[0])
#
#     # q3
#     # Match features.
#     img_pair.old_match(sort=True)
#     Display.matches(img_pair, txt="matches_pair_0.jpg")
#
#     # q4
#     img_pair.match()
#     bad_matches = list(
#         filter(lambda x: x not in img_pair.significant_matches, map(lambda x: x[0], img_pair.knn_matches)))
#
#     print("good ratio", len(img_pair.significant_matches) / 1000)
#     Display.matches(img_pair, txt="significant_matches.jpg", matches=img_pair.significant_matches)
#     Display.matches(img_pair, matches=[bad_matches[-1]], num_matches=1)
#
#
# def ex2():
#     # Intro
#     img_pair = ImagePair.StereoPair(0)
#     img_pair.feature_descriptors(MAX_NUM_FEATURES)
#     img_pair.match()
#     kp0, kp1 = img_pair.get_kps()
#     matches = img_pair.significant_matches
#
#     # q1
#     y_dist = list(map(lambda m: abs(kp0[m.queryIdx].pt[1] - kp1[m.trainIdx].pt[1]), matches))
#     h = Display.hist(y_dist, save=True)
#     deviate_2 = sum(h[3:]) * 100 / sum(h)
#     print("{:6.2f} %".format(deviate_2))
#
#     # q2
#     y_dist = np.array(y_dist)
#     matches = np.array(matches)
#     accepted, rejected = matches[y_dist <= 2], matches[y_dist > 2]
#     accepted_idx0, rejected_idx0 = np.array(list(map(lambda m: m.queryIdx, accepted))), np.array(
#         list(map(lambda m: m.queryIdx, rejected)))
#     accepted_idx1, rejected_idx1 = np.array(list(map(lambda m: m.trainIdx, accepted))), np.array(
#         list(map(lambda m: m.trainIdx, rejected)))
#
#     kp0, kp1 = np.array(kp0), np.array(kp1)
#     Display.kp_two_color(img_pair.img0, kp0[accepted_idx0], kp0[rejected_idx0], save=True)
#     Display.kp_two_color(img_pair.img1, kp1[accepted_idx1], kp1[rejected_idx1], save=True)
#     print("discarded: ", len(rejected))
#     # Assuming the Y-coordinate of erroneous matches is distributed uniformly across the
#     # image, what ratio of matches would you expect to be wrong with this rejection policy
#     print("{:6.2f} %".format((1 - (3 / len(img_pair.img0.image))) * 100))
#
#     # q3
#     k, m1, m2 = read_cameras()
#     km1, km2 = k @ m1, k @ m2
#     points = np.array([[kp0[m.queryIdx].pt, kp1[m.trainIdx].pt] for m in matches])
#     triangulated = np.array([SlamCompute.triangulate_pt(km1, km2, p0, p1) for p0, p1 in points])
#     cv_triangulated = cv2.triangulatePoints(km1, km2, points[:, 0].T, points[:, 1].T).T
#     cv_triangulated = cv_triangulated[:, :3] / cv_triangulated[:, 3:]
#
#     Display.plot_3d(triangulated)
#     Display.plot_3d(cv_triangulated)

#
# def projected(point, kp_idx, left, k, R, t, pair, stereo_dist, tresh=2):
#     kp = pair.img0.kp[kp_idx].pt if left else pair.img1.kp[kp_idx].pt
#     pt = SlamCompute.projection(point - stereo_dist,k, R, t)
#     # pt = k @ ((R @ (point - stereo_dist)) + t)
#     # pt = pt / pt[2]
#     return not (abs(pt[0] - kp[0]) > tresh or abs(pt[1] - kp[1]) > tresh)


#
# def readStereo(indx, MAX_NUM_FEATURES, km0, km1):
#     img = ImagePair.StereoPair(indx)
#     img.feature_descriptors(MAX_NUM_FEATURES)
#     img.match()
#     img.stereo_filter()
#     img.triangulate(km0, km1)
#     return img

#
# def PnP(pair1, pair2, cam, matches, idx):
#     points_3d = pair1.points_cloud[list(map(lambda m: m.queryIdx, matches[idx]))]
#     key_points = pair2.get_kps()[0][list(map(lambda m: m.trainIdx, matches[idx]))]
#     pixels = np.expand_dims(cv2.KeyPoint.convert(key_points), axis=1)
#     _, rvec, tvec = cv2.solvePnP(points_3d, pixels, cam[:, :3], np.zeros((4, 1), dtype=np.float64))
#     R = cv2.Rodrigues(rvec)[0]
#     return R, tvec.squeeze()

#
# def findSupporters(cam1, cam2, K, m, matches, R, t):
#     thresh = lambda ptx, px: np.all(np.abs(ptx - px) <= 2)
#     supporters = np.full(len(matches),True)
#     for i,match in enumerate(matches):
#         point = cam1.points_cloud[match.queryIdx]
#         left0 = thresh(SlamCompute.projection(point, K), cam1.img0.get_kp_pt(match.queryIdx))
#         right0 = thresh(SlamCompute.projection(point + m, K), cam1.img1.get_kp_pt(match.queryIdx))
#         left1 = thresh(SlamCompute.projection(point, K, R, t), cam2.img0.get_kp_pt(match.trainIdx))
#         right1 = thresh(SlamCompute.projection(point + m, K, R, t), cam2.img1.get_kp_pt(match.trainIdx))
#         supporters[i] = left0 and left1 and right0 and right1
#     return supporters

#
# def max_supporters_RANSAC(objectCam, relativeCam, k, m1, matches, max_loop=10000, min_loop=200, num_of_point=6, p=0.99):
#     num_max_supporters = 0
#     ransac_size = 0
#     count_loop = 0
#     best_R, best_t = np.eye(3), np.zeros(3)
#     while ransac_size + min_loop - count_loop > 0 and count_loop < max_loop:
#         ind = np.random.choice(len(matches), size=num_of_point, replace=len(matches) < num_of_point)
#         try:
#             R, t = SlamMovie.pnp(objectCam, relativeCam, k, matches, ind)
#             supporters = SlamMovie.findSupporters(objectCam, relativeCam, k, m1, matches, R, t)
#             if sum(supporters) > num_max_supporters:
#                 num_max_supporters, max_supporters = sum(supporters), supporters
#                 best_R, best_t = R, t
#                 ransac_size = SlamCompute.ransac_loop(p, num_max_supporters / len(matches), num_of_point)
#                 print("ransac_size: ", ransac_size)
#         except:
#             pass
#
#         count_loop += 1
#     print("before: ", num_max_supporters / len(matches), ": ", num_max_supporters, "/", len(matches))
#     while True:
#         try:
#             R, t = SlamMovie.pnp(objectCam, relativeCam, k, matches, max_supporters)
#             supporters = SlamMovie.findSupporters(objectCam, relativeCam, k, m1, matches, R, t)
#             if sum(supporters) > num_max_supporters:
#                 num_max_supporters, max_supporters = sum(supporters), supporters
#                 best_R, best_t = R, t
#             else:
#                 break
#         except:
#             break
#     print("after: ", num_max_supporters / len(matches), ": ", num_max_supporters, "/", len(matches))
#     return [best_R, best_t]
#
#
# correct = np.eye(3)
# correct[2, 2] = -1
#
#
# def showTreck(pose, num, type):
#     result = [[0, 0, 0]]
#     for i in range(1, num):
#         if type:
#             R = pose[i][0]
#             t = pose[i][1]
#             result.append(correct @ np.add(R @ (0, 0, 0), t))
#         else:
#             R = pose[i].reshape((3, 4))[:, :3]
#             t = pose[i].reshape((3, 4))[:, 3]
#             result.append(np.linalg.inv(R) @ (-t))
#
#     cameras = np.array(result)
#     # print(cameras[:,[0,2]])
#     Display.plot_2d(cameras[:, [0, 2]])


if __name__ == '__main__':
    q3()
