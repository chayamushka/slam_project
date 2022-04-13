import cv2
import numpy as np

from Constants import *
from Display import Display
from ImagePair import ImagePair
from SlamCompute import SlamCompute
from math import log

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


def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def projected(point, kp_idx, left, k, R, t, pair, stereo_dist, tresh=2):
    kp = pair.img0.kp[kp_idx].pt if left else pair.img1.kp[kp_idx].pt
    pt = k @ ((R @ (point - stereo_dist)) + t)
    pt = pt / pt[2]
    # 468,468 - 127,127. 461,461 - 127,127. 466,468 - 125,127. 459,461 - 126,127.
    a = pair.points_cloud[kp_idx]
    return not (abs(pt[0] - kp[0]) > tresh or abs(pt[1] - kp[1]) > tresh)


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
    k, m1, m2 = read_cameras()
    km1, km2 = k @ m1, k @ m2
    points = np.array([[kp0[m.queryIdx].pt, kp1[m.trainIdx].pt] for m in matches])
    triangulated = np.array([SlamCompute.triangulate_pt(km1, km2, p0, p1) for p0, p1 in points])
    cv_triangulated = cv2.triangulatePoints(km1, km2, points[:, 0].T, points[:, 1].T).T
    cv_triangulated = cv_triangulated[:, :3] / cv_triangulated[:, 3:]

    Display.plot_3d(triangulated)
    Display.plot_3d(cv_triangulated)


def readStereo(indx, MAX_NUM_FEATURES, km0, km1):
    img = ImagePair.StereoPair(indx)
    img.feature_descriptors(MAX_NUM_FEATURES)
    img.match()
    img.stereo_filter()
    img.triangulate(km0, km1)
    return img


def PnP(pair1, pair2, cam, matches, idx):
    points_3d = pair1.points_cloud[list(map(lambda m: m.queryIdx, matches[idx]))]
    key_points = pair2.get_kps()[0][list(map(lambda m: m.trainIdx, matches[idx]))]
    pixels = np.expand_dims(cv2.KeyPoint.convert(key_points), axis=1)
    _, rvec, tvec = cv2.solvePnP(points_3d, pixels, cam[:, :3], np.zeros((4, 1), dtype=np.float64))
    R = cv2.Rodrigues(rvec)[0]
    return R, tvec.squeeze()


def findSupporters(objectCam, relativeCam, km, m, matches, R, t):
    supporters = []
    for match in matches:
        point = objectCam.points_cloud[match.queryIdx]
        left0 = projected(point, match.queryIdx, True, km, np.eye(3), [0, 0, 0], objectCam, [0, 0, 0])
        right0 = projected(point, match.queryIdx, False, km, np.eye(3), [0, 0, 0], objectCam, -m[:, 3])
        left1 = projected(point, match.trainIdx, True, km, R, t, relativeCam, [0, 0, 0])
        right1 = projected(point, match.trainIdx, False, km, R, t, relativeCam, -m[:, 3])
        supporters.append(left0 and left1 and right0 and right1)
    return supporters


def max_supporters_RANSAC(objectCam, relativeCam, km1, k, m1,  matches, max_loop = 10000 ,minimum_loop = 200, num_of_point = 6, p=0.99):
    num_max_supporters = 0
    ransac_size = 0
    count_loop = 0
    while ransac_size + minimum_loop - count_loop > 0 and count_loop < max_loop:
        try:
            ind = list(np.random.choice(list(range(len(matches))), size =num_of_point, replace=False))
        except ValueError:
            ind = list(np.random.choice(list(range(len(matches))), size=num_of_point, replace=True))
        try:
            R, t = PnP(objectCam, relativeCam, km1, matches, ind)
            supporters = findSupporters(objectCam, relativeCam, k, m1, matches, R, t)
            if sum(supporters) > num_max_supporters:
                num_max_supporters = sum(supporters)
                best_R, best_t = R, t
                max_supporters = supporters
                outliers_prob = 1 - num_max_supporters/len(matches)
                ransac_size = log(1-p)/log(1-pow((1-outliers_prob),num_of_point))
                print("ransac_size: ", ransac_size)
        except:
            pass
        count_loop += 1
    if count_loop == max_loop:
        print("max_loop")
    print("before: ", num_max_supporters/len(matches), ": ", num_max_supporters,"/",len(matches))
    while True:
        try:
            R, t = PnP(objectCam, relativeCam, km1, matches, max_supporters)
            supporters = findSupporters(objectCam, relativeCam, k, m1, matches, R, t)
            if sum(supporters) > num_max_supporters:
                num_max_supporters = sum(supporters)
                best_R, best_t = R, t
                max_supporters = supporters
            else:
                break
        except:
            break
    print("after: ", num_max_supporters/len(matches), ": ", num_max_supporters,"/",len(matches))
    return [best_R, best_t]

correct = np.eye(3)
correct[2,2] = -1

def showTreck(pose, num, type):
    result = [[0,0,0]]
    for i in range(1, num):
        if type:
            R = pose[i][0]
            t = pose[i][1]
            result.append(correct @ np.add(R @ (0,0,0), t))
        else:
            R = pose[i].reshape((3, 4))[:, :3]
            t = pose[i].reshape((3, 4))[:, 3]
            result.append(np.linalg.inv(R) @ (-t))

    cameras = np.array(result)
    # print(cameras[:,[0,2]])
    Display.plot_2d(cameras[:,[0,2]])


def q3():
    ############# 2.1 ##############
    images_list = []
    k, m0, m1 = read_cameras()
    km0, km1 = k @ m0, k @ m1

    for i in range(2):
        images_list.append(ImagePair.StereoPair(i, cam1=km0, cam2=km1))

    ############# 2.2 matches ##############
    pair = ImagePair(images_list[0].img0, images_list[1].img0)
    matches = pair.match(update_des=False)

    ############# 2.3 PnP ##############
    R, t = PnP(images_list[0], images_list[1], km1, matches, [0, 1, 2, 3, 4, 5, 6, 7])

    imagePoints = images_list[1].points_cloud[list(map(lambda m: m.trainIdx, matches[0:4]))]

    left_0 = np.add((0, 0, 0) @ R.T, t)
    left_1 = [0, 0, 0]
    right_0 = np.add(((0, 0, 0) - m1[:, 3]) @ R.T, t)
    right_1 = left_1 - m1[:, 3]
    cameras = np.array(
        [[left_0[0], left_0[2]], [right_0[0], right_0[2]], [left_1[0], left_1[2]], [right_1[0], right_1[2]]])
    Display.plot_2d(cameras)

    inv_R = np.linalg.inv(R)
    left_0 = [0, 0, 0]
    right_0 = left_0 - m1[:, 3]
    left_1 = np.add(left_0 @ inv_R.T, -t)
    right_1 = np.add((- m1[:, 3]) @ inv_R.T, -t)

    cameras = np.array(
        [[left_0[0], left_0[2]], [right_0[0], right_0[2]], [left_1[0], left_1[2]], [right_1[0], right_1[2]]])
    Display.plot_2d(cameras)

    # 2.4 supporters
    supporters = findSupporters(images_list[0], images_list[1], k, m1, matches, R, t)

    good_matches = matches[supporters]
    bad_matches = matches[np.logical_not(supporters)]
    good_kp0 = list(map(lambda m: m.queryIdx, good_matches))
    bad_kp0 = list(map(lambda m: m.queryIdx, bad_matches))
    good_kp1 = list(map(lambda m: m.trainIdx, good_matches))
    bad_kp1 = list(map(lambda m: m.trainIdx, bad_matches))

    # for i in range(40):
    #     print(len(bad_kp0))
    #     print(len(good_kp1))
    #     print(len(good_kp0))
    #     Display.kp_two_color(images_list[0].img0, images_list[0].img0.kp[good_kp0], images_list[0].img0.kp[bad_kp0])
    #     Display.kp_two_color(images_list[1].img0, images_list[1].img0.kp[good_kp1], images_list[1].img0.kp[bad_kp1])

    # # 2.5 RANSAC
    # best_R, best_t = max_supporters_RANSAC(images_list[0], images_list[1], km1, k, m1, matches, 1)
    #
    # max_supporters = 0
    # for i in range(1):
    #     ind = list(np.random.randint(len(matches), size =4))
    #     R, t = PnP(images_list[0], images_list[1], km1, matches, [1,2,3,4])
    #     supporters = findSupporters(images_list[0], images_list[1], k, m1, matches, R, t)
    #     if len(supporters) > max_supporters:
    #         max_supporters = len(supporters)
    #         best_R, best_t = R, t

    # 2.6 whole movie
    images_list = []
    cameras_R_t = []
    num = 2760
    num = 300
    images_list.append(readStereo(0, MAX_NUM_FEATURES, km0, km1))
    cameras_R_t.append((np.eye(0), np.zeros(3)))
    for i in range(1, num):
        print(i)
        images_list.append(readStereo(i, MAX_NUM_FEATURES, km0, km1))
        # if abs(images_list[i].points_cloud.min()) >1000 or abs(images_list[i].points_cloud.max())>1000:
        #    print(images_list[i].points_cloud.min(), images_list[i].points_cloud.max())
        pair = ImagePair(images_list[i - 1].img0, images_list[i].img0)
        matches = pair.match(update_des=False)
        cameras_R_t.append(max_supporters_RANSAC(images_list[i - 1], images_list[i], km1, k, m1, matches, 100))
    #
    for i in range(2, num):
        cameras_R_t[i][0] = cameras_R_t[i - 1][0] @ cameras_R_t[i][0]
        cameras_R_t[i][1] = cameras_R_t[i - 1][0] @ cameras_R_t[i][1] + cameras_R_t[i - 1][1]
    #
    showTreck(cameras_R_t, num, 1)

    pose = np.loadtxt("dataset\\poses\\00.txt")
    showTreck(pose, num, 0)


if __name__ == '__main__':
    q3()
