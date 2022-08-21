import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import gtsam
from gtsam.utils import plot
import scipy.stats
import math
from Image import Image

from ba import *
from Constants import *
from Display import Display
from Frame import Frame
from ImagePair import ImagePair
from SlamCompute import SlamCompute
from SlamMovie import SlamMovie
from SlamEx import SlamEx
from mpl_toolkits.mplot3d import Axes3D



from queue import PriorityQueue


def inf_matrix():
    mat = np.identity(6)
    for i in range(6):
        mat[i][i] = np.inf
    return mat


class Graph:
    def __init__(self, indx):
        self.v = indx
        self.edges = [[inf_matrix() for i in range(len(indx))] for j in range(len(indx))]
        self.cov = [inf_matrix() for i in range(len(indx))]
        self.son = [None for i in range(len(indx))]

    def add_edge(self, u, v, weight):
        self.edges[u][v] = weight
        self.edges[v][u] = weight


# indx, start_vertex
def dijkstra(graph):
    init_single_source(graph)
    S = []
    Q = [i for i in range(len(graph.v))]

    while Q:
        u = extract_min(Q, graph)
        S.append(u)

        for vertex in range(u, len(graph.v)):
            relax(u, vertex, graph)
        # print("u: ", u)
        # print("Q: ", Q)
        Q.remove(u)


def extract_min(Q, graph):
    min_det = np.linalg.det(graph.cov[Q[0]])
    min_i = Q[0]
    for i in Q:
        new_det = np.linalg.det(graph.cov[i])
        if new_det < min_det:
            min_det = new_det
            min_i = i
    return min_i


def relax(u, vertex, graph):
    # print("check:", u," ", vertex)
    if np.linalg.det(graph.cov[u]) >= np.linalg.det(graph.cov[vertex] + graph.edges[vertex][u]):
        graph.cov[u] = graph.cov[vertex] + graph.edges[vertex][u]
        graph.son[u] = vertex


def init_single_source(graph):
    for i in range(len(graph.v)):
        graph.cov[i] = graph.edges[len(graph.v) - 1][i]


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def find_short(arg_min, indx, movie):
    print("attemt to short: ", arg_min)
    st_indx, nd_indx = arg_min, indx[-1]

    movie.frames[st_indx] = Frame(st_indx)
    movie.frames[nd_indx] = Frame(nd_indx)

    im1, im2 = movie.frames[st_indx], movie.frames[nd_indx]
    pair = ImagePair(im1.img0, im2.img0)
    matches = pair.match()

    # im1, im2 = Image(st_indx, 0), Image(nd_indx, 0)
    # pair = ImagePair(im1, im2)
    # matches = pair.match(1)
    print(st_indx, nd_indx, len(matches))

    R, t, supporters = movie.max_supporters_RANSAC(st_indx, nd_indx, matches, 10000, num_of_point=4, p=0.8)
    score = sum(supporters) / len(matches)

    return score, R, t, supporters, st_indx, nd_indx, im1, im2, matches


def ex7(movie):
    successful = 0
    num_of_pose = FRAME_NUM
    movie, PoseGraph, PoseGraph_initial, PoseGraph_result = ex6(movie,num_of_pose)
    all_indx = []
    # all indx pose graph
    for i in range(num_of_pose):
        try:
            c0 = gtsam.symbol('c', i)
            pose_c0 = PoseGraph_result.atPose3(c0)
            all_indx.append(i)
        except:
            pass
    print("all indx: ", all_indx)

    loc_before, angle_before = location_angle_uncertainty(PoseGraph, PoseGraph_result, all_indx)

    # 2. absolute location error
    # loop over indx
    scores = []
    match_count = []
    for k in list(range(2, 116)):
        # initial graph
        indx = all_indx[0:k]
        print("##### last indx: ", indx[-1], " #####")
        graph = Graph(indx)

        # adding vertex
        marginals = gtsam.Marginals(PoseGraph, PoseGraph_result)
        for i in range(len(indx)):
            for j in range(len(indx)):
                conditionalCovariance = get_conditional_cov(marginals, indx[i], indx[j])
                graph.add_edge(i, j, 1000 * conditionalCovariance)

        # find all shortest path
        dijkstra(graph)
        ### find shortest path
        graph.son = np.array(graph.son)
        graph.son = np.where(graph.son == None, len(graph.son) - 1, graph.son)
        result_cov = [np.zeros((6, 6)) for i in range(len(graph.v))]
        result_prob = np.zeros(len(graph.v) - 1)
        for i in range(len(graph.son)):
            j = i
            while j != graph.son[-1]:
                result_cov[i] += graph.edges[j][graph.son[j]]
                j = graph.son[j]
        for i in range(len(graph.son) - 1):
            c0 = gtsam.symbol('c', graph.v[i])
            pose_c0 = PoseGraph_result.atPose3(c0)
            c1 = gtsam.symbol('c', graph.v[-1])
            pose_c1 = PoseGraph_result.atPose3(c1)
            relative_pose = pose_c0.between(pose_c1)
            t = relative_pose.translation()
            R = relative_pose.rotation().matrix()
            r = rotationMatrixToEulerAngles(R)
            delta_C = np.hstack((t, r))
            result_prob[i] = delta_C.T @ np.linalg.inv(result_cov[i]) @ delta_C
        # print("2: ", all_indx[np.argmin(result_prob)], np.min(result_prob))

        st_min, st_arg_min = np.min(result_prob), all_indx[np.argmin(result_prob)]
        nd_min, nd_arg_min = result_prob[0], 0
        for elem in range(len(result_prob)):
            if result_prob[elem] < nd_min and result_prob[elem] != st_min:
                nd_min = result_prob[elem]
                nd_arg_min = all_indx[elem]
        rd_min, rd_arg_min = result_prob[0], 0
        for elem in range(len(result_prob)):
            if result_prob[elem] < rd_min and result_prob[elem] != st_min and result_prob[elem] != nd_min:
                rd_min = result_prob[elem]
                rd_arg_min = all_indx[elem]

        if (st_min < 800000000):
            for arg_min in [st_arg_min, nd_arg_min, rd_arg_min]:
                score, R, t, supporters, st_indx, nd_indx, im1, im2, matches = find_short(arg_min, indx, movie)
                if score < LOOP_SCORE:
                    continue
                else:
                    scores.append(score)
                    match_count.append(len(matches))
                    print("score: ", score)
                    print("loop num matches", len(matches))
                    successful += 1
                    f_x = movie.K[0, 0]
                    f_y = movie.K[1, 1]
                    skew = movie.K[0, 1]
                    c_x = movie.K[0, 2]
                    c_y = movie.K[1, 2]
                    baseline = movie.stereo_dist[0]
                    K = gtsam.Cal3_S2Stereo(f_x, f_y, skew, c_x, c_y, -baseline)

                    ba_frames = [st_indx, nd_indx]
                    little_graph = gtsam.NonlinearFactorGraph()
                    little_initialEstimate = gtsam.Values()
                    uncertainty = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.00001, 0.00001, 0.00001, 1, 1, 1]))

                    st_leftCamPose = gtsam.Pose3(gtsam.Rot3(R_START), gtsam.Point3(T_START))
                    c0 = gtsam.symbol('c', st_indx)
                    pose_c = st_leftCamPose
                    little_initialEstimate.insert(c0, pose_c)
                    little_graph.add(
                        gtsam.PriorFactorPose3(c0, pose_c, gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1, 1, 1, 1]))))

                    reversed_relative_R = R.T
                    reversed_relative_t = -R.T @ t
                    nd_leftCamPose = gtsam.Pose3(gtsam.Rot3(reversed_relative_R), gtsam.Point3(reversed_relative_t))
                    c1 = gtsam.symbol('c', nd_indx)
                    pose_c = nd_leftCamPose
                    little_initialEstimate.insert(c1, pose_c)
                    little_graph.add(
                        gtsam.PriorFactorPose3(c1, pose_c, gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1, 1, 1, 1]))))

                    uncertainty = gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1]))

                    for m in range(len(matches)):
                        l = gtsam.symbol('l', m)
                        match = matches[m]
                        kp_l_x = im1.img0.kp[match.queryIdx][0]
                        kp_l_y = im1.img0.kp[match.queryIdx][1]
                        kp_r_x = im1.img1.kp[match.queryIdx][0]
                        sp1 = gtsam.StereoPoint2(kp_l_x, kp_r_x, kp_l_y)

                        kp_l_x = im2.img0.kp[match.trainIdx][0]
                        kp_l_y = im2.img0.kp[match.trainIdx][1]
                        kp_r_x = im2.img1.kp[match.trainIdx][0]
                        sp2 = gtsam.StereoPoint2(kp_l_x, kp_r_x, kp_l_y)

                        factor = gtsam.GenericStereoFactor3D(sp1, uncertainty, c0, l, K)
                        little_graph.add(factor)
                        factor = gtsam.GenericStereoFactor3D(sp2, uncertainty, c1, l, K)
                        little_graph.add(factor)

                        loc_l = gtsam.StereoCamera(nd_leftCamPose, K).backproject(sp2)
                        little_initialEstimate.insert(l, loc_l)
                        little_graph.add(gtsam.NonlinearEqualityPoint3(l, loc_l))

                    optimizer = gtsam.LevenbergMarquardtOptimizer(little_graph, little_initialEstimate)
                    little_result = optimizer.optimize()

                    marginals = gtsam.Marginals(little_graph, little_result)
                    conditionalCovariance = get_conditional_cov(marginals, nd_indx, st_indx)

                    c0 = gtsam.symbol('c', st_indx)
                    pose_c0 = little_result.atPose3(c0)
                    c1 = gtsam.symbol('c', nd_indx)
                    pose_c1 = little_result.atPose3(c1)
                    relative_pose = pose_c0.between(pose_c1)

                    PoseGraph.add(
                        gtsam.BetweenFactorPose3(c0, c1, relative_pose,
                                                 gtsam.noiseModel.Gaussian.Covariance(conditionalCovariance)))
                    optimizer = gtsam.LevenbergMarquardtOptimizer(PoseGraph, PoseGraph_result)
                    PoseGraph_result = optimizer.optimize()
                    marginals = gtsam.Marginals(PoseGraph, PoseGraph_result)

                    # print_path(azim=270, pose_graph=PoseGraph_result, title="{}_{}_a".format(st_indx, nd_indx), save="{}_{}_a.png".format(st_indx, nd_indx), cov=marginals)
                    # print_path(azim=270, pose_graph=PoseGraph_result,
                    #            title="{}_{}_b\n{}\n{}".format(st_indx, nd_indx, st_min, rd_min),
                    #            save="{}_{}_b.png".format(st_indx, nd_indx))

    print("successful: ", successful)

    # 1. location uncertainty
    loc_after, angle_after = location_angle_uncertainty(PoseGraph, PoseGraph_result, all_indx)
    plt.close()

    return PoseGraph_result, loc_before, angle_before, loc_after, angle_after, np.array(scores), np.array(match_count)


def location_angle_uncertainty(PoseGraph, PoseGraph_result, all_indx):
    # find cov location uncertainty
    plt.clf()
    marginals = gtsam.Marginals(PoseGraph, PoseGraph_result)
    cov_det = []
    location_uncertainty = []
    angle_uncertainty = []
    for ind in range(len(all_indx)):
        # marginals
        conditionalCovariance = get_conditional_cov(marginals, all_indx[0], all_indx[ind])
        cov_det.append(np.linalg.det(conditionalCovariance))
        location, angle = conditionalCovariance[0:3, 0:3], conditionalCovariance[3:6, 3:6]
        location_uncertainty.append(np.linalg.det(location)), angle_uncertainty.append(np.linalg.det(angle))

    return np.array(location_uncertainty), np.array(angle_uncertainty)
