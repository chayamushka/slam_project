import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import gtsam
from gtsam.utils import plot
import scipy.stats
import math


from Constants import *
from Display import Display
from Frame import Frame
from ImagePair import ImagePair
from SlamCompute import SlamCompute
from SlamMovie import SlamMovie
from SlamEx import SlamEx
from mpl_toolkits.mplot3d import Axes3D

from ba import window_ba
from ba import ex6

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
        graph.cov[i] = graph.edges[len(graph.v)-1][i]


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def ex7():
    successful = 0
    num_of_pose = 1600
    movie, PoseGraph, PoseGraph_initial, PoseGraph_result = ex6(num_of_pose)
    all_indx = []

    # pose graph indx
    for i in range(num_of_pose):
        try:
            c0 = gtsam.symbol('c', i)
            pose_c0 = PoseGraph_result.atPose3(c0)
            all_indx.append(i)
        except:
            pass
    print("all indx: ", all_indx)

    # 1. location uncertainty
    marginals = gtsam.Marginals(PoseGraph, PoseGraph_result)
    cov_det = []
    for ind in range(len(all_indx)):
        # marginals
        keys = gtsam.KeyVector()
        keys.append(gtsam.symbol('c', all_indx[0]))
        keys.append(gtsam.symbol('c', all_indx[ind]))
        jointCovariance = marginals.jointMarginalCovariance(keys).fullMatrix()
        jointInformation = np.linalg.inv(jointCovariance)
        conditionalInformation = jointInformation[6:12, 6:12]
        conditionalCovariance = np.linalg.inv(conditionalInformation)
        cov_det.append(np.linalg.det(conditionalCovariance))
    plt.plot(np.array(all_indx), np.array(cov_det))
    plt.title("Det of Cov by Frame indx\nbefore Loop Closure")
    plt.xlabel("Frame indx")
    plt.ylabel("Det of Cov")
    plt.show()

    # 2. absolute location error








    fast_search = list(range(77, 82)) + list(range(160, 164)) + list(range(170, 173))
    # loop over indx
    for k in list(range(77, 82)):
        indx = all_indx[0:k]
        print("##### last indx: ", indx[-1], " #####")
        graph = Graph(indx)
        marginals = gtsam.Marginals(PoseGraph, PoseGraph_result)
        for i in range(len(indx)):
            for j in range(len(indx)):
                keys = gtsam.KeyVector()
                keys.append(gtsam.symbol('c', indx[i]))
                keys.append(gtsam.symbol('c', indx[j]))
                jointCovariance = marginals.jointMarginalCovariance(keys).fullMatrix()
                jointInformation = np.linalg.inv(jointCovariance)
                conditionalInformation = jointInformation[6:12, 6:12]
                conditionalCovariance = np.linalg.inv(conditionalInformation)
                graph.add_edge(i, j, conditionalCovariance)

        dijkstra(graph)
        graph.son = np.array(graph.son)
        graph.son = np.where(graph.son == None, len(graph.son)-1, graph.son)
        result_cov = [np.zeros((6,6)) for i in range(len(graph.v))]
        result_prob = np.zeros(len(graph.v)-1)
        for i in range(len(graph.son)):
            # print("start: ",i)
            j = i
            while j != graph.son[-1]:
                # print(j)
                result_cov[i] += graph.edges[j][graph.son[j]]
                j = graph.son[j]

        for i in range(len(graph.son)-1):
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
        # print("2: ", np.argmin(result_prob), np.min(result_prob))

        if np.min(result_prob) < 20000000:
            print("final close: ", indx[np.argmin(result_prob)])
            st_indx, nd_indx = indx[np.argmin(result_prob)], indx[-1]
            im1, im2 = movie.frames[st_indx], movie.frames[nd_indx]
            pair = ImagePair(im1.img0, im2.img0)
            matches = pair.match()
            R, t, supporters = movie.max_supporters_RANSAC(st_indx, nd_indx, matches, 10000, num_of_point=4, p=0.8)
            score = sum(supporters)/len(matches)
            print("score: ", score)
            if score > 0.05:
                successful += 1
                print("result_prob: ", np.min(result_prob))
                print("argmin: ", np.argmin(result_prob))
                print("score: ", score)
                print("matches: ", len(matches))

                kp0, kp1 = pair.get_kps()
                match0 = [match.queryIdx for match in matches]
                good_points0 = kp0[match0]
                bad_points0 = [kp for kp in kp0 if kp not in good_points0]

                match1 = [match.trainIdx for match in matches]
                good_points1 = kp1[match1]
                bad_points1 = [kp for kp in kp1 if kp not in good_points1]


                im_dis = cv2.drawKeypoints(im1.img0.image, good_points0, outImage=np.array([]), color=(0, 165, 255),
                                           flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                im_dis = cv2.drawKeypoints(im_dis, bad_points0, outImage=np.array([]), color=(255, 255, 0), )
                Display.end(im_dis, "kps_" + "txt", False)

                im_dis = cv2.drawKeypoints(im2.img0.image, good_points1, outImage=np.array([]), color=(0, 165, 255),
                                           flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                im_dis = cv2.drawKeypoints(im_dis, bad_points1, outImage=np.array([]), color=(255, 255, 0), )
                Display.end(im_dis, "kps_" + "txt", False)




                f_x = movie.K[0, 0]
                f_y = movie.K[1, 1]
                skew = movie.K[0, 1]
                c_x = movie.K[0, 2]
                c_y = movie.K[1, 2]
                baseline = movie.stereo_dist[0]
                K = gtsam.Cal3_S2Stereo(f_x, f_y, skew, c_x, c_y, -baseline)

                ba_frames = [st_indx, nd_indx]
                graph = gtsam.NonlinearFactorGraph()
                initialEstimate = gtsam.Values()
                uncertainty = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]))


                st_leftCamPose = gtsam.Pose3(gtsam.Rot3(np.identity(3)), gtsam.Point3(np.zeros(3)))
                c0 = gtsam.symbol('c', st_indx)
                pose_c = st_leftCamPose
                initialEstimate.insert(c0, pose_c)
                graph.add(
                    gtsam.PriorFactorPose3(c0, pose_c, gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1, 1, 1, 1]))))

                reversed_relative_R = R.T
                reversed_relative_t = -R.T @ t
                nd_leftCamPose = gtsam.Pose3(gtsam.Rot3(reversed_relative_R), gtsam.Point3(reversed_relative_t))
                c1 = gtsam.symbol('c', nd_indx)
                pose_c = nd_leftCamPose
                initialEstimate.insert(c1, pose_c)
                graph.add(
                    gtsam.PriorFactorPose3(c1, pose_c, gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1, 1, 1, 1]))))

                uncertainty = gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1]))

                for m in range(len(matches)):
                    l = gtsam.symbol('l', m)
                    match = matches[m]
                    kp_l_x = im1.img0.kp[match.queryIdx].pt[0]
                    kp_l_y = im1.img0.kp[match.queryIdx].pt[1]
                    kp_r_x = im1.img1.kp[match.queryIdx].pt[0]
                    sp1 = gtsam.StereoPoint2(kp_l_x, kp_r_x, kp_l_y)

                    kp_l_x = im2.img0.kp[match.trainIdx].pt[0]
                    kp_l_y = im2.img0.kp[match.trainIdx].pt[1]
                    kp_r_x = im2.img1.kp[match.trainIdx].pt[0]
                    sp2 = gtsam.StereoPoint2(kp_l_x, kp_r_x, kp_l_y)

                    factor = gtsam.GenericStereoFactor3D(sp1, uncertainty, c0, l, K)
                    graph.add(factor)
                    factor = gtsam.GenericStereoFactor3D(sp2, uncertainty, c1, l, K)
                    graph.add(factor)

                    loc_l = gtsam.StereoCamera(nd_leftCamPose, K).backproject(sp2)
                    initialEstimate.insert(l, loc_l)
                    graph.add(gtsam.NonlinearEqualityPoint3(l, loc_l))

                optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
                result = optimizer.optimize()

                marginals = gtsam.Marginals(graph, result)
                keys = gtsam.KeyVector()
                keys.append(gtsam.symbol('c', st_indx))
                keys.append(gtsam.symbol('c', nd_indx))
                jointCovariance = marginals.jointMarginalCovariance(keys).fullMatrix()
                jointInformation = np.linalg.inv(jointCovariance)
                conditionalInformation = jointInformation[6:12, 6:12]
                conditionalCovariance = np.linalg.inv(conditionalInformation)

                c0 = gtsam.symbol('c', st_indx)
                pose_c0 = result.atPose3(c0)
                c1 = gtsam.symbol('c', nd_indx)
                pose_c1 = result.atPose3(c1)
                relative_pose = pose_c0.between(pose_c1)

                PoseGraph.add(
                gtsam.BetweenFactorPose3(c0, c1, relative_pose, gtsam.noiseModel.Gaussian.Covariance(conditionalCovariance)))
                optimizer = gtsam.LevenbergMarquardtOptimizer(PoseGraph, PoseGraph_initial)
                PoseGraph_result = optimizer.optimize()

                fig = plt.figure()
                ax = Axes3D(fig)
                ax.view_init(elev=0, azim=270)
                plot.plot_trajectory(1, PoseGraph_result, scale=1)
                gtsam.utils.plot.set_axes_equal(1)
                plt.show()

                marginals = gtsam.Marginals(PoseGraph, PoseGraph_result)
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.view_init(elev=0, azim=270)
                plot.plot_trajectory(1, PoseGraph_result, marginals=marginals, scale=1)
                gtsam.utils.plot.set_axes_equal(1)
                plt.show()

        if indx[-1] == 1521: # 1481 # 1901, 2861
                st_indx, nd_indx = 81, 1521
                print("___check___")
                c0 = gtsam.symbol('c', st_indx)
                pose_c0 = PoseGraph_initial.atPose3(c0)
                c1 = gtsam.symbol('c', nd_indx)
                pose_c1 = PoseGraph_initial.atPose3(c1)
                # relative_pose = pose_c0.between(pose_c1)
                R = np.identity(3)
                t = np.zeros(3)
                relative_pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t))

                PoseGraph.add(
                    gtsam.BetweenFactorPose3(c0, c1, relative_pose,
                                             gtsam.noiseModel.Diagonal.Sigmas(np.array([0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]))))
                optimizer = gtsam.LevenbergMarquardtOptimizer(PoseGraph, PoseGraph_initial)
                PoseGraph_result = optimizer.optimize()

                fig = plt.figure()
                ax = Axes3D(fig)
                ax.view_init(elev=0, azim=270)
                plot.plot_trajectory(1, PoseGraph_result, scale=1)
                gtsam.utils.plot.set_axes_equal(1)
                plt.show()
                #
                marginals = gtsam.Marginals(PoseGraph, PoseGraph_result)
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.view_init(elev=0, azim=270)
                plot.plot_trajectory(1, PoseGraph_result, marginals=marginals, scale=1)
                gtsam.utils.plot.set_axes_equal(1)
                plt.show()
        if indx[-1] == 3241:  # 1481 # 1901, 3241
            st_indx, nd_indx = 1901, 3241
            print("___check___")
            c0 = gtsam.symbol('c', st_indx)
            pose_c0 = PoseGraph_initial.atPose3(c0)
            c1 = gtsam.symbol('c', nd_indx)
            pose_c1 = PoseGraph_initial.atPose3(c1)
            # relative_pose = pose_c0.between(pose_c1)
            R = np.identity(3)
            t = np.zeros(3)
            relative_pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t))

            PoseGraph.add(
                gtsam.BetweenFactorPose3(c0, c1, relative_pose,
                                         gtsam.noiseModel.Diagonal.Sigmas(
                                             np.array([0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]))))
            optimizer = gtsam.LevenbergMarquardtOptimizer(PoseGraph, PoseGraph_initial)
            PoseGraph_result = optimizer.optimize()

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(elev=0, azim=270)
            plot.plot_trajectory(1, PoseGraph_result, scale=1)
            gtsam.utils.plot.set_axes_equal(1)
            plt.show()
            #
            marginals = gtsam.Marginals(PoseGraph, PoseGraph_result)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(elev=0, azim=270)
            plot.plot_trajectory(1, PoseGraph_result, marginals=marginals, scale=1)
            gtsam.utils.plot.set_axes_equal(1)
            plt.show()
        if indx[-1] == 3421:  # 661, 3441
            st_indx, nd_indx = 681, 3421
            print("___check___")
            c0 = gtsam.symbol('c', st_indx)
            pose_c0 = PoseGraph_initial.atPose3(c0)
            c1 = gtsam.symbol('c', nd_indx)
            pose_c1 = PoseGraph_initial.atPose3(c1)
            # relative_pose = pose_c0.between(pose_c1)
            R = np.identity(3)
            t = np.zeros(3)
            relative_pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t))

            PoseGraph.add(
                gtsam.BetweenFactorPose3(c0, c1, relative_pose,
                                         gtsam.noiseModel.Diagonal.Sigmas(
                                             np.array([0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]))))
            optimizer = gtsam.LevenbergMarquardtOptimizer(PoseGraph, PoseGraph_initial)
            PoseGraph_result = optimizer.optimize()

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(elev=0, azim=270)
            plot.plot_trajectory(1, PoseGraph_result, scale=1)
            gtsam.utils.plot.set_axes_equal(1)
            plt.show()
            #
            marginals = gtsam.Marginals(PoseGraph, PoseGraph_result)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(elev=0, azim=270)
            plot.plot_trajectory(1, PoseGraph_result, marginals=marginals, scale=1)
            gtsam.utils.plot.set_axes_equal(1)
            plt.show()

    print("successful: ", successful)
    # 1. location uncertainty
    marginals = gtsam.Marginals(PoseGraph, PoseGraph_result)
    cov_det = []
    for ind in range(len(all_indx)):
        keys = gtsam.KeyVector()
        keys.append(gtsam.symbol('c', all_indx[0]))
        keys.append(gtsam.symbol('c', all_indx[ind]))
        jointCovariance = marginals.jointMarginalCovariance(keys).fullMatrix()
        jointInformation = np.linalg.inv(jointCovariance)
        conditionalInformation = jointInformation[6:12, 6:12]
        conditionalCovariance = np.linalg.inv(conditionalInformation)
        cov_det.append(np.linalg.det(conditionalCovariance))
    plt.plot(np.array(all_indx), np.array(cov_det))
    plt.title("Det of Cov by Frame indx\nafter Loop Closure")
    plt.xlabel("Frame indx")
    plt.ylabel("Det of Cov")
    plt.show()

    # 2. absolute location error








