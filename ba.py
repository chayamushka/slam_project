import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import gtsam
import gtsam.utils.plot as gtsam_plot



from Constants import *
from Display import Display
from Frame import Frame
from ImagePair import ImagePair
from SlamCompute import SlamCompute
from SlamMovie import SlamMovie
from SlamEx import SlamEx




def window_ba(movie, ba_frames, initialEstimate, graph, visited_tracks, uncertainty, K):
    first_frame = True
    for frame in movie.frames[ba_frames]:
        if first_frame:
            first_frame = False
            # R = frame.R_relative
            # t = frame.t_relative
            R = np.identity(3)
            t = np.zeros(3)
            last_R = R
            last_t = t
        else:
            R = last_R @ frame.R_relative
            t = last_R @ frame.t_relative + last_t
            last_R = R
            last_t = t

        reversed_relative_R = R.T
        reversed_relative_t = -R.T @ t

        leftCamPose = gtsam.Pose3(gtsam.Rot3(reversed_relative_R), gtsam.Point3(reversed_relative_t))
        c = gtsam.symbol('c', frame.frame_id)
        pose_c = leftCamPose
        initialEstimate.insert(c, pose_c)
        graph.add(gtsam.PriorFactorPose3(c, pose_c, gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1,1, 1, 1]))))

        for track_id in frame.tracks:
            if  len(list(set(movie.tracks[track_id].frame_ids) & set(ba_frames))) < 2:
                continue
            l = gtsam.symbol('l', track_id)
            match = movie.tracks[track_id].track_matches[frame.frame_id]
            kp_l_x = frame.img0.kp[match].pt[0]
            kp_l_y = frame.img0.kp[match].pt[1]
            kp_r_x = frame.img1.kp[match].pt[0]
            # kp_r_y = movie.frames[frame_id].img1.kp[match].pt[1]
            sp = gtsam.StereoPoint2(kp_l_x, kp_r_x, kp_l_y)

            if track_id not in visited_tracks:
                visited_tracks.append(track_id)
                loc_l = gtsam.StereoCamera(leftCamPose, K).backproject(sp)
                initialEstimate.insert(l, loc_l)
                graph.add(gtsam.NonlinearEqualityPoint3(l, loc_l))

            factor = gtsam.GenericStereoFactor3D(sp, uncertainty, c, l, K)
            graph.add(factor)


def get_array_from_values(object, symbol, values):
    result = []

    for i in range(len(object)):
        key = gtsam.symbol(symbol,i)
        try:
            if symbol == "l":
                result.append(values.atPoint3(key))
            else:
                pose = values.atPose3(key)
                result.append(pose.transformFrom((0,0,0)))
        except:
            continue
    return np.array(result)

def show_track_and_points(movie, values):
    points = get_array_from_values(movie.tracks.tracks, "l", values)
    pose = get_array_from_values(movie.frames, "c", values)
    plt.scatter(points[:, 0], points[:, 2], c='lightblue')
    plt.scatter(pose[:, 0], pose[:, 2], c="red")
    plt.ylim(-100, 500)
    plt.xlim(-300, 400)
    plt.show()





def ex5():
    frame_num = 3449 # FRAME_NUM
    movie = SlamEx.create_movie()
    movie.run(frame_num)

    f_x = movie.K[0, 0]
    f_y = movie.K[1, 1]
    skew = movie.K[0, 1]
    c_x = movie.K[0, 2]
    c_y = movie.K[1, 2]
    baseline = movie.stereo_dist[0]
    K = gtsam.Cal3_S2Stereo(f_x, f_y, skew, c_x, c_y, -baseline)
    uncertainty = gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1]))

    print("# ------------- Start 5.1 ------------- #")
    """
    print("# ------------- execute 5.1 ------------- #")
    track_length = np.array(list(map(lambda t: t.get_size(), movie.tracks)))
    big_track = movie.tracks[np.random.choice(np.argwhere(track_length > 10).squeeze(), 1)[0]]
    


    frames_list = []
    print(np.sort(big_track.frame_ids))
    for frame_id in np.sort(big_track.frame_ids):
        R = movie.frames[frame_id].R
        t = movie.frames[frame_id].t
        reversed_R = R.T
        reversed_t = -R.T @ t
        leftCamPose = gtsam.Pose3(gtsam.Rot3(reversed_R), gtsam.Point3(reversed_t))
        frames_list.append((frame_id,big_track.track_matches[frame_id], gtsam.StereoCamera(leftCamPose, K), leftCamPose))

    frame_id, match, sc, leftCamPose= frames_list[-1]
    kp_l_x = movie.frames[frame_id].img0.kp[match].pt[0]
    kp_l_y = movie.frames[frame_id].img0.kp[match].pt[1]
    kp_r_x = movie.frames[frame_id].img1.kp[match].pt[0]
    # kp_r_y = movie.frames[frame_id].img1.kp[match].pt[1]
    sp = gtsam.StereoPoint2(kp_l_x,kp_r_x,kp_l_y)
    point3 = frames_list[-1][2].backproject(sp)

    reprojection_error = []

    graph = gtsam.NonlinearFactorGraph()
    initialEstimate = gtsam.Values()

    l = gtsam.symbol('l', 1)
    loc_l = point3
    initialEstimate.insert(l, loc_l)
    graph.add(gtsam.NonlinearEqualityPoint3(l, loc_l))

    for frame in frames_list:
        frame_id, match, sc, leftCamPose = frame
        print("frame id:", frame_id)
        estimate_stereoPoint2 = sc.project(point3)
        kp_l_x = movie.frames[frame_id].img0.kp[match].pt[0]
        kp_l_y = movie.frames[frame_id].img0.kp[match].pt[1]
        kp_r_x = movie.frames[frame_id].img1.kp[match].pt[0]
        # kp_r_y = movie.frames[frame_id].img1.kp[match].pt[1]
        real_stereoPoint2 = gtsam.StereoPoint2(kp_l_x, kp_r_x, kp_l_y)
        left_error = np.linalg.norm(estimate_stereoPoint2.vector()[[0,2]] - real_stereoPoint2.vector()[[0,2]])
        right_error = np.linalg.norm(estimate_stereoPoint2.vector()[[1, 2]] - real_stereoPoint2.vector()[[1, 2]])

        c = gtsam.symbol('c', frame_id)
        pose_c = leftCamPose
        initialEstimate.insert(c, pose_c)

        graph.add(gtsam.NonlinearEqualityPose3(c, pose_c))
        factor = gtsam.GenericStereoFactor3D(real_stereoPoint2, uncertainty, c, l, K)
        graph.add(factor)
        factor_error = factor.error(initialEstimate)
        reprojection_error.append([frame_id, left_error, right_error, factor_error])
    print("len: ",len(reprojection_error))
    print(reprojection_error)
    gtsam_plot.plot_trajectory(1, initialEstimate)
    gtsam_plot.set_axes_equal(1)
    plt.show()

    reprojection_error = np.array(reprojection_error)

    plt.plot(reprojection_error[:,2], reprojection_error[:,3])
    plt.show()
    plt.plot(reprojection_error[:,1])
    plt.show()
    plt.plot(reprojection_error[:,2])
    plt.show()
    plt.plot(reprojection_error[:,3])
    plt.show()
    """
    print("# ------------- End 5.1 ------------- #")

    # ------------------------- 5.2 ------------------------- #
    # exit()
    print("# ------------- Start 5.2 ------------- #")
    """
    print("# ------------- execute 5.2 ------------- #")
    visited_tracks = []
    ba_start_frame = 1
    keyframes_lens = 20
    ba_frames = list(range(ba_start_frame, ba_start_frame+keyframes_lens))
    graph = gtsam.NonlinearFactorGraph()
    initialEstimate = gtsam.Values()

    window_ba(movie, ba_frames, initialEstimate, graph, visited_tracks, uncertainty, K)

    show_track_and_points(movie, initialEstimate)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
    print("error before: ", optimizer.error())
    result = optimizer.optimize()
    print("error after: ", optimizer.error())
    show_track_and_points(movie, result)


    # gtsam_plot.plot_trajectory(1, initialEstimate)
    # gtsam_plot.set_axes_equal(1)
    # plt.show()
    #
    # gtsam_plot.plot_trajectory(1, result)
    # gtsam_plot.set_axes_equal(1)
    # plt.show()




    """
    print("# ------------- End 5.2 ------------- #")

    print("# ------------- Start 5.3 ------------- #")
    # """
    print("# ------------- execute 5.3 ------------- #")
    ba_start_frame = 1
    keyframes_lens = 20
    last_frame = frame_num - 1

    result_keyPoint = gtsam.Values()
    result_just_keyPoint = gtsam.Values()
    keyframe_list = []
    for i in range(ba_start_frame, frame_num, keyframes_lens):

        ba_frames = list(range(i, min(i + keyframes_lens + 1, last_frame)))
        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()

        visited_tracks = []  # ?????
        print(ba_frames)
        window_ba(movie, ba_frames, initialEstimate, graph, visited_tracks, uncertainty, K)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        result = optimizer.optimize()

        if i != 1:
            first_pose = result_keyPoint.atPose3(gtsam.symbol('c', i))
        for j in range(i+1, min(i + keyframes_lens + 1, last_frame)):
            c = gtsam.symbol('c',j)
            pose_c = result.atPose3(c)
            pose = first_pose * pose_c if i != 1 else pose_c
            result_keyPoint.insert(c, pose)
        keyframe_list.append(j)
        result_just_keyPoint.insert(c, pose)
        for k in range(len(movie.tracks.tracks)):
            key = gtsam.symbol("l", k)
            try:
                result_keyPoint.insert(key, result.atPoint3(key))
            except:
                continue



    # plot 1
    show_track_and_points(movie, result_keyPoint)



    # plot 2
    pose = get_array_from_values(movie.frames, "c", result_keyPoint)
    track1 = movie.get_positions(frame_num)
    track2 = SlamEx.load_positions(frame_num)
    Display.scatter_2d(track1[:, [0, 2]], track2[:, [0, 2]], pose[:, [0, 2]], ("initial","truth","optimizer"))

    # plot 3
    pose = get_array_from_values(movie.frames, "c", result_just_keyPoint)
    track2 = SlamEx.load_positions(frame_num)[keyframe_list]
    distance = np.sum((pose - track2)**2,axis=1)**0.5

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(distance)
    plt.title("distance by keyPoint")
    plt.show()

    # """
    print("# ------------- End 5.3 ------------- #")







