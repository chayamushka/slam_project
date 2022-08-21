from Display import Display
import numpy as np
from SlamMovie import SlamMovie
from SlamCompute import SlamCompute
from SlamEx import SlamEx
import cv2
from loopCl import rotationMatrixToEulerAngles
from Constants import *


class Statistics:
    @staticmethod
    def projection_error(movie: SlamMovie, track_id: int):

        track = movie.tracks[track_id]
        frames_id = track.get_frame_ids()
        frame_0_id = frames_id[0]
        frame_0 = movie.frames[frame_0_id]
        pt0 = frame_0.img0.get_kp(track[frame_0_id])
        pt1 = frame_0.img1.get_kp(track[frame_0_id])

        position = cv2.triangulatePoints(movie.cam1, movie.cam2, pt0, pt1)
        position = (position[:3] / position[3:]).squeeze()
        left_error = [0, ]
        right_error = [0, ]
        # -- left cam --
        Rl, tl = SlamMovie.get_non_relative_Rt(movie.frames[frames_id[1:]])
        Rr, tr = SlamMovie.get_non_relative_Rt(movie.frames[frames_id[1:]], movie.stereo_dist)
        # TODO fix right camera
        for (Rli, tli, Rri, tri) in zip(Rl, tl, Rr, tr):
            proj_pt_l = SlamCompute.projection(position, movie.K, Rli, tli)
            proj_pt_r = SlamCompute.projection(position, movie.K, Rri, tri)
            left_error.append(np.sqrt(np.sum(np.abs(np.array(pt0) - proj_pt_l) ** 2)))
            right_error.append(np.sqrt(np.sum(np.abs(np.array(pt1) - proj_pt_r) ** 2)))
        return left_error, right_error

    @staticmethod
    def pose_graph_estimation_error(movie: SlamMovie):
        pose_graph = SlamEx.load_poses()
        positions = SlamEx.load_positions()
        estimated_R, estimated_t = SlamMovie.get_non_relative_Rt(movie.frames)
        estimated_positions = movie.get_t_positions()
        xyz = np.array([0, 0, 0, 0])
        angle_err = np.array([0])
        for p, ep, Rt, eR, et in zip(positions, estimated_positions, pose_graph, estimated_R, estimated_t):
            xyz = np.vstack((xyz, np.append(np.abs(p - ep), np.linalg.norm(p - ep))))
            angle_err = np.vstack((angle_err, np.linalg.norm(rotationMatrixToEulerAngles(Rt[:, :3]) - rotationMatrixToEulerAngles(eR))))
            # TODO how to get angle error

        return xyz, angle_err


def median_projection_error(movie, txt="PnP", length=40, ba_result=None):
    movie.update_poses(ba_result)
    big_tracks = np.argwhere(movie.tracks.get_track_lengths() > length).squeeze()
    left_errors, right_errors = np.array(Statistics.projection_error(movie, big_tracks[0]))[:, :length]
    for i in big_tracks[1:]:
        lerrors, rerrors = np.array(Statistics.projection_error(movie, i))[:, :length]
        left_errors = np.vstack((left_errors, lerrors))
        right_errors = np.vstack((right_errors, rerrors))
    lerror_median = np.median(left_errors, axis=0)
    rerror_median = np.median(right_errors, axis=0)
    Display.simple_plot(np.stack((lerror_median, rerror_median)).T, "Frame in Track",
                        ["Left Median Error Projection", "Right Median Error Projection"],
                        f"Median Error Projection For {length} Tracks-{txt}",
                        True)


def bundle_median_projection_error(movie:SlamMovie, txt="PnP", length=40, ba_result=None):
    movie.update_poses(ba_result)

    big_tracks = np.argwhere(movie.tracks.get_track_lengths() > length).squeeze()
    left_errors, right_errors = np.array(Statistics.projection_error(movie, big_tracks[0]))[:, :length]
    for i in big_tracks[1:]:
        lerrors, rerrors = np.array(Statistics.projection_error(movie, i))[:, :length]
        left_errors = np.vstack((left_errors, lerrors))
        right_errors = np.vstack((right_errors, rerrors))
    lerror_median = np.median(left_errors, axis=0)
    rerror_median = np.median(right_errors, axis=0)
    Display.simple_plot(np.stack((lerror_median, rerror_median)).T, "Frame in Track",
                        ["Left Median Error Projection", "Right Median Error Projection"],
                        f"Median Error Projection For {length} Tracks-{txt}",
                        True)


def graphs(movie: SlamMovie):
    # -- statistics --
    print("Total Number Of Tracks:", len(movie.tracks))
    print("Total Number Of Frames:", len(movie.frames))
    print("Mean track length:", np.mean(movie.tracks.get_track_lengths()))
    connectivity = [sum([not movie.tracks[t].is_last(frame.frame_id) for t in frame.tracks]) for frame in movie.frames]
    print("Mean number of frame links:", np.mean(connectivity))

    # -- Number of Matches per Frame --
    nmatches = [len(f.matches) for f in movie.frames]
    Display.simple_plot(nmatches, "frame", "Matches per Frame", "Number of Matches per Frame", True)

    # -- Matches Inlier Percentage per Frame --
    inliers = [f.supporter_ratio for f in movie.frames]
    Display.simple_plot(inliers[1:], "frame", "Inlier Percentage", "Matches Inlier Percentage per Frame", True, 1)

    # -- Number of Links To The Next Frame per Frame --
    Display.simple_plot(connectivity[1:-1], "frame", "outgoing tracks",
                        "Connectivity", True)

    # -- Track Length Histogram
    Display.hist(movie.tracks.get_track_lengths(), "Tracks", "Track's length", "Track Length Histogram", True)


def pose_graph_estimation_error(movie, txt="out"):
    xyz, angle = Statistics.pose_graph_estimation_error(movie)
    Display.simple_plot(xyz, "Frame",
                        ["x", "y", "z", "norm"],
                        f"Pose Graph x,y,z estimation error with{txt} loop closure",
                        True, np.max(xyz))
    Display.simple_plot(angle, "Frame",
                        "Angle Error",
                        f"Pose Graph angle estimation error with{txt} loop closure",
                        True, np.max(angle))


def loop_closer_graphs(movie):
    from loopCl import ex7

    _, loc_before, angle_before, loc_after, angle_after, scores, match_count = ex7(movie)
    Display.simple_plot(loc_before, "keyframe", "uncertainty before loop closure",
                        "Location uncertainty before", True, max(loc_before))
    Display.simple_plot(loc_after, "keyframe", "uncertainty after loop closure",
                        "Location uncertainty after", True, max(loc_after))
    Display.simple_plot(angle_before, "keyframe",
                        "uncertainty before loop closure",
                        "Angle uncertainty before", True, max(angle_before))
    Display.simple_plot(angle_after, "keyframe",
                        "uncertainty after loop closure",
                        "Angle uncertainty after", True, max(angle_after))
    print("scores: ", scores)
    print("matches:", match_count)
    Display.simple_bars((100 * scores).astype(int), 'loop', "inlier percentage", "Inlier percentage per loop closure",
                        True, 100)
    Display.simple_bars(match_count, 'loop', "matches", "matches per loop closure", True)
