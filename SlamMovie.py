import cv2
import numpy as np

from Constants import *
from Frame import Frame
from ImagePair import ImagePair
from SlamCompute import SlamCompute
from Tracks import Tracks


class SlamMovie:
    def __init__(self, K, stereo_dist):
        self.frames = np.array([])
        self.tracks = Tracks()
        self.K, self.stereo_dist = K, stereo_dist

    def add_frame(self, idx: int):
        cam1, cam2 = np.c_[np.eye(3), np.zeros(3)], np.c_[np.eye(3), np.zeros(3)]
        cam2[:, 3] += self.stereo_dist

        frame = Frame(idx, self.K @ cam1, self.K @ cam2)
        self.frames = np.append(self.frames, [frame])
        return frame

    def get_frames(self, frame_id):
        return self.frames[frame_id]

    def get_size(self) -> int:
        return len(self.frames)

    def get_track_location(self, frame_id: int, track_id: int):
        # TODO test this one!
        frame = self.get_frames(frame_id)
        track = self.tracks.get_track(track_id)
        pt0 = frame.img0.get_kp_pt(track.get_match(frame_id))
        pt1 = frame.img1.get_kp_pt(track.get_match(frame_id))
        return pt0[0], *pt1

    def transformation(self, frame_id: int):
        if frame_id > self.frames.size - 1:
            raise Exception(
                f"Can't calculate the transformation of frame {frame_id}, we have only {len(self.frames)} frames")
        if len(self.frames) == 1:
            return np.eye(3), np.zeros(3)
        im1, im2 = self.frames[frame_id - 1], self.frames[frame_id]
        pair = ImagePair(im1.img0, im2.img0)
        matches = pair.match()

        R, t, supporters = self.max_supporters_RANSAC(frame_id, matches, 100)
        R = im1.R @ R
        t = im1.R @ t + im1.t
        im2.set_position(R, t, supporters)
        im2.set_tracks_ids(self.tracks.update_tracks(frame_id, matches[supporters]))
        return R, t

    def run(self, num=FRAME_NUM):
        for i in range(num):
            print(i)
            self.add_frame(i)
            self.transformation(i)

    def get_positions(self, n):
        return np.array(list(map(lambda f: f.t, self.frames[:n]))) * [1, 1, -1]

    def pnp(self, idx, matches, ind):
        queries, trains = ImagePair.get_match_idx(matches[ind])
        points_3d = self.frames[idx - 1].points_cloud[queries]
        key_points = self.frames[idx].get_kps()[0][trains]
        pixels = np.expand_dims(cv2.KeyPoint.convert(key_points), axis=1)
        _, rvec, tvec = cv2.solvePnP(points_3d, pixels, self.K, np.zeros((4, 1), dtype=np.float64))
        R = cv2.Rodrigues(rvec)[0]
        return R, tvec.squeeze()

    def find_supporters(self, idx, matches, R, t, pixels=2):
        threshold = lambda ptx, px: np.all(np.abs(ptx - px) <= pixels)
        supporters = np.full(len(matches), True)
        pair1, pair2 = self.frames[idx - 1], self.frames[idx]
        for i, match in enumerate(matches):
            point = pair1.points_cloud[match.queryIdx]
            left0 = threshold(SlamCompute.projection(point, self.K), pair1.img0.get_kp_pt(match.queryIdx))
            right0 = threshold(SlamCompute.projection(point + self.stereo_dist, self.K),
                               pair1.img1.get_kp_pt(match.queryIdx))
            left1 = threshold(SlamCompute.projection(point, self.K, R, t), pair2.img0.get_kp_pt(match.trainIdx))
            right1 = threshold(SlamCompute.projection(point + self.stereo_dist, self.K, R, t),
                               pair2.img1.get_kp_pt(match.trainIdx))
            supporters[i] = left0 and left1 and right0 and right1
        return supporters

    def max_supporters_RANSAC(self, idx, matches, max_loop=10000, min_loop=200, num_of_point=6, p=0.99):
        num_max_supporters = 0
        ransac_size = 0
        count_loop = 0
        best_R, best_t = np.eye(3), np.zeros(3)
        max_supporters = np.full(len(matches), False)
        while ransac_size + min_loop - count_loop > 0 and count_loop < max_loop:
            ind = np.random.choice(len(matches), size=num_of_point, replace=len(matches) < num_of_point)
            try:
                R, t = self.pnp(idx, matches, ind)
                supporters = self.find_supporters(idx, matches, R, t)
                if sum(supporters) > num_max_supporters:
                    num_max_supporters, max_supporters = sum(supporters), supporters
                    best_R, best_t = R, t
                    ransac_size = SlamCompute.ransac_loop(p, num_max_supporters / len(matches), num_of_point)
                    print("ransac_size: ", ransac_size)
            except:
                pass
            count_loop += 1
        print("before: ", num_max_supporters / len(matches), ": ", num_max_supporters, "/", len(matches))
        while True:
            try:
                R, t = self.pnp(idx, matches, max_supporters)
                supporters = self.find_supporters(idx, matches, R, t)
                if sum(supporters) <= num_max_supporters:
                    break
                num_max_supporters, max_supporters = sum(supporters), supporters
                best_R, best_t = R, t
            except:
                break
        print("after: ", num_max_supporters / len(matches), ": ", num_max_supporters, "/", len(matches))
        return best_R, best_t, max_supporters
