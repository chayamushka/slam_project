import cv2
import numpy as np

from Constants import *
from Frame import Frame
from ImagePair import ImagePair
from SlamCompute import SlamCompute
from Track import Track


class SlamMovie:
    def __init__(self, K, stereo_dist):
        self.frames = []
        self.track_n = 0

        self.tracks = []
        self.K, self.stereo_dist = K, stereo_dist

    @classmethod
    def Movie(cls):
        K, _, stereo_dist = SlamMovie.read_cameras()
        movie = cls(K, stereo_dist[:, 3])
        return movie

    def add_pair(self, idx: int):
        cam1, cam2 = np.c_[np.eye(3), np.zeros(3)], np.c_[np.eye(3), np.zeros(3)]
        cam2[:, 3] += self.stereo_dist
        frame = Frame(idx, self.K @ cam1, self.K @ cam2)
        self.frames.append(frame)
        return frame

    def get_frame(self, frame_id) -> Frame:
        return self.frames[frame_id]

    def get_track(self, track_id) -> Track:
        if track_id not in self.tracks.keys():
            raise Exception(f"track id {track_id} wasn't found in given tracks man")
        return self.tracks[track_id]

    def get_location(self, frame_id, track_id):
        frame = self.get_frame(frame_id)
        track = self.get_track(track_id)
        pt0 = frame.img0.get_kp_pt(track.get_match_id(frame_id))
        pt1 = frame.img1.get_kp_pt(track.get_match_id(frame_id))
        return pt0[0], *pt1

    def transformation(self, frame_id):
        if frame_id > len(self.frames) - 1:
            raise Exception(
                f"Can't calculate the transformation of frame {frame_id}, we have only {len(self.frames)} frames")
        if frame_id == 0:
            R, t = np.eye(3), np.zeros(3)

            return R, t
        im1, im2 = self.frames[frame_id - 1], self.frames[frame_id]
        pair = ImagePair(im1.img0, im2.img0)
        matches = pair.match()

        R, t = self.max_supporters_RANSAC(frame_id, matches, 100)
        # supporters = matches[self.find_supporters(frame_id, matches, R, t)]
        # TODO update tracks!
        R = im1.R @ R
        t = im1.R @ t + im1.t
        im2.set_position(R, t)
        return R, t

    def update_tracks(self, frame_id, matches):
        # TODO write more efficient code that
        #  will not require filtering all the track for each match

        for m in matches:
            tracks = list(filter(lambda t: t.is_last(frame_id - 1, m.queryIdx), self.tracks))
            for t in tracks:
                t.add_frame(frame_id, m.trainIdx)
            self.add_track(frame_id, m) if not len(tracks) else None

    def add_track(self, frame_id, match):
        track = Track(self.track_n)
        self.track_n += 1
        track.add_frame(frame_id - 1, match.queryIdx)
        track.add_frame(frame_id, match.trainIdx)
        self.tracks.append(track)

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

    def get_positions(self, n):
        return np.array(list(map(lambda f: f.t, self.frames[:n]))) * [1, 1, -1]

    def pnp(self, idx, matches, ind):
        points_3d = self.frames[idx - 1].points_cloud[list(map(lambda m: m.queryIdx, matches[ind]))]
        key_points = self.frames[idx].get_kps()[0][list(map(lambda m: m.trainIdx, matches[ind]))]
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
        return [best_R, best_t]
