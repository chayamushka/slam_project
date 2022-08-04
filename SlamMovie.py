import cv2
import numpy as np
import pickle
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
        cam1, cam2 = np.c_[np.eye(3), np.zeros(3)], np.c_[np.eye(3), np.zeros(3)]
        cam2[:, 3] += self.stereo_dist
        self.cam1, self.cam2 = K @ cam1, K @ cam2

    def add_frame(self, idx: int):
        frame = Frame(idx)
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
        return pt0[0], pt1[0], pt1[1]

    def transformation(self, frame_id: int):
        if frame_id > self.frames.size - 1:
            raise Exception(
                f"Can't calculate the transformation of frame {frame_id}, we have only {len(self.frames)} frames")
        if len(self.frames) == 1:
            return np.eye(3), np.zeros(3)
        im1, im2 = self.frames[frame_id - 1], self.frames[frame_id]
        pair = ImagePair(im1.img0, im2.img0)
        matches = pair.match()
        R, t, supporters = self.max_supporters_RANSAC(frame_id - 1, frame_id, matches, 100)
        im2.set_relative_position(R, t, supporters)
        # R = im1.R @ R
        # t = im1.R @ t + im1.t
        # im2.set_position(R, t, supporters)
        im2.set_tracks_ids(self.tracks.update_tracks(frame_id, matches[supporters]))
        return R, t

    def run(self, num=FRAME_NUM):
        for i in range(num):
            print(i)
            self.add_frame(i)
            self.transformation(i)

    def get_t_positions(self, n=FRAME_NUM):
        R, t = self.get_non_relative_Rt(n)
        return np.array(t) * [1, 1, -1]

    def get_non_relative_Rt(self, n=FRAME_NUM):
        R = [self.frames[0].R_relative, ]
        t = [self.frames[0].t_relative]
        for f in self.frames[1:n]:
            R.append(R[-1] @ f.R_relative)
            t.append(R[-1] @ f.t_relative + t[-1])
        return R, t

    def pnp(self, point_cloud, idx, matches, ind):
        queries, trains = ImagePair.get_match_idx(matches[ind])
        points_3d = point_cloud[queries]
        key_points = self.frames[idx].get_kps()[0][trains]
        pixels = np.expand_dims(key_points, axis=1)
        _, rvec, tvec = cv2.solvePnP(points_3d, pixels, self.K, np.zeros((4, 1), dtype=np.float64))
        R = cv2.Rodrigues(rvec)[0]
        return R, tvec.squeeze()

    def find_supporters(self, first_ind, idx, matches, R, t, pixels=2):
        threshold = lambda ptx, px: np.all(np.abs(ptx - px) <= pixels)
        supporters = np.full(len(matches), True)
        pair1, pair2 = self.frames[first_ind], self.frames[idx]
        points = pair1.triangulate(self.cam1, self.cam2)
        for i, match in enumerate(matches):
            point =points[match.queryIdx]
            left0 = threshold(SlamCompute.projection(point, self.K), pair1.img0.get_kp_pt(match.queryIdx))
            right0 = threshold(SlamCompute.projection(point + self.stereo_dist, self.K),
                               pair1.img1.get_kp_pt(match.queryIdx))
            left1 = threshold(SlamCompute.projection(point, self.K, R, t), pair2.img0.get_kp_pt(match.trainIdx))
            right1 = threshold(SlamCompute.projection(point + self.stereo_dist, self.K, R, t),
                               pair2.img1.get_kp_pt(match.trainIdx))
            supporters[i] = left0 and left1 and right0 and right1
        return supporters

    def max_supporters_RANSAC(self, first_ind, idx, matches, max_loop=10000, min_loop=20, num_of_point=6, p=0.99):
        num_max_supporters = 0
        ransac_size = 100
        count_loop = 0
        best_R, best_t = np.eye(3), np.zeros(3)
        max_supporters = np.full(len(matches), False)
        point_cloud = self.frames[first_ind].triangulate(self.cam1, self.cam2)
        while ransac_size + min_loop - count_loop > 0 and count_loop < max_loop:
            ind = np.random.choice(len(matches), size=num_of_point, replace=len(matches) < num_of_point)
            try:
                R, t = self.pnp(point_cloud, idx, matches, ind)
                supporters = self.find_supporters(first_ind, idx, matches, R, t)
                if sum(supporters) > num_max_supporters:
                    num_max_supporters, max_supporters = sum(supporters), supporters
                    best_R, best_t = R, t
                    ransac_size = SlamCompute.ransac_loop(p, num_max_supporters / len(matches), num_of_point)
                    print("ransac_size: ", ransac_size)
            except:
                pass
            count_loop += 1
        # print("before: ", num_max_supporters / len(matches), ": ", num_max_supporters, "/", len(matches))
        # while True:
        #     try:
        #         R, t = self.pnp(idx, matches, max_supporters)
        #         supporters = self.find_supporters(idx, matches, R, t)
        #         if sum(supporters) <= num_max_supporters:
        #             break
        #         num_max_supporters, max_supporters = sum(supporters), supporters
        #         best_R, best_t = R, t
        #     except:
        #         break
        # print("after: ", num_max_supporters / len(matches), ": ", num_max_supporters, "/", len(matches))
        return best_R, best_t, max_supporters

    @staticmethod
    def save(movie):
        # decrease movie size:
        for frame in movie.frames:
            frame.img0.des = None
            frame.img1.des = None
        with open(MOVIE_FILE, 'wb') as f:
            pickle.dump(movie, f)

    @staticmethod
    def load():
        with open(MOVIE_FILE, 'rb') as f:
            return pickle.load(f)
