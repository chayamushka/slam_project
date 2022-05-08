import pickle

import numpy as np

from Constants import *


class Tracks:
    class Track:
        def __init__(self, track_id):
            self.track_id = track_id
            self.frame_ids = []
            self.track_matches = dict()  # frame_id: match_id

        def extend_track(self, frame_id, match_id):
            self.frame_ids.append(frame_id)
            self.track_matches[frame_id] = match_id

        def get_id(self) -> int:
            return self.track_id

        def get_size(self):
            return len(self.frame_ids)

        def get_frame_ids(self):
            return self.frame_ids

        def is_last(self, frame_id):
            return len(self.frame_ids) == 0 or self.frame_ids[-1] == frame_id

        def get_match(self, frame_id: int) -> int:
            return self.track_matches[frame_id]

    def __init__(self):
        self.tracks = []
        self.cur_tracks = np.array([])  # TODO : change to use frame.get_track_ods()

    def __iter__(self):
        return iter(self.tracks)

    def __getitem__(self, item):
        return self.tracks[item]

    def get_size(self):
        return len(self.tracks)

    def get_track_lengths(self):
        return list(map(lambda t: t.get_size(), self.tracks))

    def get_track(self, track_id: int) -> Track:
        if track_id >= len(self.tracks):
            raise Exception(f"track id {track_id} wasn't found in given tracks man")
        return self.tracks[track_id]

    def get_track_ids(self, tracks=None):
        tracks = tracks if tracks is not None else self.tracks
        return list(map(lambda t: t.get_id(), tracks))

    def add_track(self, frame_id, match):
        track = self.Track(len(self.tracks))
        track.extend_track(frame_id - 1, match.queryIdx)
        track.extend_track(frame_id, match.trainIdx)
        self.tracks.append(track)
        return track

    def update_tracks(self, frame_id, matches):
        last_matches = np.array(list(map(lambda t: t.get_match(frame_id - 1), self.cur_tracks)))
        new_cur_tracks = []
        for m in matches:
            extend_tracks = last_matches == m.queryIdx
            for t in self.cur_tracks[extend_tracks]:
                t.extend_track(frame_id, m.trainIdx)
                new_cur_tracks.append(t)
            if not np.any(extend_tracks):
                track = self.add_track(frame_id, m)
                new_cur_tracks.append(track)
        self.cur_tracks = np.array(new_cur_tracks)
        return self.get_track_ids(new_cur_tracks)

    @staticmethod
    def save(tracks):
        with open(TRACK_FILE, 'wb') as f:
            pickle.dump(tracks, f)

    @staticmethod
    def load():
        with open(TRACK_FILE, 'rb') as f:
            return pickle.load(f)
