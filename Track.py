class Track:
    def __init__(self, track_id):
        self.track_id = track_id
        self.frame_ids= []
        self.track_matches = dict()  # frame_id: match_id

    def add_frame(self, frame_id, match_id):
        self.frame_ids.append(frame_id)
        self.track_matches[frame_id] = match_id

    def get_frame_ids(self):
        return list(map(lambda t: t.frame_id, self.track_matches.keys()))

    def get_match_id(self, frame_id):
        return self.track_matches[frame_id]

    def last_track(self):
        if not len(self.frame_ids):
            return None
        return self.frame_ids[-1]

    def is_last(self,frame_id, match_id) -> bool:
        if not len(self.frame_ids):
            return False
        return self.frame_ids[-1] == frame_id and self.track_matches[frame_id] == match_id
