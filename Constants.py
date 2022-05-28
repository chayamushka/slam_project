import os.path
MAX_NUM_FEATURES = 1000
DATA_PATH = os.path.join("dataset", "sequences", "00")

CAMERA_PATH = os.path.join(DATA_PATH, 'calib.txt')
GT_POSES = os.path.join("dataset", "poses", "00.txt")

FRAME_NUM = 3449
TRACK_FILE = "TracksOfLove.SLAM"
SIGNIFICANCE_RATIO = 0.9
