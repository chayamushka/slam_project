from time import time

import GraphSlam

start = time()
from Display import Display
from SlamMovie import *
from loopCl import ex7
from ba import ex6
from GraphSlam import *


def bundle(movie):
    track1 = movie.get_t_positions()
    track2 = SlamEx.load_positions()
    Display.scatter_2d(track1[:, [0, 2]], track2[:, [0, 2]])
    GraphSlam.median_projection_error(movie, txt="PnP")
    movie, PoseGraph, PoseGraph_initial, result = ex6(movie)
    movie.update_poses(result)
    track1 = movie.get_t_positions()
    track2 = SlamEx.load_positions()
    Display.scatter_2d(track1[:, [0, 2]], track2[:, [0, 2]])
    GraphSlam.median_projection_error(movie, "Bundle Adjustment")

if __name__ == '__main__':
    # TODO change in transformation n Movie, the update tracks from im2 to im1
    movie = SlamMovie.load()
    track1 = movie.get_t_positions()
    track2 = SlamEx.load_positions()
    Display.scatter_2d(track1[:, [0, 2]], track2[:, [0, 2]], label = "after PnP")
    result, _, _, _, _,_, _ = ex7(movie)
    movie.update_poses(result)
    track1 = movie.get_t_positions()
    Display.scatter_2d(track1[:, [0, 2]], track2[:, [0, 2]], label = "After Loop Closer")







    # print("time", (time() - start))



