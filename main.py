from time import time
start = time()
from ba import ex6
from SlamEx import SlamEx

from SlamMovie import *

if __name__ == '__main__':
    # ex6()
    print("time", (time() - start))
    # SlamEx.go()
    movie = SlamMovie.load()
    g = 3
    g+=3
    r = g

