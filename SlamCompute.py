import numpy as np


class SlamCompute:

    @staticmethod
    def triangulate_pt(c1, c2, p1, p2):
        r1 = c1[2] * p1[0] - c1[0]
        r2 = c1[2] * p1[1] - c1[1]
        r3 = c2[2] * p2[0] - c2[0]
        r4 = c1[2] * p2[1] - c2[1]
        return SlamCompute.svd_decomposition(np.vstack((r1, r2, r3, r4)))

    @staticmethod
    def svd_decomposition(A):
        __, _, v = np.linalg.svd(A)
        X = v[-1]
        return X[:-1] / X[-1]