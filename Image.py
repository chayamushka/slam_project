import cv2

DATA_PATH = "dataset\\sequences\\00\\"


class Image:
    def __init__(self, idx, side):
        img_name = '{:06d}.png'.format(idx)
        self.image = cv2.imread(f'{DATA_PATH}image_{side}\\{img_name}', 0)
        if self.image is None:
            raise Exception(f"image {DATA_PATH}image_{side}\\{img_name} doesn't exist in current path")
        self.idx = idx
        self.side = side

    def detect_kp_compute_des(self, orb):
        self.kp, self.des = orb.detectAndCompute(self.image, None)
        return self.kp, self.des

    def get_des(self):
        return self.des

    def get_kp(self):
        return self.kp

    def get_kp_pt(self, idx):
        return self.kp[idx].pt

    def set_des(self, des):
        self.des = des

    def set_kp(self, kp):
        self.kp = kp
