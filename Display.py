import cv2
import matplotlib.pyplot as plt
import numpy as np

from Image import Image
from ImagePair import ImagePair


class Display:

    @staticmethod
    def kp(img: Image, txt="img.jpg", save=False):
        im_dis = cv2.drawKeypoints(img.image, img.kp, outImage=np.array([]), color=(255, 0, 0),
                                   flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        Display.end(im_dis, "kp_" + txt, save)

    @staticmethod
    def kp_two_color(im: Image, kp0, kp1, txt="kp_2_color.jpg", save=False):
        im_dis = cv2.drawKeypoints(im.image, kp0, outImage=np.array([]), color=(0, 165, 255),
                                   flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        im_dis = cv2.drawKeypoints(im_dis, kp1, outImage=np.array([]), color=(255, 255, 0), )

        Display.end(im_dis, "kps_" + txt, save)

    @staticmethod
    def matches(pair: ImagePair, num_matches=20, txt="img.jpg", save=False, matches=None):
        kp0, kp1 = pair.get_kps()
        matches = pair.get_matches() if matches is None else matches
        img_matches = cv2.drawMatches(pair.get_image(0), kp0, pair.get_image(1), kp1,
                                      np.random.choice(matches, size=num_matches, replace=False), None)
        img_matches = cv2.resize(img_matches, (1300, 360))

        Display.end(img_matches, "matches_" + txt, save)

    @staticmethod
    def hist(arr, txt="hist.jpg", save=False):
        h, _, __ = plt.hist(arr, label=txt)
        plt.xlabel('Y Distance in Pixels')
        plt.ylabel('Matches Num')
        if save:
            plt.savefig(txt)
        plt.show()
        return h

    @staticmethod
    def end(dis, txt, save):
        if save:
            cv2.imwrite(txt, dis)
        cv2.imshow(txt, dis)
        cv2.waitKey()

    @staticmethod
    def plot_3d(points, points2=None):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        if points2 is not None:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='coral')
            ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='lightblue')
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=-points[:, 2], cmap='viridis')
        # ax.set_ylim(-20,5)
        # ax.set_xlim(-20, 15)
        # ax.set_zlim(0,100)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    @staticmethod
    def plot_2d(points1, points2=None):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.scatter(points1[:, 0], points1[:, 1], c='coral')
        None if points2 is None else ax.scatter(points2[:, 0], points2[:, 1], c='lightblue')
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plt.show()

    @staticmethod
    def show_track(pose, num):

        return np.array([t * [1, 1, -1] for r, t in pose[:num]])
