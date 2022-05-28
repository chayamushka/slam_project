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
    def track(movie, track_id, save=False):
        track = movie.tracks[track_id]
        frames = movie.get_frames(track.get_frame_ids())
        for i, frame in enumerate(frames):
            match_id = track.get_match(frame.frame_id)
            image0, image1 = frame.img0, frame.img1
            kp0, kp1 = image0.get_kp(match_id), image1.get_kp(match_id)

            img_0 = cv2.drawKeypoints(image0.get_image(), [kp0], outImage=np.array([]), color=(0, 0, 255),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            img_1 = cv2.drawKeypoints(image1.get_image(), [kp1], outImage=np.array([]), color=(0, 0, 255),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            dis = np.hstack((Display.crop_around(img_0, kp0.pt), Display.crop_around(img_1, kp1.pt)))

            Display.end(dis, f"track_{track_id}_frame_{i}.jpg", save)

    @staticmethod
    def hist(arr, x_label='x', y_label='y', title='histo', save=False):
        h, _ = np.histogram(arr, bins=np.arange(max(arr) + 2))
        plt.plot(h)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if save:
            plt.savefig(title + '.jpg')
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
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    @staticmethod
    def scatter_2d(points1, points2=None, points3=None, label=None):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.scatter(points1[:, 0], points1[:, 1], c='coral',linewidths=0.1)
        None if points2 is None else ax.scatter(points2[:, 0], points2[:, 1], c='lightblue',linewidths=0.1)
        None if points2 is None else ax.scatter(points3[:, 0], points3[:, 1], c='green',linewidths=0.1)
        None if label is None else ax.legend(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plt.show()

    @staticmethod
    def crop_around(image, pt, r=100):
        r = int(r / 2)
        px, py = int(pt[0]), int(pt[1])
        return image[max(py - r,0):py + r, max(px - r,0):px + r]

    @staticmethod
    def simple_plot(y, x_name='x', y_name='y', plot_name='plot', save=False, max_y=None):
        mean = np.mean(y)
        fig, ax = plt.subplots()

        ax.plot(y, label=y_name)
        ax.plot([mean] * len(y), label="Mean", linestyle='--')
        plt.ylim([0, 2 * mean if max_y == None else max_y])
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.legend()
        plt.title(plot_name)
        if save:
            plt.savefig(plot_name)
        plt.show()
