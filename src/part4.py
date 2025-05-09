import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    w = 0
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]
        w += im1.shape[1]

        # TODO: 1.feature detection & matching
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        matches = bf.knnMatch(des1, des2, k=2)
        good_match_u = []
        good_match_v = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_match_u.append(kp1[m.queryIdx].pt)
                good_match_v.append(kp2[m.trainIdx].pt)
        good_match_u = np.array(good_match_u)
        good_match_v = np.array(good_match_v)
        # TODO: 2. apply RANSAC to choose best H
        times = 5000
        threshold = 4
        best_inlier = 0
        Hmax = np.eye(3)
        for i in range(times+1):
            random_u = np.zeros((4, 2))
            random_v = np.zeros((4, 2))
            for j in range(4):
                index = random.randint(0, len(good_match_u) - 1)
                random_u[j] = good_match_u[index]
                random_v[j] = good_match_v[index]
            H = solve_homography(random_v, random_u)
            one = np.ones((1, len(good_match_u)))
            M = np.concatenate((np.transpose(good_match_v), one), axis=0)
            W = np.concatenate((np.transpose(good_match_u), one), axis=0)
            Mbar = np.dot(H, M)
            Mbar = np.divide(Mbar, Mbar[-1,:])
            err = np.linalg.norm((Mbar - W)[:-1,:], ord=1, axis=0)
            inliner = np.sum(err < threshold)
            if inliner > best_inlier:
                best_inlier = inliner
                Hmax = H
        # TODO: 3. chain the homographies

        # TODO: 4. apply warping
        last_best_H = np.dot(last_best_H, Hmax)
        out = warping(im2, dst, last_best_H, 0, im2.shape[0], w, w+im2.shape[1], direction='b')

    return out 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)