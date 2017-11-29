import cv2
import numpy as np
import scipy.ndimage

def getScaledPyr(img, levels):
    # scaled pyramid
    G = img.copy()
    gp = [G]
    for i in range(levels):
        G = cv2.resize(G, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        G = np.clip(G, 0., 1.)
        gp.append(G)
    gp.reverse()
    return gp

def getGaussianPyr(img, levels):
    # gaussian pyramid
    G = img.copy()
    gp = [G]
    for i in range(levels):
        G = cv2.pyrDown(G)
        G = np.clip(G, 0., 1.)
        gp.append(G)
    return gp

def getLaplacianPyr(img, levels):
    gp = getGaussianPyr(img, levels)
    # generate Laplacian Pyramid for img
    lp = [gp[levels-1]]
    for i in np.arange(levels-1,0,-1):
        GE = cv2.pyrUp(gp[i])
        GE = np.clip(GE, 0., 1.)
        L = cv2.subtract(gp[i-1],GE)
        L = np.clip(L, 0., 1.)
        lp.append(L)
    return lp

def diffWeightedLaplacianBlend(a, b, ratio, pyrLevels=2):
    # generate a ratio value for every pixel based on the difference and input ratio
    # when ratio is 0, output all 0s
    # when ratio is 1, output all 1s
    # when ratio is [0-1], use a linearly scaled normalized difference array as ratio
    def diffWeightedRatio(a, b, ratio):
        diff = np.abs(b - a)
        diff = scipy.ndimage.gaussian_filter(diff, 3)
        if diff.max() > 1:
            diff /= diff.max()  # normalized

        r = ratio * 2.0 - 1.0
        if r >= 0:
            # closer to a
            return diff * (1 - r) + r
        else:
            # closer to b
            return diff * (1 + r)

    d = diffWeightedRatio(a, b, ratio)

    lpa = getLaplacianPyr(a, pyrLevels)
    lpb = getLaplacianPyr(b, pyrLevels)
    lpd = getScaledPyr(d, pyrLevels - 1)

    LS = []
    for ld,(la,lb) in zip(lpd, zip(lpa, lpb)):
        ls = la * (1.0 - ld) + lb * (ld)
        LS.append(ls)

    # reconstruct
    ls = LS[0]
    for i in np.arange(1, pyrLevels, 1):
        ls = cv2.pyrUp(ls)
        ls = cv2.add(ls, LS[i])

    return np.clip(ls, 0., 1.)
