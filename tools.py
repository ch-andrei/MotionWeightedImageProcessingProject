import cv2, os, shutil
import numpy as np

########################################################################################################################
# color correction

def remapRange(x, min=0.5, max=1.0, colorRange=1.0):
    return (colorRange / (max - min) * (-min + x)).clip(0, colorRange)

# helps with motion tracking
def discretize(img, bands=40):
    # remap from full color range to a total of <bands**3> possible colors for RGB, or <bands> for grayscale input
    scaler = bands / 255
    return np.round(np.round(img * scaler) / scaler)

def enhanceImgForMotionEstimation(img):
    img = remapRange(img, img.min(), img.max(), 255)
    img = discretize(img)
    return img.astype(np.uint8)

########################################################################################################################

def resize(img, resizeRatio=1):
    return cv2.resize(img, (0, 0), fx=resizeRatio, fy=resizeRatio, interpolation=cv2.INTER_CUBIC)

def makeFolders(path, folderNames, verbose=False):
    for foldername in folderNames:
        try:
            os.makedirs("{}/{}".format(path, foldername))
        except FileExistsError:
            if verbose:
                print("Folder <{}/{}> already exists: delete contents.".format(path, foldername))
            p = "{}/{}".format(path, foldername)
            shutil.rmtree(p)
            os.makedirs("{}/{}".format(path, foldername))

def writeFlowMap(folderFramesOut, flow, count, suffix=""):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((flow.shape[:2] + (3,)), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255 # saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('{}/_flow_frame{}-{}.jpg'.format(folderFramesOut, suffix, ("{}".format(count)).zfill(4)), bgr)

