import cv2
from tools import *
from enum import Enum

from operators import *

########################################################################################################################
# input configuration

folder = "data"

# file must be in .mp4 format!
# name = "hoonigan_short"
name = "805391579"

########################################################################################################################
# output configuration

fps = 30
"""@fps Frames per second in the output video.
        Note: if this doesnt match the original, timing will appear different.
        ex: slow down by 4x, but original is 30fps, and output is 60fps, will result in an overall 2x slow down 
        (not 4x as queried)."""

downsampleFactor = 2 # ex: 2x downsample for 1920x1080 input -> 960x540 output
maxFramesToWrite = 500 # how many input frames to process from the input video
skipFrames = 0 # how many frames to skip in the beginning

########################################################################################################################
# configuration: which effect to apply

class Configuration(Enum):
    SLOW_MOTION = "slowMotion"
    MOTION_BACKGROUND_EXTRACT = "motionBackground"
    BACKGROUND_EXTRACT = "backgroundExtract"

# controls which effect will be applied
configuration = Configuration.MOTION_BACKGROUND_EXTRACT

########################################################################################################################

# optical flow window size (in pixels? not certain, see opencv implementation)
flowWindowSize = 30

########################################################################################################################
# SLOW_MOTION configurable parameters

# how much the time is slowed down
ms_slowDownFactor = 10

# controls how much the overall flow is influenced by the movement of edges detected in the two adjacent frames.
# when 0, only normal optical flow is used, when set to 1, only edge optical flow is used
# using edge flow regularizes the problem, effectively decreasing the blending artifacts near edges of objects.
# helps with cases where the background is not moving, but an object in the foreground is. normally, this causes
# the background to be (severely) distorted/transformed, as the algorithm compensates for motion.
# one disadvantage of relying on edge motion too much is sparsity of flow in inner regions of motions, ore regions
# that are not considered as edges, hence this constant should be set according to the input image (trial and error)
ms_edgeFlowInfluence = 0.05

########################################################################################################################
# MOTION_DUPLICATION configurable parameters

# how many frames frames to consider for history of motion duplication
md_bufferSize = 2 # TODO: remove this param

# how much to decay intensity of the duplication over time, 0 for full intensity; [0-1], 1 turns the effect off
md_strength = 1

md_threshold = 0.1

########################################################################################################################

folderOut = "{}-{}".format(name, str(configuration))
pathFramesOut = "{}/{}".format(folder, folderOut)

videoFile_name_in = "{}/{}.mp4".format(folder, name)
videoFile_name_out = "{}/__{}_out.mp4".format(pathFramesOut, name)

makeFolders(folder, [folderOut], True)

print("Processing [{}.mp4], writing to folder [{}]".format(name, pathFramesOut))

# input frames
cap = cv2.VideoCapture(videoFile_name_in)

ret, frame1 = cap.read()
frame1 = resize(frame1, 1.0 / downsampleFactor)

h, w = frame1.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = cv2.VideoWriter(videoFile_name_out, fourcc, fps, (w, h), True)

prvs = frame1

operator = None # operator must be initialized later to support skipping frames

count = 0
while(cap.isOpened()):

    ret, frame2 = cap.read()

    if ret == True:
        # preprocessing
        frame2 = resize(frame2, 1.0 / downsampleFactor)

        next = frame2

        if count < skipFrames:
            # skip current frame
            pass
        else:
            # operator must be initialized here to support skipping frames
            if operator == None:
                if configuration == Configuration.SLOW_MOTION:
                    operator = MCFI(prvs, ms_slowDownFactor, ms_edgeFlowInfluence, flowWindowSize)
                elif configuration == Configuration.MOTION_BACKGROUND_EXTRACT:
                    operator = MCBE(prvs, md_bufferSize, md_strength, md_threshold, flowWindowSize)
                elif configuration == Configuration.BACKGROUND_EXTRACT:
                    operator = BE(prvs)
                else:
                    print("Unsupported configuration.")
                    exit(1)

            # process frame
            extra, flow, frames = operator.processFrame(next)

            # write resulting frames
            for index, frame in enumerate(frames):
                cv2.imwrite("{}/frame-{}-{}.jpg".format(pathFramesOut, count, index), frame) # image
                writer.write(frame) # video

            # write flow as images
            for index, _flow in enumerate(flow):
                writeFlowMap(pathFramesOut, _flow, count, str(index))

            # write flow as images
            for index, motionMap in enumerate(extra):
                cv2.imwrite("{}/motionMap-{}-{}.jpg".format(pathFramesOut, count, index), motionMap)  # image

        prvs = next

        count += 1
        print("Processed frame {}: {}/{}".format(count, count - skipFrames, maxFramesToWrite))
    else:
        print("No more frames in the input video.")
        break

    # stop condition
    if count - skipFrames >= maxFramesToWrite:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
