from skimage.measure import block_reduce

import scipy.ndimage
from meshDisplace import *
from tools import *
from laplacianBlend import diffWeightedLaplacianBlend

########################################################################################################################

class BE:
    def __init__(self, frame):
        self.count = 1
        self.image = frame.astype(np.float32)

    def processFrame(self, frame):
        image = (self.image * self.count + frame) / (self.count + 1)
        self.image = np.clip(image, 0, 255)
        self.count += 1
        return [], [], [self.image.astype(np.uint8)]

# motion compensated background extractor
class MCBE:
    def __init__(self, frame, windowSize, strength, threshold, flowWinSize,
                 flowMotionTreshold=0.05,
                 flowPow=0.5,
                 countPow=0.9
                 ):
        self.bufferSize = windowSize
        self.frameBuffer = []
        self.flowBuffer = []
        self.diffBuffer = []
        # initially this was intended to have a fixed window size, but then i just implemented a global average
        # that uses motion weighted averaging

        self.strength = strength
        self.threshold = threshold
        self.flowWinSize = flowWinSize
        self.flowWinSizeDiag = np.sqrt(2 * self.flowWinSize**2)
        self.flowMotionTreshold = flowMotionTreshold
        self.flowPow = flowPow
        self.countPow = countPow

        self.processedCount = 0
        self.addFrame(frame)

        self.motionMap = np.zeros(frame.shape[:2], np.float32)
        self.bg = frame.copy()
        self.count = 1
        self.diff = np.zeros(frame.shape[:2], np.float32)

    def processFrame(self, frame):
        self.addFrame(frame)

        bg = self.bg

        motionMap = None

        # analyze background
        if len(self.flowBuffer) > 0:
            _frame = self.frameBuffer[-1]
            _flow = self.flowBuffer[-1]

            # returns binary decision: is pixel moving or not
            def getMotionMask(_flow):
                flow = np.linalg.norm(np.abs(_flow), axis=2) # magnitude of flow
                flow /= self.flowWinSizeDiag

                # flow = flow**self.flowPow

                flow = scipy.ndimage.gaussian_filter(flow, 3, mode="constant", cval=0) # smoothen

                flow = remapRange(flow, 0, max(0.5, flow.max()))
                flow = remapRange(flow, self.flowMotionTreshold, 1. - self.flowMotionTreshold)

                # binarize decision
                flow[flow >= self.flowMotionTreshold] = 1
                flow[flow < self.flowMotionTreshold] = 0

                flow = scipy.ndimage.gaussian_filter(flow, 3, mode="constant", cval=0) # smoothen the boundary

                return flow

            motionMap = getMotionMask(_flow)

            for i in range(3):
                countFactor = self.count**self.countPow
                nonMovingPart = _frame[..., i] * (self.strength * (1. - motionMap))
                bgPart = bg[..., i] * (countFactor + self.strength * motionMap)
                bg[..., i] = (bgPart + nonMovingPart) / (countFactor + self.strength)

            self.count += 1

        bg = np.clip(bg, 0, 255)
        self.bg = bg

        return [(motionMap * 255).astype(np.uint8)], [self.flowBuffer[len(self.flowBuffer) - 1]], [bg.astype(np.uint8)]

    def getCurrent(self):
        return self.frameBuffer[-1]

    def getPrevious(self):
        return self.frameBuffer[-2]

    # window of size N frame buffer
    def addFrame(self, frame):
        print(len(self.frameBuffer))
        self.processedCount += 1
        self.frameBuffer.append(frame)

        if len(self.frameBuffer) > self.bufferSize:
            self.frameBuffer.pop(0)

        try:
            prvs = self.getPrevious()

            flow = self.opticalFlow(self.getCurrent(), prvs) # backwards flow
            diff = np.mean(np.abs(frame - prvs), axis=2)

            self.flowBuffer.append(flow)
            self.diffBuffer.append(diff)

            if len(self.flowBuffer) > self.bufferSize - 1:
                self.flowBuffer.pop(0)

            if len(self.diffBuffer) > self.bufferSize - 1:
                self.diffBuffer.pop(0)

        except IndexError:
            print("[Warning] MotionDuplicator.getPrevious(): Not enough frames in buffer.")

    def opticalFlow(self, prvs, next):
        frame1 = enhanceImgForMotionEstimation(cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY))
        frame2 = enhanceImgForMotionEstimation(cv2.cvtColor(next, cv2.COLOR_BGR2GRAY))
        flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 7, self.flowWinSize, 6, 5, 1.2, 0)

        return flow

########################################################################################################################

# nearest frame interpolation
class NFI:
    def __init__(self, frame,
                 slowDownFactor,
                 edgeFlowInfluence,
                 flowWinSize):
        self.prvs = frame
        self.slowDownFactor = slowDownFactor
        self.count = 0

    def processFrame(self, next):
        self.count += 1

        frames = []
        for i in range(self.slowDownFactor):
            frames.append(self.prvs)

        self.prvs = next

        return [], [], frames

# linear blend frame interpolation
class LBFI:
    def __init__(self, frame,
                 slowDownFactor,
                 edgeFlowInfluence,
                 flowWinSize):
        self.prvs = frame
        self.slowDownFactor = slowDownFactor
        self.count = 0

    def processFrame(self, next):
        self.count += 1

        frames = []
        lerpIncr = 1.0 / (self.slowDownFactor)
        for i in range(self.slowDownFactor):
            ratio = (i + 1) * lerpIncr

            frame = self.lerpFrame(self.prvs, next, ratio).astype(np.uint8)

            frames.append(frame)

        self.prvs = next

        return [], [], frames

    # simple ratio linear blend
    def lerpFrame(self, frame1, frame2, ratio=0):
        ratio = max(0., min(1., ratio)) # clip to [0-1]
        return (1.0 - ratio) * frame1 + ratio * frame2

# motion compensated frame interpolation
class MCFI:
    def __init__(self, frame,
                 slowDownFactor,
                 edgeFlowInfluence,
                 flowWinSize):
        self.prvs = frame
        self.slowDownFactor = slowDownFactor
        self.flowWinSize = flowWinSize
        self.edgeFlowInfluence = edgeFlowInfluence
        self.count = 0

    def processFrame(self, next):
        self.count += 1
        frame1 = cv2.cvtColor(self.prvs, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

        frameEnh1 = enhanceImgForMotionEstimation(frame1)
        frameEnh2 = enhanceImgForMotionEstimation(frame2)

        edges1 = cv2.Canny(frameEnh1, 100, 200)
        edges2 = cv2.Canny(frameEnh2, 100, 200)

        # cv2.calcOpticalFlowFarneback: prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags

        # scale 1
        flowForwardRaw  = cv2.calcOpticalFlowFarneback(frameEnh1, frameEnh2, None, 0.5, 7, self.flowWinSize, 6, 5, 1.2, 0)
        flowBackwardRaw = cv2.calcOpticalFlowFarneback(frameEnh2, frameEnh1, None, 0.5, 7, self.flowWinSize, 6, 5, 1.2, 0)

        # edges scale 1
        flowForwardEdges  = cv2.calcOpticalFlowFarneback(edges1, edges2, None, 0.5, 7, self.flowWinSize, 6, 5, 1.2, 0)
        flowBackwardEdges = cv2.calcOpticalFlowFarneback(edges2, edges1, None, 0.5, 7, self.flowWinSize, 6, 5, 1.2, 0)

        # include edge flow influence
        flowForward =  (flowForwardRaw  + flowForwardEdges  * self.edgeFlowInfluence) / (1.0 + self.edgeFlowInfluence)
        flowBackward = (flowBackwardRaw + flowBackwardEdges * self.edgeFlowInfluence) / (1.0 + self.edgeFlowInfluence)

        flow = [flowForward, flowBackward, flowForwardRaw, flowBackwardRaw, flowForwardEdges, flowBackwardEdges]

        def motionFlow2MeshDisplaceMap(flow):
            gridPixelFlowX = (block_reduce(flow[..., 0], (self.flowWinSize, self.flowWinSize),
                                           func=np.mean)).astype(np.int16)
            gridPixelFlowY = (block_reduce(flow[..., 1], (self.flowWinSize, self.flowWinSize),
                                           func=np.mean)).astype(np.int16)

            gridH, gridW = gridPixelFlowX.shape[0], gridPixelFlowX.shape[1]

            gridPixelFlow = np.zeros((gridH, gridW, 2), np.float32)
            gridPixelFlow[..., 0] = -gridPixelFlowX
            gridPixelFlow[..., 1] = -gridPixelFlowY

            return gridPixelFlow

        pixelFlowForward = motionFlow2MeshDisplaceMap(flowForward)
        pixelFlowBackward = motionFlow2MeshDisplaceMap(flowBackward)

        frames = []
        lerpIncr = 1.0 / (self.slowDownFactor)
        for i in range(self.slowDownFactor):
            ratio = (i + 1) * lerpIncr
            print("Processing slowmo between frames {}-{}-{}".format(self.count, self.count+1, ratio))

            pff = (pixelFlowForward * (ratio)).astype(np.int16)
            pfb = (pixelFlowBackward * (1. - ratio)).astype(np.int16)
            frame = self.motionWeightedLerpFrame(next, pff, pfb, ratio)

            frames.append(frame)

        self.prvs = next

        return [], flow, frames

    def motionWeightedLerpFrame(self, next, gridPixelFlow1, gridPixelFlow2, ratio=0):
        ratio = max(0., min(1., ratio)) # clip to [0-1]

        frame1 = self.prvs.copy()
        frame2 = next.copy()

        frame1 = meshDisplace(frame1, gridPixelFlow1) / 255
        frame2 = meshDisplace(frame2, gridPixelFlow2) / 255
        frame = diffWeightedLaplacianBlend(frame1, frame2, ratio)

        return (frame * 255).astype(np.uint8)