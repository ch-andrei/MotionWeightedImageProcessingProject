from skimage.measure import block_reduce

import scipy.ndimage
from meshDisplace import *
from tools import *
from laplacianBlend import diffWeightedLaplacianBlend

########################################################################################################################

class BackgroundExtractor:
    def __init__(self, frame):
        self.count = 1
        self.image = frame.astype(np.float32)

    def processFrame(self, frame):
        image = (self.image * self.count + frame) / (self.count + 1)
        self.image = np.clip(image, 0, 255)
        self.count += 1
        return [], [self.image.astype(np.uint8)]

class MotionExtractor:
    def __init__(self, frame, windowSize, strength, threshold, flowWinSize, momentum=0.5):
        self.bufferSize = windowSize
        self.frameBuffer = []
        self.flowBuffer = []
        self.diffBuffer = []

        self.strength = strength
        self.threshold = threshold
        self.flowWinSize = flowWinSize
        self.momentum = momentum

        self.processedCount = 0
        self.addFrame(frame)

        self.bg = frame.copy()
        self.count = 1
        self.diff = np.zeros(frame.shape[:2], np.float32)

    def processFrame(self, frame):
        self.addFrame(frame)

        bg = self.bg

        # analyze background
        if len(self.flowBuffer) > 0:
            _frame = self.frameBuffer[-1]
            _flow = self.flowBuffer[-1]

            crtFlow = np.linalg.norm(np.abs(_flow), axis=2) # magnitude of flow
            crtFlow /= np.sqrt(2 * self.flowWinSize**2)
            crtFlow = crtFlow ** 0.1
            crtFlow = scipy.ndimage.gaussian_filter(crtFlow, 1, mode="constant", cval=0)
            crtFlow = remapRange(crtFlow, crtFlow.min(), crtFlow.max()) # stretch to fill [0-1] fully
            crtFlow = remapRange(crtFlow, self.threshold, 1. - self.threshold) # condense within smaller range

            for i in range(3):
                nonMovingPart = _frame[..., i] * (1. - self.strength * crtFlow)
                bgPart = bg[..., i] * (self.count + self.strength * crtFlow)

                bg[..., i] = (bgPart + nonMovingPart) / (self.count + 1)

            self.count += 1

        bg = np.clip(bg, 0, 255)
        self.bg = bg

        return [self.flowBuffer[len(self.flowBuffer) - 1]], [bg.astype(np.uint8)]

    def getCurrent(self):
        return self.frameBuffer[-1]

    def getPrevious(self):
        return self.frameBuffer[-2]

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
        # helps with motion tracking
        def discretize(img, bandsPerChannel=10):  # remap from full color range to a total of bands**3 possible colors
            return np.round(np.round(img / bandsPerChannel) * bandsPerChannel).astype(np.uint8)

        def enhanceImgForMotionEstimation(img, offset=50):
            return discretize(img)

        frame1 = enhanceImgForMotionEstimation(cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY))
        frame2 = enhanceImgForMotionEstimation(cv2.cvtColor(next, cv2.COLOR_BGR2GRAY))
        flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 7, self.flowWinSize, 6, 5, 1.2, 0)

        return flow

########################################################################################################################

class MotionSlower:
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

        # helps with motiin tracking
        def discretize(img, bands=20):
            # remap from full color range to a total of bands**3 possible colors, or bands for grayscale input
            scaler = bands / 255
            return np.round(np.round(img * scaler) / scaler).astype(np.uint8)

        def enhanceImgForMotionEstimation(img):
            img = remapRange(img, 35, 245, 255)
            return discretize(img)

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
        flowForward =  (flowForwardRaw  * (1.0 - self.edgeFlowInfluence) + flowForwardEdges  * self.edgeFlowInfluence)
        flowBackward = (flowBackwardRaw * (1.0 - self.edgeFlowInfluence) + flowBackwardEdges * self.edgeFlowInfluence)

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

        return flow, frames

    # simple ratio linear blend
    def lerpFrame(self, frame1, frame2, ratio=0):
        ratio = max(0., min(1., ratio)) # clip to [0-1]
        return (1.0 - ratio) * frame1 + ratio * frame2

    def motionWeightedLerpFrame(self, next, gridPixelFlow1, gridPixelFlow2, ratio=0):
        ratio = max(0., min(1., ratio)) # clip to [0-1]

        frame1 = self.prvs.copy()
        frame2 = next.copy()

        frame1 = meshDisplace(frame1, gridPixelFlow1) / 255
        frame2 = meshDisplace(frame2, gridPixelFlow2) / 255
        frame = diffWeightedLaplacianBlend(frame1, frame2, ratio)

        return (frame * 255).astype(np.uint8)