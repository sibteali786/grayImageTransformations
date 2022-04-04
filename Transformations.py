import cv2 as cv
import numpy as np


class grayImgTransformations:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.image = cv.imread(path, 0)
        self.meanPixel = np.mean(self.image)

    def negTrans(self):
        imgRes = 255 - self.image
        return imgRes

    def logTrans(self):
        c = 255 / (np.log(1 + np.amax(self.image)))
        imgS = c * np.log(self.image + 1)
        imgS = imgS.astype(np.uint8)
        return imgS

    def powerLawTrans(self, gamma):
        imgResult = 255 * ((self.image / 255) ** float(gamma))
        imgResult = imgResult.astype(np.uint8)
        return imgResult

    def replaceLowHighValue(self, lowerLimit, higherLimit, replaceLower, replacehigher):
        height = self.image.shape[0]
        width = self.image.shape[1]
        for i in range(height):
            for j in range(width):
                if self.image[i][j] < lowerLimit:
                    self.image[i][j] = replaceLower
                elif self.image[i][j] > higherLimit:
                    self.image[i][j] = replacehigher
                else:
                    self.image[i][j] = ((self.image[i][j] - lowerLimit) / (higherLimit - lowerLimit)) * 255
        return self.image

    def grayLevelSlicing(self, lowerLimit, upperLimit, replaceValue):
        height = self.image.shape[0]
        width = self.image.shape[1]
        imgResult = np.zeros([height, width])
        for i in range(height):
            for j in range(width):
                if lowerLimit <= self.image[i][j] <= upperLimit:
                    imgResult[i][j] = replaceValue
        return imgResult

    def histogramForImage(self):
        histVec = np.zeros(255, dtype=np.uint8)
        height = self.image.shape[0]
        width = self.image.shape[1]
        for i in range(height):
            for j in range(width):
                histVec[self.image[i][j]] += 1

        return histVec

    def replPixActoMean(self, lowValue, highValue):
        height = self.image.shape[0]
        width = self.image.shape[1]
        imgRes = np.zeros([height, width])
        for i in range(height):
            for j in range(width):
                if self.image[i][j] <= self.meanPixel:
                    imgRes[i][j] = highValue
                if self.image[i][j] >= self.meanPixel:
                    imgRes[i][j] = lowValue
        return imgRes

    def replPixActoMeanWithBounds(self, bound, lowValue, highValue):
        height = self.image.shape[0]
        width = self.image.shape[1]
        imgRes = np.zeros([height, width])
        for i in range(height):
            for j in range(width):
                if self.meanPixel + bound >= self.image[i][j] >= self.meanPixel - bound:
                    imgRes[i][j] = lowValue
                else:
                    imgRes[i][j] = highValue

        return imgRes
