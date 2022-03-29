import cv2 as cv
import numpy as np

path = r'C:\Users\lenovo\PycharmProjects\Lab_4'


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


myImage = grayImgTransformations("ClassTaskImage", path + "\Fig0241(a)(einstein low contrast).tif")
# negTrans = myImage.negTrans()
# varyingRange = myImage.replPixActoMean(0,255)
varyingRangeBound = myImage.replPixActoMeanWithBounds(20, 0, 255)
cv.imshow("Negative Transformed", varyingRangeBound)
cv.waitKey()

# logTransformed = myImage.logTrans()
# powerlawtrans = myImage.powerLawTrans(2.2)
# grayLevelScaling = myImage.grayLevelSlicing(150,240,200)
# histogram = myImage.histogramForImage()
# print(histogram)
