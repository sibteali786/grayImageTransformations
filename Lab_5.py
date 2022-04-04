import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import Transformations

path = r'C:\Users\lenovo\PycharmProjects\imageTransformations\einstein.tif'
myImage = Transformations.grayImgTransformations("lab_5", path)

image = cv.imread(path, 0)
fifth_perc = np.percentile(image, 5)
ninetyFifth_perc = np.percentile(image, 95)

newImage = myImage.replaceLowHighValue(fifth_perc, ninetyFifth_perc, 0, 255)
print(newImage)
cv.imshow("Conrast Stretching", newImage)
cv.waitKey()

histogram = np.zeros(256, dtype=int)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        histogram[image[i][j]] += 1

plt.plot(histogram)
plt.show()
# pdf
pdf = histogram / (image.shape[0] * image.shape[1])
plt.plot(pdf)
plt.show()


#
def cummDensityFunction(pdf):
    sum = float(pdf[0])
    cdf = np.zeros(len(pdf))
    cdf[0] = sum
    for i in range(256):
        cdf[i] = pdf[i] + cdf[i - 1]
    return cdf


cdf = cummDensityFunction(pdf)
plt.plot(cdf)
plt.show()
# transformation Function
transformFunc = cdf * 255
plt.plot(transformFunc)
plt.show()

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        image[i][j] = transformFunc[image[i][j]]

image = image.astype(np.uint8)
plt.plot(image)
plt.show()

cv.imshow("histogram Equalization", image)
cv.waitKey()
