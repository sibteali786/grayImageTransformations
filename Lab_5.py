import cv2 as cv
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
