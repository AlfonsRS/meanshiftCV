import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth

# input image
img = cv2.imread("pills.jpg")
cv2.imshow('Original', img)

# filter to reduce noise
# img = cv2.GaussianBlur(img, [5,5],cv2.BORDER_DEFAULT)
img = cv2.medianBlur(img, 7)
# cv2.imshow('blurred', img)

# flatten the image
flat_image = np.reshape(img, (-1,3))
flat_image = np.float32(flat_image)
# print(flat_image)
# print(len(flat_image))
# cv2.imshow('flat', flat_image)

# meanshift
bandwidth = estimate_bandwidth(flat_image, quantile=.1, n_samples=300)
print(bandwidth)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
#ms = MeanShift(bin_seeding=True)
ms.fit(flat_image)
labeled=ms.labels_
# print(labeled)

# get number of segments
segments = np.unique(labeled)
print('Number of clusters: ', segments.shape[0])


# get the average color of each segment
total = np.zeros((segments.shape[0], 3), dtype=float)
count = np.zeros((segments.shape[0], 3), dtype=float)
for i, label in enumerate(labeled):
    # if i < 300:
    #     print("i = " + str(i))
    #     print("label = " + str(label))
    total[label] = total[label] + flat_image[i]
    count[label] += 1
# print(count)
avg = total/count
avg = np.uint8(avg)

# cast the labeled image into the corresponding average color
res = avg[labeled]
result = res.reshape((img.shape))
# show the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

