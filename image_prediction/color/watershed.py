import numpy as np
import cv2
from matplotlib import pyplot as plt


def saliency_map_by_backprojection(image, levels=2, scale=1):
    # OpenCV is BGR, Pillow is RGB
    height, width, _ = image.shape
    prior_mask = np.zeros((height, width), np.uint8)
    prior_mask[height // 4: 3 * height // 4, width // 4: 3 * width // 4] = 1
    source_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # calculating object histogram and normalize histogram
    source_hist = cv2.calcHist([source_hsv], [0, 1], prior_mask, [levels, levels], [0, 180, 0, 256])
    cv2.normalize(source_hist, source_hist, 0, 255, cv2.NORM_MINMAX)

    # apply back-projection
    target_dst = cv2.calcBackProject([target_hsv], [0, 1], source_hist, [0, 180, 0, 256], scale)
    cv2.normalize(target_dst, target_dst, 0, 255, cv2.NORM_MINMAX)
    cv2.equalizeHist(target_dst, target_dst)

    return target_dst


if __name__ == '__main__':
    # http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html

    img = cv2.imread(
        '/Users/zezzhang/Workspace/img2tags_serving/image_prediction/data/A/train/image/id_00000001_856.jpg'
    )

    saliency_map = saliency_map_by_backprojection(img)
    threshold, binary_map = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('binary_map', binary_map)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel, iterations=2)  # 形态开运算
    cv2.imshow('opening', opening)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=5)  # 形态闭运算
    cv2.imshow('closing', closing)

    # sure background area
    sure_bg = cv2.dilate(closing, kernel, iterations=5)
    cv2.imshow('sure_bg', sure_bg)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    cv2.imshow('Distance Transform Image', dist_transform)

    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, cv2.THRESH_BINARY)
    cv2.imshow('sure_fg', sure_fg)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    num_markers, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1

    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    # getting mask with connectComponents
    for label in np.unique(markers):
        mask = np.array(markers, dtype=np.uint8)
        mask[markers == label] = 255
        cv2.imshow('component' + str(label), mask)

    img[markers <= 1] = [255, 255, 255]

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
