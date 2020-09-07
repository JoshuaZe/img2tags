import cv2
import numpy as np
import math
from PIL import Image
from colorthief import MMCQ
from nearest_color import find_nearest_color, rgb_to_hex


def saliency_map_by_backprojection(image, levels=3, scale=1):
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


def generate_saliency_map(image):
    saliency_map = saliency_map_by_backprojection(image)

    # threshold over a saliency map
    threshold, binary_map = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # noise removal
    # https://zhuanlan.zhihu.com/p/46306138
    kernel = np.ones((3, 3), np.uint8)
    binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel, iterations=2)  # 形态开运算
    binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel, iterations=5)  # 形态闭运算

    return saliency_map, binary_map


def find_largest_contour(binary_map):
    max_contour = None
    # find Max Countour
    try:
        contours, hierarchy = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea)
        max_contour = contours[-1]
    except Exception as error_msg:
        print(error_msg)
        print("Failed to find Max Contour")
    # extract the convex Hull or approx PloyDP
    try:
        # epsilon = 0.01 * cv2.arcLength(max_contour, True)
        # max_contour = cv2.approxPolyDP(max_contour, epsilon, True)
        max_contour = cv2.convexHull(max_contour)
    except Exception as error_msg:
        print(error_msg)
        print("Failed to Build Approx PolyDP or Convex Hull for Max Contour")
    return max_contour


def generate_foreground_mask(image, binary_map, max_contour):
    try:
        # Initial Mask
        mask = binary_map.copy()
        mask[np.where(mask > 0)] = cv2.GC_FGD
        bbox = cv2.boundingRect(max_contour)
        # GraphCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask, bbox, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == 2) | (mask == 0), 0, 1)

        # # watershed
        # max_contour_map = binary_map.copy()
        # cv2.drawContours(max_contour_map, [max_contour], 0, (0, 255, 0), 3)
        # kernel = np.ones((3, 3), np.uint8)
        # sure_bgd = cv2.dilate(max_contour_map, kernel, iterations=5)
        # dist_transform = cv2.distanceTransform(max_contour_map, cv2.DIST_L2, 5)
        # _, sure_fgd = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, cv2.THRESH_BINARY)
        # sure_fgd = sure_fgd.astype('uint8')
        # unknown = cv2.subtract(sure_bgd, sure_fgd)
        # _, markers = cv2.connectedComponents(sure_fgd)
        # markers = markers + 1
        # markers[unknown == 255] = 0
        # markers = cv2.watershed(img, markers)
        # mask = np.where(markers <= 1, 0, 1).astype('uint8')
    except Exception as error_msg:
        print(error_msg)
        print("Failed to Build Grab Cut Segmentation")
        # contour
        if max_contour is None:
            mask = binary_map.copy()
            mask[np.where(mask > 0)] = cv2.GC_FGD
        else:
            mask = np.zeros(binary_map.shape)
            cv2.fillConvexPoly(mask, max_contour, cv2.GC_FGD)

    foreground_mask = mask.astype('uint8')
    return foreground_mask


def generate_attention_mask(image):
    # generate saliency map and threshold-based map with center prior
    saliency_map, binary_map = generate_saliency_map(image)
    # find largest contour on threshold-based map
    max_contour = find_largest_contour(binary_map)
    # generate foreground mask
    mask = generate_foreground_mask(image, binary_map, max_contour)
    # # debugging
    # cv2.imshow("Original", image)
    # cv2.imshow("SaliencyMap", saliency_map)
    # cv2.imshow("BinaryMap", binary_map)
    # cv2.imshow("Segmentation", image * mask[:, :, np.newaxis])
    # cv2.imshow("MaxContours", cv2.drawContours(image, [max_contour], 0, (0, 255, 0), 3))
    # cv2.waitKey(-1)
    return mask


if __name__ == '__main__':

    image = Image.open('/Users/zezzhang/Workspace/img2tags_serving/image_prediction/data/A/train/image/id_00000001_856.jpg')
    image = image.convert('RGB')

    img = np.array(image)
    # Convert RGB to BGR
    img = img[:, :, ::-1].copy()
    mask = generate_attention_mask(img)

    image = Image.fromarray(np.dstack((np.array(image) * mask[:, :, np.newaxis], mask * 255)).astype('uint8'), 'RGBA')
    # https://github.com/fengsp/color-thief-py
    # https://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image
    # https://www.hisour.com/zh/color-appearance-model-24824/

    color_count = 8
    imageWithColorPalette = image.quantize(colors=color_count, method=2)
    palette_colors = sorted(imageWithColorPalette.getcolors(), key=lambda x: x[0])
    palette_rgb = imageWithColorPalette.getpalette()
    palette = [tuple(palette_rgb[4 * idx:(4 * idx + 3)]) for _, idx in palette_colors]

    for color in palette:
        print(rgb_to_hex(color))
        result = find_nearest_color(color, dist_type='CIE76')
        print(result)

    image.show()
    imageWithColorPalette.show()
