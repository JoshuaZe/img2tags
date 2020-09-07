import numpy as np
import cv2

if __name__ == '__main__':
    # http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html

    img = cv2.imread(
        '/Users/zezzhang/Workspace/img2tags_serving/image_prediction/data/A/train/image/id_00000001_856.jpg'
    )
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(gray)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    cv2.imshow('saliencyMap', saliencyMap)

    blur = cv2.GaussianBlur(saliencyMap, (5, 5), 0)
    cv2.imshow('blur', blur)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel, iterations=2)  # 形态开运算
    cv2.imshow('opening', opening)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    cv2.imshow('sure_bg', sure_bg)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, cv2.THRESH_BINARY)

    cv2.imshow('sure_fg', sure_fg)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers <= 1] = [255, 255, 255]

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
