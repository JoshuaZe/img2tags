import cv2

if __name__ == '__main__':
    # http://bcmi.sjtu.edu.cn/~zhangliqing/Papers/2007CVPR_Houxiaodi_04270292.pdf
    # https://www.cnblogs.com/ccbb/archive/2011/05/19/2051442.html
    # https://github.com/uoip/SpectralResidualSaliency/blob/master/src/saliency.py
    # https://mathpretty.com/10683.html
    # https://en.wikipedia.org/wiki/Saliency_map
    # https://zhuanlan.zhihu.com/p/115002897
    # load the input image
    image = cv2.imread(
        '/Users/zezzhang/Workspace/img2tags_serving/image_prediction/data/A/train/image/id_00000001_856.jpg'
    )
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # initialize OpenCV's static saliency spectral residual detector and
    # compute the saliency map
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    # if we would like a *binary* map that we could process for contours,
    # compute convex hull's, extract bounding boxes, etc., we can
    # additionally threshold the saliency map
    _, threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # show the images
    print(success)
    cv2.imshow("Image", image)
    cv2.imshow("Output", saliencyMap)
    cv2.imshow("Thresh", threshMap)
    cv2.waitKey(0)

    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    # if we would like a *binary* map that we could process for contours,
    # compute convex hull's, extract bounding boxes, etc., we can
    # additionally threshold the saliency map
    _, threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # show the images
    print(success)
    cv2.imshow("Image", image)
    cv2.imshow("Output", saliencyMap)
    cv2.imshow("Thresh", threshMap)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
