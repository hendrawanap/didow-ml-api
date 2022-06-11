# import the necessary packages
import numpy as np
import imutils
import cv2
from imutils.contours import sort_contours

def preprocess_image(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
    contours = sort_contours(contours, method="left-to-right")[0]
    img_height = image.shape[0]
    return contours, gray, image, img_height


# initialize the list of contour bounding boxes and associated characters that we'll be OCR'ing
def set_box(contours, gray, width, height, img_height):
    chars = []
    for c in contours:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # filter out bounding boxes, ensuring they are neither too small
        # nor too large
       	if (h >= int(img_height * 0.3)):
            # extract the character and threshold it to make the character
            # appear as *white* (foreground) on a *black* background, then
            # grab the width and height of the thresholded image
            roi = gray[y:y + h, x:x + w]
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.threshold(
                roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            (tH, tW) = thresh.shape
            # if the width is greater than the height, resize along the
            # width dimension
            if tW > tH:
                thresh = imutils.resize(thresh, width)
            # otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height)
            # re-grab the image dimensions (now that its been resized)
            # and then determine how much we need to pad the width and
            # height such that our image will be 32x32
            (tH, tW) = thresh.shape
            dX = int(max(0, width - tW) / 2.0)
            dY = int(max(0, height - tH) / 2.0)
            # pad the image and force 32x32 dimensions
            padded = cv2.copyMakeBorder(
                thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            padded = cv2.resize(padded, (width, height))
            # prepare the padded image for classification via our
            # handwriting OCR model
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)
            # update our list of characters that will be OCR'd
            chars.append((padded, (x, y, w, h)))
    return chars


def get_dsyl_inputs(contours, gray, img_height):
    chars = set_box(contours, gray, 150, 150, img_height)
    chars = np.array([cv2.cvtColor(c[0], cv2.COLOR_BGR2RGB) for c in chars], dtype="float32")
    return chars


def get_hwt_inputs(contours, gray, img_height):
    chars = set_box(contours, gray, 28, 28, img_height)
    chars = np.array([c[0] for c in chars], dtype="float32")
    return chars
