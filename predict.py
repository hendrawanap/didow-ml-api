# import the necessary packages
import numpy as np
import imutils
import wget
import cv2
from os import getenv, path
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours

def shadow_remove(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(
            diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov

def preprocess_image(img):
    image = cv2.imread(img)
    image = imutils.resize(image, width=300)
    image = shadow_remove(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # perform edge detection, find contours in the edge map, and sort the resulting contours from left-to-right
    edged = cv2.Canny(blurred, 30, 150)
    # ret, im = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    return cnts, gray, image

# initialize the list of contour bounding boxes and associated characters that we'll be OCR'ing
def set_box(contours, gray, width, height):
    chars = []
    for c in contours:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # filter out bounding boxes, ensuring they are neither too small
        # nor too large
        if (w >= 23 and w <= 150) and (h >= 40 and h <= 125):
            # extract the character and threshold it to make the character
            # appear as *white* (foreground) on a *black* background, then
            # grab the width and height of the thresholded image
            roi = gray[y:y + h, x:x + w]
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
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
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            padded = cv2.resize(padded, (width, height))
            # prepare the padded image for classification via our
            # handwriting OCR model
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)
            # update our list of characters that will be OCR'd
            chars.append((padded, (x, y, w, h)))
    return chars

def getModel():
  handwritten_url = getenv('MODEL_HANDWRITTEN_PUBLIC_URL')
  dyslexia_url = getenv('MODEL_DYSLEXIA_PUBLIC_URL')
  wget.download(handwritten_url, './model/handwritten/model_combine.h5')
  wget.download(dyslexia_url, './model/dyslexia/model_dyslexia.h5')

def check_prediction(expected: str, predict: str):
    output = [char for char in expected]
    for i in predict:
        try:
            output.remove(i.lower())
        except:
            pass
    is_correct = len(output) == 0
    predicted = [char for char in expected]
    for i in output:
        try:
            predicted.remove(i.lower())
        except:
            pass
    predicted = ''.join(predicted)
    return is_correct, predicted

def predict(filename, expected):
  # load the input image from disk, convert it to grayscale, and blur it to reduce noise
  dyslexia_path = './model/dyslexia/model_dyslexia.h5'
  handwritten_path = './model/handwritten/model_combine.h5'
  if not (path.exists(dyslexia_path) & path.exists(handwritten_path)) :
    getModel()

  hwt_model = load_model(handwritten_path)
  dysl_model = load_model(dyslexia_path)

  # extract the bounding box locations and padded characters
  contours, gray, image = preprocess_image(filename)
  chars = set_box(contours, gray, 28, 28)
  dysl_chars = set_box(contours, gray, 150, 150)

  # OCR the characters using our handwriting recognition model
  boxes = [b[1] for b in chars]
  chars = np.array([c[0] for c in chars], dtype="float32")
  hwt_preds = hwt_model.predict(chars)

  # define the list of label names
  labelAlpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  labelAlpha = [l for l in labelAlpha]

  labelDysl = ['Corrected', 'Normal', 'Reversal']

  result = ''
  for pred in hwt_preds:
    i = np.argmax(pred)
    prob = pred[i]
    label = labelAlpha[i]
    result += label
  print('result: {}'.format(result))

  is_correct, predicted = check_prediction(expected, result)
  print(is_correct, predicted)

  dysl_result = []
  if not is_correct:
    dysl_chars = np.array([cv2.cvtColor(c[0], cv2.COLOR_BGR2RGB) for c in dysl_chars], dtype="float32")
    dysl_preds = dysl_model.predict(dysl_chars)
    for pred in dysl_preds:
        i = np.argmax(pred)
        prob = pred[i]
        label = labelDysl[i]
        dysl_result.append(label)
    print(dysl_result)

  return result, dysl_result
