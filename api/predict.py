import numpy as np
import json
import aiohttp as http
from preprocess import get_hwt_inputs, get_dsyl_inputs, preprocess_image
from os import getenv, path


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


async def predict(filename, expected):
    # preprocess the input image
    contours, gray, img, img_height = preprocess_image(filename)

    # get the handwritten inputs
    hwt_inputs = get_hwt_inputs(contours, gray, img_height)

    # get the prediction from tf-serving-handwritten
    hwt_preds = await predict_handwritten(hwt_inputs.tolist())

    # define the list of label names
    labelAlpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelAlpha = [l for l in labelAlpha]
    labelDysl = ['Corrected', 'Normal', 'Reversal']

    hwt_result = ''
    for pred in hwt_preds:
        i = np.argmax(pred)
        prob = pred[i]
        label = labelAlpha[i]
        hwt_result += label.lower()
    print('Handwritten: {}'.format(hwt_result))

    is_correct = hwt_result == expected

    dysl_result = []
    if not is_correct:
        # if incorrect, get the dyslexia inputs
        dysl_inputs = get_dsyl_inputs(contours, gray, img_height)

        # get the prediction from tf-serving-dyslexia
        dysl_preds = await predict_dyslexia(dysl_inputs.tolist())
        for pred in dysl_preds:
            i = np.argmax(pred)
            prob = pred[i]
            label = labelDysl[i]
            dysl_result.append(label)
    print('Dyslexia: {}'.format(dysl_result))
    dysl_result = 'Reversal' in dysl_result

    return hwt_result, dysl_result


async def predict_handwritten(inputs):
    # Make request to tf-serving-handwritten
    async with http.ClientSession() as session:
        payload = { 'instances': inputs }
        jsonPayload = json.dumps(payload)
        base_url = getenv("TF_SERVING_HANDWRITTEN_URL")
        model_version = getenv("HANDWRITTEN_MODEL_VERSION")
        tf_serving_url = ''
        if base_url is None:
            raise Exception('No TF Serving Handwritten URL Provided')
        if model_version is None:
            tf_serving_url = f'{base_url}/v1/models/consumption:predict'
        else:
            tf_serving_url = f'{base_url}/v1/models/consumption/versions/{model_version}:predict'
        async with session.post(tf_serving_url, data=jsonPayload) as response:
            jsonResponse = await response.json()
            return jsonResponse['predictions']


async def predict_dyslexia(inputs):
    # Make request to tf-serving-dyslexia
    async with http.ClientSession() as session:
        payload = { 'instances': inputs }
        jsonPayload = json.dumps(payload)
        base_url = getenv("TF_SERVING_DYSLEXIA_URL")
        model_version = getenv("DYSLEXIA_MODEL_VERSION")
        tf_serving_url = ''
        if base_url is None:
            raise Exception('No TF Serving Dyslexia URL Provided')
        if model_version is None:
            tf_serving_url = f'{base_url}/v1/models/consumption:predict'
        else:
            tf_serving_url = f'{base_url}/v1/models/consumption/versions/{model_version}:predict'
        async with session.post(tf_serving_url, data=jsonPayload) as response:
            jsonResponse = await response.json()
            return jsonResponse['predictions']
