import json
import aiofiles
from fastapi import FastAPI, UploadFile, Form
from predict import predict
from os import getenv, path, environ, remove
from typing import Union

app = FastAPI()

@app.post('/api/v1/handwritings')
async def analyze_handwriting(handwriting: UploadFile, data = Form(), version: Union[str, None] = None):
    out_file_path = 'temp/' + handwriting.filename
    in_file = handwriting
    parsed_json = json.loads(data)
    expected_word = parsed_json['expectedWord']

    async with aiofiles.open(out_file_path, 'wb') as out_file:
        while content := await in_file.read(1024):
            await out_file.write(content)

    try:
        handwritten_result, dyslexia_result = await predict(out_file_path, parsed_json['expectedWord'], version)
    except:
        remove(out_file_path)
        raise Exception('Server Error')
    # is_correct, predicted = check_prediction(parsed_json['expectedWord'], handwritten_result)
    is_correct = expected_word == handwritten_result
    remove(out_file_path)

    return {
        'expectedWord': expected_word,
        'predictedWord': handwritten_result,
        'dyslexia': dyslexia_result,
        'isCorrect': is_correct
    }

# def check_prediction(expected: str, predict: str):
#     output = [char for char in expected]
#     for i in predict:
#         try:
#             output.remove(i.lower())
#         except:
#             pass
#     is_correct = len(output) == 0
#     predicted = [char for char in expected]
#     for i in output:
#         try:
#             predicted.remove(i.lower())
#         except:
#             pass
#     predicted = ''.join(predicted)
#     return is_correct, predicted
