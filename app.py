# -*- coding: utf-8 -*-
"""The app module, containing the app factory function."""
import os
import logging
import pandas as pd
from io import BytesIO
from PIL import Image
from img2tags.run import Img2Tags
from img2tags.color import extract_main_color
from flask import Flask, request, jsonify


logging.basicConfig(level=logging.DEBUG)

APP_DIR = os.path.abspath(os.path.dirname(__file__))  # This directory
ENCODER_PATH = os.path.abspath(os.path.join(APP_DIR, 'models/encoder-final.ckpt'))
DECODER_PATH = os.path.abspath(os.path.join(APP_DIR, 'models/decoder-final.ckpt'))
VOCAB_PATH = os.path.abspath(os.path.join(APP_DIR, 'models/vocab.pkl'))
GENERAL_PREDICTOR_PATH = os.path.abspath(os.path.join(APP_DIR, 'models/categ_predictor-final.ckpt'))
EMBED_SIZE = 512
HIDDEN_SIZE = 512
NUM_LAYER = 1
NUM_GENERAL_CATEGORIES = 3
model = Img2Tags(
    ENCODER_PATH, DECODER_PATH, VOCAB_PATH, GENERAL_PREDICTOR_PATH,
    EMBED_SIZE, HIDDEN_SIZE, NUM_LAYER, NUM_GENERAL_CATEGORIES
)

COLOR_DICT_PATH = os.path.abspath(os.path.join(APP_DIR, 'models/color_dictionary.csv'))
df_color_dict = pd.read_csv(COLOR_DICT_PATH)

app = Flask(__name__)
app.url_map.strict_slashes = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/', methods=['POST'])
def handler():
    try:
        images = request.files.getlist('images')
        logging.info('Processing %d Images', len(images))
        ret = []
        for each_img in images:
            img_name = each_img.filename
            try:
                img_binary = each_img.read()
                img_colors = extract_main_color(BytesIO(img_binary), df_color_dict)
                image = Image.open(BytesIO(img_binary))
                img_tags = model.sample_process(image)
                logging.info('Image %s (%d bytes) successfully processed', img_name, len(img_binary))
                each_ret = {
                    'colors': img_colors,
                    'tags': img_tags,
                    'filename': img_name
                }
            except Exception as e:
                each_ret = {'error': repr(e), 'filename': img_name}
                logging.error('Image %s processing error message %s', img_name, repr(e))
            ret.append(each_ret)
    except Exception as e:
        ret = {'error': repr(e)}
        logging.error(e)
    resp = jsonify(ret)
    resp.headers.add('Access-Control-Allow-Origin', '*')
    resp.headers.add('Content-Type', 'application/json')
    return resp


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8081)
