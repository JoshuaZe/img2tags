# -*- coding: utf-8 -*-
"""The app module, containing the app factory function."""
import os
import logging
from io import BytesIO
from PIL import Image
from img2tags.vocabulary import Vocabulary
from img2tags.run import Img2Tags
from flask import Flask, request, jsonify
import gunicorn.app.base


logging.basicConfig(level=logging.DEBUG)

APP_DIR = os.path.abspath(os.path.dirname(__file__))  # This directory
ENCODER_PATH = os.path.abspath(os.path.join(APP_DIR, 'models/encoder-final.ckpt'))
DECODER_PATH = os.path.abspath(os.path.join(APP_DIR, 'models/decoder-final.ckpt'))
VOCAB_PATH = os.path.abspath(os.path.join(APP_DIR, 'models/vocab.pkl'))
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYER = 1
model = Img2Tags(ENCODER_PATH, DECODER_PATH, VOCAB_PATH, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYER)

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
                image = Image.open(BytesIO(img_binary))
                img_tags = model.sample_process(image)
                logging.info('Image %s (%d bytes) successfully processed', img_name, len(img_binary))
                each_ret = {'tags': img_tags, 'filename': img_name}
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


class StandaloneApplication(gunicorn.app.base.BaseApplication):

    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items()
                  if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


if __name__ == '__main__':
    options = {
        'bind': '%s:%s' % ('0.0.0.0', '8081'),
        'workers': 4,
    }
    StandaloneApplication(app, options).run()
    # app.run(debug=True, port=8081)
