from flask import Flask, request, url_for, render_template
from nn_model.config import config
import time
import os

app = Flask(__name__)


@app.route("/upload_image", methods=["GET", "POST"])
def upload_file():
    return render_template('img_upload.html')


@app.route("/result", methods=["POST"])
def process_upload():
    result_em = {}
    request_timestamp = int(time.time()*1000)
    result_em["timestamp"] = request_timestamp

    img_file = request.files["image"]

    img_filename = "img_upload_" + str(request_timestamp) + ".jpg"
    img_filepath = os.path.join(config.TEMP_IMAGE_DIR, img_filename)
