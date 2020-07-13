import os
import logging

from flask import Flask, flash, request, url_for, render_template, redirect, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import tensorflow.keras.backend as K
from nn_model import predict

from nn_model.config import config

_logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = config.TEMP_IMAGE_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.secret_key = "K8pl6793AsqrT1Hd3"
outcome = {}


def allowed_file(filename):
    return "." in filename and \
            filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route("/health", methods=["GET"])
def health():
    if request.method == "GET":
        _logger.info("health status OK")
        return "ok"


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # if user doesnt select file, browser also
        # submit an empty part without filename
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            outcome["Filename"] = url_for("uploaded_file", filename=filename)
            _logger.info("image uploaded and sent to prediction")
            return redirect(url_for("pred", filename=filename))
    return render_template('img_upload.html')


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/predict/<filename>", methods=["GET", "POST"])
def pred(filename):
    K.clear_session()
    result, proba = predict.make_single_prediction(app.config["UPLOAD_FOLDER"] / filename)
    print("Confidence: {}".format(proba))
    print("Result: {}".format(result))
    outcome["Result"] = int(result)
    outcome["Confidence"] = round(float(proba), 3)
    _logger.info("prediction made and result sent")
    return render_template("result.html", **outcome)


def main():
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()