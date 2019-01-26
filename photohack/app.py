import os
from collections import namedtuple
from enum import Enum
from pathlib import Path
from uuid import uuid1
import scipy.misc

from flask import Flask, redirect, request, send_file, url_for

from photohack.parallaxer.models import MonodepthModel, DeeplabModel
from photohack.parallaxer.pipeline import process_file

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOADS_DIR = BASE_DIR / 'uploads'
if not UPLOADS_DIR.exists():
    UPLOADS_DIR.mkdir()
MODELS_DIR = BASE_DIR / 'models'
if not MODELS_DIR.exists():
    MODELS_DIR.mkdir()
ALLOWED_EXTENSIONS = (
    'jpg',
    'jpeg',
    'png',
)

#: Models
MODELS = {
    'depth': MonodepthModel(MODELS_DIR / 'model_city2kitti_resnet.pb'),
    'segmentation': DeeplabModel(MODELS_DIR / 'deeplabv3.pb'),
    'inpainting': None}

ImageStepFile = namedtuple('ImageStep', 'name mime')


class ImageStep(Enum):
    original = ImageStepFile('original.jpg', 'image/jpeg')
    depth_estimation = ImageStepFile('depth_estimation.jpg', 'image/jpeg')
    segmentation = ImageStepFile('segmentation.jpg', 'image/jpeg')
    inpainting = ImageStepFile('inpainting.jpg', 'image/jpeg')
    final = ImageStepFile('final.gif', 'image/gif')


app = Flask(__name__)


@app.route('/')
def index():
    return """
    <!doctype html>
    <title>Upload image</title>
    <h1>Upload image</h1>
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    """


@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files.get('file')
    if not uploaded_file:
        return 'No file uploaded.', 400
    if not uploaded_file.filename:
        return 'No file selected.', 400
    try:
        ext = uploaded_file.filename.rsplit('.', 1)[1].lower()
    except KeyError:
        return 'File extension not found.', 400
    if ext not in ALLOWED_EXTENSIONS:
        return 'File extension not allowed.', 400
    results_id = str(uuid1())
    new_dir = UPLOADS_DIR / results_id
    new_dir.mkdir()
    original_filename = str(new_dir / ImageStep.original.value.name)
    uploaded_file.save(original_filename)
    seg_map, depth_map = process_file(original_filename, MODELS['depth'],
                                      MODELS['segmentation'])
    scipy.misc.imsave(ImageStep.depth_estimation.name, depth_map)
    scipy.misc.imsave(ImageStep.segmentation.name, seg_map)
    return redirect(url_for('results', results_id=results_id))


@app.route('/image/<uuid:results_id>/<step>')
def image(results_id, step):
    try:
        step = ImageStep[step]
    except KeyError:
        return 'Invalid image step.', 400
    image_path = UPLOADS_DIR / str(results_id) / step.value.name
    return send_file(str(image_path), mimetype=step.value.mime)


@app.route('/results/<uuid:results_id>')
def results(results_id):
    table_rows = []
    for step in ImageStep:
        img_src = url_for('image', results_id=results_id, step=step.name)
        row = f"""
        <tr>
            <td>{step.name}</td>
            <td><img src="{img_src}" height="256" width="512"></td>
        </tr>
        """
        table_rows.append(row)
    return f"""
    <!doctype html>
    <title>Results</title>
    <h1>Results</h1>
    <table>
        <thead>
            <tr>
                <td><b>Step</b></td>
                <td><b>Image</b></td>
            </tr>
        </thead>
        <tbody>
            {''.join(table_rows)}
        </tbody>
    </table>
    <a href="{url_for('index')}">Return to upload</a>
    """

app.run()
