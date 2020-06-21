import logging
import json
import uuid

import falcon
from falcon_multipart.middleware import MultipartMiddleware
from pathlib import Path

logger = logging.getLogger()

from cnn_dog import run_pipeline, load_model_transfer_breed_dog, load_vgg16_model

logger.info("Loading VGG16 model")
load_vgg16_model()
logger.info("Loading model transfer")
load_model_transfer_breed_dog()



app = falcon.API(middleware=[MultipartMiddleware()])
PATH_IMAGES = 'images/'

class Inference:

    def on_post(self, req, resp, **params):
        logger.debug("Start inference")
        if 'file' not in req.params.keys():
            resp.status = falcon.HTTP_400
            resp.body = json.dumps({"error": "Please send a file"})
            resp.content_type = falcon.MEDIA_JSON
        else:
            data = req.get_param("file")
            name_file = str(uuid.uuid4()) + "." + data.filename.split(".")[-1]

            try:
                path_file = Path(PATH_IMAGES) / name_file
                open(str(path_file), "wb").write(data.file.read())

                #start inference
                response = run_pipeline(str(path_file))

                resp.status = falcon.HTTP_200
                resp.body = json.dumps({"response": response, "name": name_file })
            except Exception as ex:
                resp.status = falcon.HTTP_500
                resp.body = json.dumps({"error": str(ex), "name": name_file })


class Images:

    def on_get(self, req, resp, name):
        logger.debug("get Images")

        image_path = Path(PATH_IMAGES) / name
        logger.info("GET: " + str(image_path))
        if image_path.exists():
            with open(str(image_path), 'rb') as img_file:
                resp.status = falcon.HTTP_200
                resp.body = img_file.read()

                if image_path.suffix == '.png':
                    resp.content_type = falcon.MEDIA_PNG
                if image_path.suffix == '.jpeg':
                    resp.content_type = falcon.MEDIA_JPEG
        else:
            resp.status = falcon.HTTP_400
            resp.body = json.dumps({"error": "image not found"})

class Home:

    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.content_type = falcon.MEDIA_HTML
        with open("index.html", "r") as file:
            resp.body = file.read()


app.add_route('/image/{name}', Images())
app.add_route('/inference', Inference())
app.add_route('/', Home())
app.add_route('/index', Home())


