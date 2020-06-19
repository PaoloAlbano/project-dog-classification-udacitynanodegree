import logging
import falcon
from falcon_multipart.middleware import MultipartMiddleware

logger = logging.getLogger()

from .cnn_dog import run_pipeline, load_model_transfer_breed_dog, load_vgg16_model

logger.info("Loading VGG16 model")
load_vgg16_model()
logger.info("Loading model transfer")
load_model_transfer_breed_dog()



app = falcon.API(middleware=[MultipartMiddleware()])

