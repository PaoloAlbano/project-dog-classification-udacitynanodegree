import torch
import torchvision.transforms as T
import torchvision.models as models
import cv2
from pathlib import Path

from PIL import Image
from utils import get_class_names, test_data_transform

class_names = get_class_names()
model_transfer = None
VGG16 = None
face_cascade = None

use_cuda = False #for inference no gpu

def face_detector(img_path):
    global face_cascade
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def load_vgg16_model():
    global VGG16
    if VGG16 is None:
        VGG16 =  models.vgg16(pretrained=True, progress=False)

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to
    predicted ImageNet class for image at specified path

    Args:
        img_path: path to an image

    Returns:
        Index corresponding to VGG-16 model's prediction
    '''

    image = Image.open(img_path).convert('RGB')
    # from https://pytorch.org/docs/stable/torchvision/models.html
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    transformator = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    transformed_image = transformator(image)[:3, :, :].unsqueeze(0)

    if use_cuda:
        transformed_image = transformed_image.cuda()

    output = VGG16(transformed_image)
    # print(torch.max(output,1)[1])
    return torch.max(output, 1)[1].item()

def dog_detector(img_path):
    predicted_index = VGG16_predict(img_path)
    return predicted_index >=151 and predicted_index <=268 # true/false

def load_model_transfer_breed_dog(filename='model_transfer.pt'):
    global model_transfer

    model_transfer = models.resnet18(pretrained=True)
    model_transfer.fc = torch.nn.Linear(model_transfer.fc.in_features, 133)
    model_transfer = model_transfer.to(torch.device("cuda:0" if use_cuda else "cpu"))

    if not Path(filename).exists():
        filename = Path().resolve().parent / filename

    if model_transfer is None:
        model_transfer.load_state_dict(torch.load(filename))
        model_transfer = model_transfer.to(torch.device("cuda:0" if use_cuda else "cpu"))

def predict_breed_transfer(img_path, model):
    # load the image and return the predicted breed
    image = Image.open(img_path).convert('RGB')
    transfomed = test_data_transform(image).unsqueeze(0).to(torch.device("cuda:0" if use_cuda else "cpu"))

    model.eval()
    idx = torch.argmax(model(transfomed))
    return class_names[idx.int()].split('.')[-1]

def run_pipeline(img_path: str) -> str:
    '''
    Run inference pipeline
    :param img_path: path to image
    :return:
    '''
    try:
        if dog_detector(img_path):
            phrase = "hello, dog! Goooood booy!!!"
            phrase += f"\n you look like a {predict_breed_transfer(img_path, model_transfer)}"
        elif face_detector(img_path) > 0:
            phrase = "hello, human!!!"
            phrase += f"\n you look like a {predict_breed_transfer(img_path, model_transfer)}"
        else:
            phrase = "nothing to predict"
        return phrase
    except Exception as ex:
        print(str(ex))
        return "nothing to predict"
