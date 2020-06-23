#Used for pre download the vgg16 models
import torchvision.models as models
VGG16 =  models.vgg16(pretrained=True, progress=True)