import torchvision.transforms as T

def get_class_names(filename='class_names.txt'):
    names = []
    with open(filename) as f:
        for l in f.readlines():
            names.append(l)
    return names


test_data_transform = T.Compose([
        T.Resize(size=(224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


