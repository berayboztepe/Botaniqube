import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

classes = {
    0:'Apple___Apple_scab',
    1:'Apple___Black_rot',
    2:'Apple___Cedar_apple_rust',
    3:'Apple___healthy',
    4:'Blueberry___healthy',
    5:'Cherry_(including_sour)___healthy',
    6:'Cherry_(including_sour)___Powdery_mildew',
    7:'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8:'Corn_(maize)___Common_rust_',
    9:'Corn_(maize)___healthy',
    10:'Corn_(maize)___Northern_Leaf_Blight',
    11:'Grape___Black_rot',
    12:'Grape___Esca_(Black_Measles)',
    13:'Grape___healthy',
    14:'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    15:'Orange___Haunglongbing_(Citrus_greening)',
    16:'Peach___Bacterial_spot',
    17:'Peach___healthy',
    18:'Pepper,_bell___Bacterial_spot',
    19:'Pepper,_bell___healthy',
    20:'Potato___Early_blight',
    21:'Potato___healthy',
    22:'Potato___Late_blight',
    23:'Raspberry___healthy',
    24:'Soybean___healthy',
    25:'Squash___Powdery_mildew',
    26:'Strawberry___healthy',
    27:'Strawberry___Leaf_scorch',
    28:'Tomato___Bacterial_spot',
    29:'Tomato___Early_blight',
    30:'Tomato___healthy',
    31:'Tomato___Late_blight',
    32:'Tomato___Leaf_Mold',
    33:'Tomato___Septoria_leaf_spot',
    34:'Tomato___Spider_mites Two-spotted_spider_mite',
    35:'Tomato___Target_Spot',
    36:'Tomato___Tomato_mosaic_virus',
    37:'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
}

def create_cnn_model(params: dict):
    image_size = params["image_size"]
    num_layers = params["num_layers"]
    hidden_units = params['hidden_units']
    num_classes = params["num_classes"]

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(128 * 14 * 14, 512)
            self.fc2 = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = self.pool(nn.functional.relu(self.conv3(x)))
            x = self.pool(nn.functional.relu(self.conv4(x)))
            x = x.view(-1, 128 * 14 * 14)
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    return CNN()

def load_model(params):
    # load model from local path
    model_path = "deployment/trained_model.pth"

    model = create_cnn_model(params)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def _load_image(params: dict):
    img_size = params['image_size']
    batch_size = params['batch_size']
    data_transforms_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir_test = Path.cwd() / "data" / "01_raw" / "disease_dataset"
    test_dataset = datasets.ImageFolder(root=f"{data_dir_test}/test", transform=data_transforms_test)
    test_loader = DataLoader(test_dataset, shuffle=False)
    
    return test_loader    

def predict(model, test_loader):
    correct = 0
    total = 0
    predictions_classes = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels.data)
            prediction_int = int(predicted.numpy())
            predictions_classes.append(classes[prediction_int])

    accuracy = 100. * correct / total
    return accuracy, predictions_classes



if __name__=="__main__":
    params = {"type": "CNN", "num_layers": 5, "hidden_units": 256, "num_classes": 38, "image_size": [224, 224], "batch_size": 5}

    model = load_model(params)
    test_loader = _load_image(params)
    accuracy, predictions_classes = predict(model, test_loader)
    print(accuracy)
    print(predictions_classes)