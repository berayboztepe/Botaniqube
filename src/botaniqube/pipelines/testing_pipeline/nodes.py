import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from ..training_pipeline.nodes import create_cnn_model
import wandb
import os

def prepare_test_data(params: dict):
    img_size = params['image_size']
    data_transforms_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir_test = Path.cwd() / "data" / "01_raw" / "disease_dataset"
    test_dataset = datasets.ImageFolder(root=f"{data_dir_test}/test", transform=data_transforms_test)
    test_loader = DataLoader(test_dataset, shuffle=True)
    return test_loader

def fetch_model(params):
    with wandb.init(project="save_and_restore") as run:
        model_artifact = run.use_artifact("trained-model:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "trained_model.pth")
        trained_model = create_cnn_model(params)
        trained_model.load_state_dict(torch.load(model_path))
    return trained_model

def evaluate_model(trained_model,test_loader):
    wandb.init(project="save_and_restore")
    trained_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = trained_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels.data)

    accuracy = 100. * correct / total
    wandb.log({'accuracy': accuracy})
    wandb.finish()

    return accuracy
