import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml
import torch.nn as nn
from pathlib import Path
import logging

def prepare_test_data(params):
    data_transforms_test = transforms.Compose([
        transforms.Resize(params['preprocessing']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir_test = Path.cwd() / "data" / "01_raw" / "disease_dataset" / "new-plant-diseases-dataset"
    test_dataset = datasets.ImageFolder(root=f"{data_dir_test}/test", transform=data_transforms_test)
    test_loader = DataLoader(test_dataset, batch_size=params['training']['batch_size'], shuffle=False)
    
    logging.info("Test Data Loaded!")
    
    return test_loader

def evaluate_model(params, model_trained):
    test_loader = prepare_test_data(params)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model_trained(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
