import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml
import torch.nn as nn
from pathlib import Path
import logging
from ..training_pipeline.nodes import create_cnn_model
import wandb

def prepare_test_data(img_size, batch_size):
    data_transforms_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir_test = Path.cwd() / "data" / "01_raw" / "disease_dataset"
    test_dataset = datasets.ImageFolder(root=f"{data_dir_test}/test", transform=data_transforms_test)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    logging.info("Test Data Loaded!")
    
    return test_loader

def evaluate_model(img_size,batch_size,PATH,params):
    model = create_cnn_model(img_size,params)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    test_loader = prepare_test_data(img_size,batch_size)
    correct = 0
    err = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            err += (predicted != labels).sum().item()
            correct += (predicted == labels).sum().item()

    loss = 100 * err / total
    accuracy = 100 * correct / total
    metrics = {"loss": loss, "accuracy": accuracy}
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    wandb.init(project='botaniqube', entity='antoni-krzysztof-czapski')

    wandb.log({'accuracy': accuracy})

    wandb.finish()

    return metrics
