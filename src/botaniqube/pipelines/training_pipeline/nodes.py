import torch
import torch.nn as nn
import torch.optim as optim
from kedro.pipeline import node
import logging
from pathlib import Path

def create_cnn_model(image_size: tuple, params: dict):
    num_layers = params["num_layers"]
    hidden_units = params['hidden_units']
    num_classes = params["num_classes"]

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                *[nn.Sequential(
                    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ) for _ in range(num_layers - 1)]
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(16 * (image_size[0] // (2 ** num_layers)) * (image_size[1] // (2 ** num_layers)), hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, num_classes)
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layers(x)
            return x
    logging.info("CNN Model Created!")
    return CNN()

def train_model(model, dataloaders, dataset_sizes, training):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(training['epochs']):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            correct = 0

            for inputs, labels in dataloaders[phase]:
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = correct.double() / dataset_sizes[phase]

            print(f'Epoch {epoch+1}/{epoch} | {phase} loss: {epoch_loss:.4f} | {phase} accuracy: {epoch_acc:.4f}')
    logging.info("Model Trained!")
    return model

def save_model(model_trained):
    torch.save(model_trained.state_dict(), Path.cwd())
    logging.info("Model Saved!")