import torch
import torch.nn as nn
import torch.optim as optim
import wandb

def create_cnn_model(params: dict):
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

def train_model(model, dataloaders, params: dict):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    wandb.init(project="save_and_restore")
    for epoch in range(params['epochs']):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in dataloaders[phase]:
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                total += labels.size(0)
                correct += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = 100. * correct / total
            metrics = None
            if phase == 'train':
                metrics = {"Train Loss": epoch_loss, "Train accuracy": epoch_acc}
            else:
                metrics = {"Valid Loss": epoch_loss, "Valid accuracy": epoch_acc}
            wandb.log(metrics)
    wandb.finish()
    return model

def save_model(model_trained):
    PATH = "trained_model.pth"
    with wandb.init(project="save_and_restore") as run:
        model_artifact = wandb.Artifact(
            "trained-model", type="model",
            description="Trained NN model")

        torch.save(model_trained.state_dict(), PATH)
        model_artifact.add_file("trained_model.pth")
        wandb.save("trained_model.pth")
        run.log_artifact(model_artifact)

    return PATH