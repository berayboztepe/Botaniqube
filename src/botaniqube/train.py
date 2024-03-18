import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import yaml

# Load parameters from parameters.yml
with open('../../conf/base/parameters.yml', 'r') as file:
    params = yaml.safe_load(file)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(params['preprocessing']['image_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(params['preprocessing']['augmentation'][0]['rotation_range']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(params['preprocessing']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Define datasets
data_dir = 'data/01_raw/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)'
image_datasets = {x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=params['training']['batch_size'], shuffle=True, num_workers=4) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_layers, hidden_units):
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
            nn.Linear(16 * (params['preprocessing']['image_size'][0] // (2 ** params['network']['num_layers'])) * (params['preprocessing']['image_size'][1] // (2 ** params['network']['num_layers'])), hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, len(image_datasets['train'].classes))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

model = CNN(params['network']['num_layers'], params['network']['hidden_units'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=params['training']['learning_rate'])

# Train the model
for epoch in range(params['training']['epochs']):
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

        print(f'Epoch {epoch+1}/{params["training"]["epochs"]} | {phase} loss: {epoch_loss:.4f} | {phase} accuracy: {epoch_acc:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'vanilla_model.pth')
