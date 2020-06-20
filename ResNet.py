import torch
import os
import time
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch import nn, optim
from utils import *
from tqdm import tqdm


__all__ = ['ResNet50', 'ResNet101','ResNet152']

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=1000, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(512, 128),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(128, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet50():
    return ResNet([3, 4, 6, 3], num_classes=3)

def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet152():
    return ResNet([3, 8, 36, 3])

def train_and_validate(model, loss_criterion, optimizer, epochs=25, patience=3, save_name='checkpoint'):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)

    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''

    start = time.time()
    history = []
    best_acc = 0.0
    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        patience=patience, verbose=False, save_name=save_name)
    for epoch in range(epochs):
        epoch_start = time.time()
        # Set to training mode
        model.train()
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels = data[0].to(device), data[1].to(device)
            # Clean existing gradients
            optimizer.zero_grad()
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            # Compute loss
            loss = loss_criterion(outputs, labels)
            # Backpropagate the gradients
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            # Compute the accuracy
            _, predictions = torch.max(outputs.data, 1)
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += (predictions == labels).sum().item()

        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()
            # Validation loop
            for i, data in enumerate(valid_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)
                # Compute loss
                loss = loss_criterion(outputs, labels)
                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)
                # Calculate validation accuracy
                _, predictions = torch.max(outputs.data, 1)
                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += (predictions == labels).sum().item()

        # Find average training loss and training accuracy
        avg_train_loss = train_loss/len(train_data)
        avg_train_acc = train_acc/len(train_data)

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/len(valid_data)
        avg_valid_acc = valid_acc/len(valid_data)

        history.append([avg_train_loss, avg_valid_loss,
                        avg_train_acc, avg_valid_acc])

        epoch_end = time.time()

        print("Epoch: {}/{}, loss: {:.3f}, acc: {:.2f}%, val_oss : {:.3f}, val_acc: {:.2f}%, Time: {:.2f}s".format(
            epoch+1, epochs, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))

        early_stopping(avg_valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    print('Finished Training')
    history = np.array(history)
    return model, history


def computeTestSetAccuracy(model, test_data, test_loader, loss_criterion):
    '''
    Function to compute the accuracy on the test set
    Parameters
        :param model: Model to test
        :param loss_criterion: Loss Criterion to minimize
    '''
    test_acc = 0.0
    test_loss = 0.0
    confusion_matrix = np.zeros((3, 3), dtype=np.int)

    # Validation - No gradient tracking needed
    with torch.no_grad():

        # Set to evaluation mode
        model.eval()

        # Validation loop
        for i, data in enumerate(tqdm(test_loader)):
            inputs, labels = data[0].to(device), data[1].to(device)
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            # Compute loss
            loss = loss_criterion(outputs, labels)
            # Compute the total loss for the batch and add it to valid_loss
            test_loss += loss.item() * inputs.size(0)
            # Calculate validation accuracy
            _, predictions = torch.max(outputs.data, 1)
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean((predictions == labels).type(torch.FloatTensor))
            # Compute total accuracy in the whole batch and add to valid_acc
            test_acc += (predictions == labels).sum().item()
            for t, p in zip(labels.view(-1), predictions.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(
                i, loss.item(), acc.item()))

    # Find average test loss and test accuracy
    avg_test_loss = test_loss/len(test_data)
    avg_test_acc = test_acc/len(test_data)

    print("Test accuracy : " + str(avg_test_acc))
    return confusion_matrix, avg_test_loss, avg_test_acc


if __name__ == '__main__':
    model_name = os.path.basename(__file__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print("Found", device, "device")

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image_transforms = {'train': transforms.Compose([transforms.RandomRotation(40),
                                                     transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
                                                     transforms.RandomHorizontalFlip(p=0.5),
                                                     transforms.Resize((224, 224)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean, std),
                                                     ]),
                        'valid': transforms.Compose([transforms.Resize((224, 224)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean, std),
                                                    ])}
    
    train_dir = './data/training2'
    validation_dir = './data/validation2'
    BATCH_SIZE = 64
    
    train_data = ImageFolder(
        root=train_dir, transform=image_transforms['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True)

    valid_data = ImageFolder(
        root=validation_dir, transform=image_transforms['valid'])
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_data, batch_size=BATCH_SIZE, shuffle=True)

    model = ResNet50()
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)


    model, history = train_and_validate(
            model, criterion, optimizer, epochs=150, patience=7, save_name = model_name)

    show_train_history(history[:, 0], history[:, 1], monitor = 'Loss', save_name = model_name + "_loss_history")
    show_train_history(history[:, 2], history[:, 3], monitor = 'Accuracy', save_name = model_name + "_acc_history")

    cnn = ResNet50()
    cnn.to(device)
    cnn.load_state_dict(torch.load(model_name + '.pt'))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr = 1e-3)
    confusion_matrix, avg_val_loss, avg_val_acc = computeTestSetAccuracy(
        cnn, valid_data, valid_loader, criterion)
    print_confusion_matrix(
        confusion_matrix, ['A', 'B', 'C'], figsize=(4, 3), fontsize=12, save_name= model_name +"_confusion")


    test_image_name  = os.listdir('./data/test_black')
    test_image_name.sort()
    prediction = predict(cnn, test_image_name, transform = image_transforms['valid'], save_name = model_name)
