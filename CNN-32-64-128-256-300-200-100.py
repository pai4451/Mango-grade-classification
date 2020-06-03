import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from utils import *


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv = nn.Sequential( 
                # 3 * 224 * 224 -> 32 * 112 * 112
                nn.Conv2d(3, 32, 3, padding = 1), 
                nn.BatchNorm2d(32),
                nn.Dropout(0.4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2),

                # 32 * 112 * 112 -> 64 * 56 * 56
                nn.Conv2d(32, 64, 3, padding = 1),
                nn.BatchNorm2d(64),
                nn.Dropout(0.4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2),

                # 64 * 56 * 56 -> 128 * 28 * 28 
                nn.Conv2d(64, 128, 3, padding = 1), 
                nn.BatchNorm2d(128),
                nn.Dropout(0.4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2),

                # 128 * 28 * 28 -> 256 * 14 * 14
                nn.Conv2d(128, 256, 3, padding = 1), 
                nn.BatchNorm2d(256),
                nn.Dropout(0.4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2),
            )

        self.fc = nn.Sequential(
                nn.Linear(256 * 14 * 14, 300),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(300, 200),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(200, 100),
                nn.ReLU(),
                nn.Dropout(0.5),
            )

        self.out = nn.Linear(100, 3)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.out(x)
        return x

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
        for i, data in enumerate(train_loader):
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
        for i, data in enumerate(test_loader):
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
    model_name = "CNN-32-64-128-256-300-200-100"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print("Found", device, "device")
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_transforms = {'train': transforms.Compose([transforms.Resize((224, 224)),
                                                     MyRotationTransform(angles=15),
                                                     transforms.RandomHorizontalFlip(p=0.5),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(
                                                         mean, std),
                                                     ]),
                        'valid': transforms.Compose([transforms.Resize((224, 224)),
                                                     MyRotationTransform(angles=15),
                                                     transforms.RandomHorizontalFlip(p=0.5),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(
                                                     mean, std),
                                                     ])}
    BATCH_SIZE = 64
    train_data = MangoDataset(
        './data/train.csv', './data/C1-P1_Train', image_transforms['train'])
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    valid_data = MangoDataset(
        './data/dev.csv', './data/C1-P1_Dev', image_transforms['valid'])
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_data, batch_size=BATCH_SIZE, shuffle=False)

    cnn = CNN()
    cnn.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr = 0.001, betas = (0.9, 0.99))

    cnn, history = train_and_validate(
        cnn, criterion, optimizer, epochs=100, patience=5, save_name = model_name)

    show_train_history(history[:, 0], history[:, 1], monitor = 'Loss', save_name = model_name + "_loss_history")
    show_train_history(history[:, 2], history[:, 3], monitor = 'Accuracy', save_name = model_name + "_acc_history")

    cnn = CNN()
    cnn.to(device)
    cnn.load_state_dict(torch.load(model_name + '.pt'))

    confusion_matrix, avg_val_loss, avg_val_acc = computeTestSetAccuracy(
        cnn, valid_data, valid_loader, criterion)
    print_confusion_matrix(
        confusion_matrix, ['A', 'B', 'C'], figsize=(4, 3), fontsize=12, save_name= model_name +"_confusion")
