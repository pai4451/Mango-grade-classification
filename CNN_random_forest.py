import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.init import xavier_uniform_
from torchvision.datasets import ImageFolder
from dropblock import DropBlock2D, LinearScheduler
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier


from utils_ import *

class Mish(nn.Module):
    def forward(self, input):
        return input * torch.tanh(F.softplus(input))

mode = 'test'
BATCH_SIZE = 64
grayscale = False
do_dropblock = True
forest = True

class CNN(nn.Module):
    def __init__(self, grayscale = False):
        super(CNN, self).__init__()

        # hyperparameters
        # Activation = Mish()
        Activation = nn.ReLU()
        drop_prob = 0.2
        block_size = 5

        kernel_sizes = np.array([7,3,3,3,3])
        conv_strides = np.array([3,1,1,1,1])
        pad_sizes    = (kernel_sizes-1)/2
        out_channels = np.array([64,128,256,256,512])
        pool_sizes   = np.array([2,2,2,2,2])
        pool_strides = np.array([2,2,2,2,2])

        fcs = [512, 100] #512, 100

        if do_dropblock:
            self.Dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_prob, block_size=block_size),
                start_value=0.,
                stop_value=drop_prob,
                nr_steps=5000
            )

        conv_layers = []
        assert len(kernel_sizes)==len(out_channels)==len(pool_sizes), "inconsistent layer length"
        layers = range(len(kernel_sizes))
        input_dim = 1 if grayscale else 3
        out_channels = np.insert(out_channels,0,input_dim)
        for l in layers:
            conv_layers.append(nn.Conv2d(out_channels[l], out_channels[l+1], kernel_sizes[l], 
                stride = conv_strides[l], padding = int(pad_sizes[l])))
            conv_layers.append(nn.BatchNorm2d(out_channels[l+1]))
            # conv_layers.append(nn.Dropout(0.1))
            if do_dropblock:
                conv_layers.append(self.Dropblock)
            conv_layers.append(Activation)
            if pool_sizes[l]:
                conv_layers.append(nn.MaxPool2d(int(pool_sizes[l]), stride = int(pool_strides[l])))

        self.conv = nn.Sequential(*conv_layers)

        # compute input shape of FCs
        x = torch.zeros([1,input_dim,224,224])
        x = self.conv(x)
        x = x.view(1, -1)
        input_FC = x.size(1)

        FCs = []
        layers = range(len(fcs))
        # input length: input_FC
        fcs = np.insert(fcs,0,input_FC)
        for l in layers:
            FCs.append(nn.Linear(fcs[l], fcs[l+1]))
            FCs.append(Activation)
            FCs.append(nn.BatchNorm1d(fcs[l+1]))
            FCs.append(nn.Dropout(0.5))

        self.fc = nn.Sequential(*FCs)

        self.out = nn.Linear(fcs[-1], 3) # three categories: class A, B, C

    def forward(self, x):
        if do_dropblock:
            self.Dropblock.step()  # increment number of iterations
        x = self.conv(x)
        x = x.view(x.size(0), -1) # flattened
        if not do_dropblock:
            x = F.dropout(x, p=0.5)
        x = self.fc(x)
        x = self.out(x)
        return x

    def get_conv_output(self, x):
        if do_dropblock:
            self.Dropblock.step()  # increment number of iterations
        x = self.conv(x)
        x = x.view(x.size(0), -1) # flattened
        return x.cpu().numpy()

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

    print("Training...")
    start = time.time()
    history = []
    best_acc = 0.0
    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        patience=patience, verbose=False, save_name=save_name)
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
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
            print("Validating...")
            for i, data in enumerate(tqdm(valid_loader)):
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

        print("Epoch: {}/{}, loss: {:.3f}, acc: {:.2f}%, val_loss : {:.3f}, val_acc: {:.2f}%, Time: {:.2f}s".format(
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
        print("Testing...")
        # validation loop
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
            print("Test Batch number: {:03d}, Test Loss: {:.4f}, Accuracy: {:.4f}".format(
                i, loss.item(), acc.item()))

    # Find average test loss and test accuracy
    avg_test_loss = test_loss/len(test_data)
    avg_test_acc = test_acc/len(test_data)

    print("Test accuracy : " + str(avg_test_acc))
    return confusion_matrix, avg_test_loss, avg_test_acc

def RF(cnn_model, mode = 'train'):
    with torch.no_grad():
        if mode == 'train':
            N = 5600
            loader = train_loader
        elif mode == 'valid':
            N = 800
            loader = valid_loader

        mul = 2
        RF_X = np.zeros((mul*N,2048))
        RF_Y = np.zeros((mul*N,))
        for i_ in range(mul):
            for i, data in enumerate(tqdm(loader)):
                X, Y = data[0].to(device), data[1].to(device)
                if i == len(loader)-1:
                    batch = 32
                else:
                    batch = 64
                RF_X[N*i_ + batch*i : N*i_ + batch*(i+1),:] = cnn_model.get_conv_output(X)
                RF_Y[N*i_ + batch*i : N*i_ + batch*(i+1)] = Y.cpu().numpy()

        if mode == 'train':
            clf = RandomForestClassifier(n_estimators=512, max_depth=None,
                min_samples_split=2, random_state=0)
            clf = clf.fit(RF_X, RF_Y)
            with open('rf_model.pt', 'wb') as f:
                pickle.dump(clf, f)
        elif mode == 'valid':
            with open('rf_model.pt', 'rb') as f:
                clf = pickle.load(f)
                score = clf.score(RF_X, RF_Y)
            print(score)

    return

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print("Found", device, "device")

    if grayscale:
        mean, std = np.array([0.45]), np.array([0.22])
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    if grayscale:
        trans.insert(0, transforms.Grayscale(num_output_channels=1))

    image_transforms = {'train': transforms.Compose([transforms.RandomRotation(40),
                                                     transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
                                                     transforms.RandomHorizontalFlip(p=0.5),
                                                     transforms.Resize((224, 224)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean, std),
                                                     ]),
                        'valid': transforms.Compose([transforms.RandomRotation(40),
                                                     transforms.Resize((224, 224)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean, std),
                                                     ]),
                        }

    train_data = MangoDataset(
        './processed/train.csv', './processed/C1-P1_Train', image_transforms['train'])
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    valid_data = MangoDataset(
        './processed/dev.csv', './processed/C1-P1_Dev', image_transforms['valid'])
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=False)

    test_data = MangoDataset(
        './processed/dev.csv', './processed/C1-P1_Test', image_transforms['valid'])
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

    CNN_model_name = 'CNN-try.py'
    RF_model_name = 'CNN_rf'

    if mode == 'train':
        cnn = CNN(grayscale)
        cnn.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn.parameters(), lr = 1e-3)
        
        if not forest:
            cnn, history = train_and_validate(
                cnn, criterion, optimizer, epochs=150, patience=15, save_name = CNN_model_name)

            show_train_history(history[:, 0], history[:, 1], monitor = 'Loss', save_name = CNN_model_name + "_loss_history")
            show_train_history(history[:, 2], history[:, 3], monitor = 'Accuracy', save_name = CNN_model_name + "_acc_history")
            confusion_matrix, avg_val_loss, avg_val_acc = computeTestSetAccuracy(
                cnn, valid_data, valid_loader, criterion)
        else:
            cnn = CNN(grayscale)
            cnn.to(device)
            cnn.load_state_dict(torch.load(CNN_model_name + '.pt'))

            RF(cnn, mode = 'train')
            RF(cnn, mode = 'valid')

    if mode == 'test':
        cnn = CNN(grayscale)
        cnn.to(device)
        cnn.load_state_dict(torch.load(CNN_model_name + '.pt'))
        if forest:
            with open('rf_model.pt', 'rb') as f:
                clf = pickle.load(f)
        else:
            clf = None

        test_image_name  = os.listdir('./processed/C1-P1_Test')
        test_image_name.sort()
        prediction = predict(cnn, test_image_name, transform = image_transforms['valid'], save_name = 'CNN_rf', rf_model = clf)
