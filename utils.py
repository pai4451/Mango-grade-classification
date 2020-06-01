import os
import time
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
from torchvision import transforms, models


class MyRotationTransform:
    """Rotate by the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        return TF.rotate(x, self.angles)

class MangoDataset(Dataset):
    """Mango dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.class_to_idx = {'A':0,'B':1,'C':2}
        self.mango_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.mango_frame)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.mango_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.mango_frame.iloc[idx, 1]
        label = self.class_to_idx[label]

        if self.transform:
            image = self.transform(image)

        return image, label



class EarlyStopping:
    """
    from pytorchtools 
    https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, save_name = 'checkpoint'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_name = save_name
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_name + '.pt')
        self.val_loss_min = val_loss


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14, save_name = 'confusion_matrix'):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sn.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=90, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_name + ".jpg", format="jpg")
    #plt.show()

def show_train_history(train_history, val_history, monitor = 'Loss',save_name = 'train_history'):
    fig = plt.figure()
    plt.plot(train_history)
    plt.plot(val_history)
    plt.title('Train History')
    plt.ylabel(monitor)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper right')
    plt.tight_layout()
    plt.savefig(save_name + ".jpg", format="jpg")
    #plt.show()