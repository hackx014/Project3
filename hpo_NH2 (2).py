#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
'''import argparse
import json
import logging
import os
import sys


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms, models'''

import torch
import argparse
import os
import json
import logging
import sys

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from torch import nn, optim
from PIL import ImageFile

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
     
'''data = "s3://project3-image-data/image-data/"
SM_CHANNEL_TRAINING = "s3://project3-image-data/image-data/"
SM_OUTPUT_DATA_DIR= "s3://project3-image-data/image-data/output/"
SM_MODEL_DIR= "s3://project3-image-data/image-data/model/"'''


def train(model, train_loader, loss_criterion, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = loss_criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    return model


def test(model, test_loader, loss_criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    #with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_criterion(output, target)
        percentage, preds = torch.max(output, 1)
        test_loss += loss.item() * data.size(0) # sum up batch loss
            #pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += torch.sum(preds == target)
    avg_loss = test_loss / len(test_loader.dataset)
    # Calculation of accuracy:
    acc = correct / len(test_loader.dataset)

    #test_loss /= len(test_loader.dataset)
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            avg_loss, correct, len(test_loader.dataset), 100.0 * acc
        )
    )
    pass
    
def net():
    model = models.resnet50(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential( nn.Linear( num_features, 256), 
                             nn.ReLU(inplace = True),
                             nn.Linear(256, 133),
                             nn.ReLU(inplace = True) 
                            )
    return model
    pass

def create_data_loaders(train_dir, test_dir, eval_dir, batch_size):
    '''Train = os.path.join(data,"train")
    Test = os.path.join(data,"test")
    Vald = os.path.join(data,"valid")
    print(data)
    print(Train)
    print(Test)'''
    training_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(), # randomly flip and rotate
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    valid_transform : transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    testing_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_set = ImageFolder(root = train_dir, transform = training_transform)
    test_set = ImageFolder(root = test_dir, transform = testing_transform)
    #valid_set = ImageFolder(root = eval_dir, transform = valid_transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    #valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    
    
    return train_loader, test_loader #valid_loader, test_loader
    pass

def main(args):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    model = model.to(device)
    #with open(os.path.join(model_dir, "model.pth"), "rb") as f:
    #    model.load_state_dict(torch.load(f))
    #return model
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    #optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 
    #loss_criterion = F.nll_loss
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    
    test_loader, train_loader = create_data_loaders(args.train_dir, args.test_dir, args.eval_dir, args.batch_size)
    train(model, train_loader, loss_criterion, optimizer, args.epochs, device)
    #model=
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device)
    print("Saving the model.")
    #path = os.path.join(data, "model.pth")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
    
    '''
    TODO: Save the trained model
    '''
    #torch.save(model, path)

if __name__=='__main__':
    '''SM_CHANNEL_TRAINING = "s3://project3-image-data/image-data/"
    SM_OUTPUT_DATA_DIR= "s3://project3-image-data/image-data/output/"
    SM_MODEL_DIR= "s3://project3-image-data/image-data/model/"'''
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test-dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--eval-dir", type=str, default=os.environ["SM_CHANNEL_VAL"])
    #parser.add_argument("--model_dir", type=str, default=SM_MODEL_DIR)
    #parser.add_argument("--data", type=str, default=SM_CHANNEL_TRAINING)
    #parser.add_argument("--output_dir", type=str, default=SM_OUTPUT_DATA_DIR)
    
    args = parser.parse_args()
    main(args)

    #Train(parser.parse_args('train_loader', 'criterion', 'optimizer'))

