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



def train(model, epochs, train_loader, valid_loader, loss_criterion, optimizer, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
    '''

    loader = {'train': train_loader, 'eval': valid_loader}
    epochs = epochs
    best_loss = 1e6
    loss_counter = 0

    for epoch in range(epochs):  # Iteration over all epochs
        for phase in ['train', 'eval']:
            print(f'Epoch: {epoch}, Phase: {phase}')
            # Sets model to train or evaluation modes
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # Initialization of varialbes to hold model outcomes
            total_loss = 0
            total_correct = 0
            sample_count = 0  

            for train_input, labels in loader[phase]:  # Iteration over data in train loader


                train_input = train_input.to(device)
                labels = labels.to(device)

                output = model(train_input)
                # Getting loss value
                loss = loss_criterion(output, labels)

                # Train phase 
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward() 
                    optimizer.step()

                percentages, preds = torch.max(output, dim=1)
                total_loss += loss.item() * train_input.size(0)
                total_correct += torch.sum(preds == labels).item()
                sample_count += len(train_input)

                # Log print after 3000 samples are run
                if sample_count % 3000 == 0:
                    accuracy = total_correct / sample_count
                    print(f'Log Entry: Epoch {epoch}, Phase {phase}.')
                    print(
                        'Images [{}/{} ({:.0f}%)] / Loss: {:.2f} / Accumulated Loss: {:.0f} / '
                        'Accuracy: {}/{} ({:.2f}%)'.format(
                            sample_count,
                            len(loader[phase].dataset),
                            100 * (sample_count / len(loader[phase].dataset)),
                            loss.item(),
                            total_loss,
                            total_corrects,
                            sample_count,
                            100 * accuracy
                        ))

                # stops training after the count of samples trained is equal to 25% of the total samples in the training file.
                if sample_count >= (0.25 * len(loader[phase].dataset)):
                    break
            # Evaluation phase
            if phase == 'eval':
                avg_epoch_loss = total_loss / sample_count
                if avg_epoch_loss < best_loss:  # If the avg_loss increased in the epoch, signalize it.
                    best_loss = avg_epoch_loss  # Update of maximum loss so far
                else:
                    loss_counter += 1  # update loss counter which will end the phase as the loss counter will be greater than 0.

        if loss_counter > 0:
            break


def test(model, test_loader, loss_criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_criterion(output, target)
        percentage, preds = torch.max(output, 1)
        test_loss += loss.item() * data.size(0) # sum up batch loss
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

    
def net():
    model = models.resnet50(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential( nn.Linear( num_features, 128), 
                             nn.ReLU(inplace = True),
                             nn.Linear(128, 133)
                            )
    return model


def create_data_loaders(train_dir, test_dir, eval_dir, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    https://github.com/aws-samples/amazon-sagemaker-debug-ml-model-training/blob/main/debug_ml_model_training.ipynb transform code modeled after github
    '''

    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    testing_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = ImageFolder(root=train_dir, transform=training_transform)
    test_data = ImageFolder(root=test_dir, transform=testing_transform)
    valid_data = ImageFolder(root=eval_dir, transform=testing_transform)

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)
    valid_loader = DataLoader(valid_data, batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def main(args):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    model = model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.fc.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, valid_loader, test_loader = create_data_loaders(args.train_dir, args.test_dir, args.eval_dir,
                                                                       args.batch_size)

    train(model, args.epochs, train_loader, valid_loader, loss_criterion, optimizer, device)
    
    '''    
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device)
    print("Saving the model.")
    #path = "./Dog_Image_Pred_Model"
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
    
    '''
    TODO: Save the trained model
    '''
    #torch.save(model, path)

if __name__=='__main__':
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
    
    args = parser.parse_args()
    main(args)


