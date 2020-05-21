#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# PROGRAMMER: Janani
# Imports here
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch.autograd import Variable
import args

def main():
    in_arg = args.get_input_args_train()
    print("Arguments: ", in_arg)

    train_dir = in_arg.data_dir + '/train'
    valid_dir = in_arg.data_dir + '/valid'
    test_dir = in_arg.data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    val_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform=val_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    if in_arg.arch == "densenet121":
        model = models.densenet121(pretrained=True)
    elif in_arg.arch == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        print("Please select between densenet121 and vgg16 for the architecture")
    
    for param in model.parameters():
            param.requires_grad = False

    if in_arg.arch == "densenet121":
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, in_arg.hidden)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(in_arg.hidden, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    elif in_arg.arch == "vgg16":
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 4096)),
            ('relu', nn.ReLU()),
            ('relu', nn.Dropout(p=0.4)),
            ('fc2', nn.Linear(4096, in_arg.hidden)),
            ('relu', nn.ReLU()),
            ('relu', nn.Dropout(p=0.4)),
            ('fc3', nn.Linear(in_arg.hidden, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        

    # Set the models classifier to the new arch
    model.classifier = classifier
    
    print("model \n", model)

    if in_arg.gpu == 'gpu':
        device = 'cuda'
    else:
        device = 'cpu'

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Use Negative log loss coz of using logsoftmax
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)

    model.to(device)


    # Function for validation & testing
    def validation(model, val_loader, criterion):
        model.to(device)
        model.eval()
        correct = 0.0
        loss_all = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                target = Variable(target)
                loss = criterion(output, target)
                pred = output.max(dim=1)[1]
                correct += pred.eq(target).sum().item()
                loss_all += loss.item()


        return loss_all / len(val_loader.dataset), float(correct) / len(val_loader.dataset)


    def train(model, criterion, train_loader, optimizer):
        model.to(device)
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # print("data", data.shape)
            output = model(data)
            target = Variable(target)
            # print("train target", target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader.dataset)
        return train_loss

    epochs = in_arg.epoch
    for epoch in range(epochs):
        train_loss = train(model, criterion, train_loader, optimizer)
        valid_loss, valid_accuracy = validation(model, val_loader, criterion)
        print("epoch:", epoch, "train loss:", train_loss, "validation loss:", valid_loss, "validation accuracy:",
              valid_accuracy)



#     test_loss, test_accuracy = validation(model, test_loader, criterion)
#     print("Test loss:", test_loss, "Test accuracy:", test_accuracy)


    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'classifier': model.classifier,
                  'arch': in_arg.arch,
                  'epoch': epochs,
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, in_arg.save_dir)


# Call to main function to run the program
if __name__ == "__main__":
    main()

