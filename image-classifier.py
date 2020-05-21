# Imports here
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from time import time
from torch.autograd import Variable
from PIL import Image
import numpy as np

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

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
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
val_data = datasets.ImageFolder(data_dir + '/valid', transform=val_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# TODO: Build and train your network
model = models.densenet121(pretrained=True)
print("model \n", model)

# Freeze feature parameters (weights and biases)
for param in model.parameters():
    param.requires_grad = False

# Reconstruct the architecture acc. to our needs, input exp. 1024. output 102 classes
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 500)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))
# Set the models classifier to the new arch
model.classifier = classifier

# Specify loss fn, optimizer
#To record time
start_time = time()
#To choose between gpu and cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Use Negative log loss coz of using logsoftmax
criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

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

# Training code
epochs = 5
running_loss = 0
for epoch in range(epochs):
    train_loss = train(model, criterion, train_loader, optimizer)
    valid_loss, valid_accuracy = validation(model, val_loader, criterion)
    print("epoch:", epoch, "train loss:", train_loss, "validation loss:", valid_loss, "validation accuracy:",
          valid_accuracy)


end_time = time()

# TODO 0: Computes overall runtime in seconds & prints it in hh:mm:ss format
tot_time = end_time - start_time  # calculate difference between end time and start time
print("\n** Total Elapsed Runtime:",
      str(int((tot_time / 3600))) + ":" + str(int((tot_time % 3600) / 60)) + ":"
      + str(int((tot_time % 3600) % 60)))

test_loss, test_accuracy = validation(model, test_loader, criterion)
print("Test loss:", test_loss, "Test accuracy:", test_accuracy)


model.class_to_idx = train_data.class_to_idx
checkpoint = {'classifier': model.classifier,
              'epoch': epochs,
              'optimizer': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')


def load_the_checkpoint(path):
    load_dict_checkpt = torch.load(path)
    model = models.densenet121(pretrained=True)
    model.classifier = load_dict_checkpt['classifier']
    model.epochs = load_dict_checkpt['epoch']
    optimizer.load_state_dict(load_dict_checkpt['optimizer'])
    model.class_to_idx = load_dict_checkpt['class_to_idx']
    model.load_state_dict(load_dict_checkpt['state_dict'])
    for param in model.parameters():
            param.requires_grad = False
    # model.eval()
    return model, optimizer


model, optimizer = load_the_checkpoint('checkpoint.pth')


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #RESIZE:
    im_process = Image.open(image)
    print("size before resize", im_process.size)
    # new_width, new_height = 0, 0
    # aspect_ratio = im_process.size[0]/im_process.size[1]
    #Use ANTIALIAS to preserve aspect ratio
    if im_process.size[0] < im_process.size[1]:
        new_width = 256
        im_process.thumbnail((new_width, im_process.size[1]), Image.ANTIALIAS)
    else:
        new_height = 256
        im_process.thumbnail((im_process.size[0], new_height), Image.ANTIALIAS)
    print("size after resize", im_process.size)

    #CROP:
    #Since both w and h are > than 224
    left_pt = (im_process.size[0] - 224) / 2
    right_pt = left_pt + 224
    top_pt = (im_process.size[1] - 224) / 2
    bottom_pt = top_pt + 224

    im_process = im_process.crop((left_pt, top_pt, right_pt, bottom_pt))
    print("size after crop", im_process.size)

    #TO NP ARRAY:
    np_image = np.array(im_process)
    #normalize to 0-1
    np_image = np_image / 255
    print("image R G B", np_image[0][0])
    means = [0.485, 0.456, 0.406]
    standard_deviations = [0.229, 0.224, 0.225]
    np_image = np_image - means
    print("image R G B after subtraction", np_image[0][0])
    np_image = np_image / standard_deviations
    print("image R G B after division", np_image[0][0])
    print("Before transpose", np_image.shape)
    np_image = np.transpose(np_image, (2, 0, 1))
    print("After transpose", np_image.shape)
    print("Type of array", type(np_image))
    return np_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    # image = image.numpy().transpose((1, 2, 0)) It is already numpy
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    # plt.show()

    return ax


image_path = data_dir + '/test/1/image_06764.jpg'
imshow(process_image(image_path))

def predict(image_path, model, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    image_input = process_image(image_path)
    # image_input = torch.from_numpy(image_input).float().to(device)
    image_input = torch.from_numpy(image_input).float()
    #add one dimension for batch as tensor expects batch number
    image_input = image_input.unsqueeze(0)
    print("predict one image shape is {}".format(image_input.shape))

    # model.to(device)
    model.eval()
    output = model(image_input)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(top_k)
    # Detach from tensor and convert to numpy
    top_p = top_p.detach().numpy().tolist()[0]
    top_class = top_class.detach().numpy().tolist()[0]

    #model.class_to_idx has mapping from class->idx. So reverse it to idx->class/cat
    idx_to_class = {}
    for key, value in model.class_to_idx:
        idx_to_class[value] = key
    #Now iterate idx->class to get top 5 classes
    top_c = []
    for clas in top_class:
        top_c.append(idx_to_class[clas])

    return top_p, top_c



imagepred_path = data_dir + '/test/1/image_06764.jpg'

predict(imagepred_path, model)

def sanity_check(image_path, model):
    fig, ax = plt.subplots()
    image = process_image(image_path)
    ax.imshow(image)

    fig, ax = plt.subplots()
    pil_im = Image.open(image_path)
    ax.set_title('ax1 title')
    ax.imshow(pil_im)

    plt.subplot(234)
    objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
    y_pos = np.arange(len(objects))
    performance = [10, 8, 6, 4, 2, 1]

    plt.barh(y_pos, performance, color='royalblue', align='center', alpha=0.5)
    plt.yticks(y_pos, objects)
    plt.xlabel('Usage')
    plt.title('Programming language usage')

    plt.show()





