import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


num_classes = 20
n_test = 20

# Training settings
parser = argparse.ArgumentParser(description='Final Project')
parser.add_argument('--data', type=str, default='PascalSentenceDataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default:)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)



# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from data import data_transforms, data_transforms_aug


train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/dataset',
                         transform=data_transforms_aug),
    batch_size=args.batch_size, shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/validation_images',
                         transform=data_transforms),
    batch_size=n_test, shuffle=False, num_workers=1)

# Neural network and optimizer

# model = models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, num_classes)

model = models.alexnet(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Linear(4096, num_classes)
print(model)

if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': 0.001},
        {'params': model.classifier[1].parameters(), 'lr': 0.002},
        {'params': model.classifier[4].parameters(), 'lr': 0.002},
        {'params': model.classifier[6].parameters(), 'lr': 0.01}
], lr=10**-2, momentum=0.9, weight_decay = 0.0005)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

def output_space():
    model.eval()
    outputspace = np.zeros((num_classes,n_test,num_classes))
    j=0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = F.softmax(model(data), dim=1).detach().numpy()
        outputspace[j] = output
        j += 1
    return outputspace


initial_lr = [0.001,0.002,0.002,0.01]

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    optimizer.param_groups[0]['lr'] = initial_lr[0]*(0.1**(epoch // 5))
    optimizer.param_groups[1]['lr'] = initial_lr[1]*(0.1**(epoch // 5))
    optimizer.param_groups[2]['lr'] = initial_lr[2]*(0.1**(epoch // 5))
    optimizer.param_groups[3]['lr'] = initial_lr[3]*(0.1**(epoch // 5))

for epoch in range(1, args.epochs + 1):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    validation()

model_file = args.experiment + '/model_' + str(epoch) + '.pth'
torch.save(model.state_dict(), model_file)

outputspace = output_space()
np.save('experiment/cnnfeat',outputspace)    # to load : np.load('experiment/cnnfeat.npy')
