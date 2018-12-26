import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='Final Project')
parser.add_argument('--data', type=str, default='PascalSentenceDataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                    help='learning rate (default:)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
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
from data import data_transforms


# train_idx, valid_idx = indices[split:], indices[:split]
# sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/dataset',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/validation_images',
                         transform=data_transforms),
    batch_size=20, shuffle=False, num_workers=1)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script


num_classes = 20

#model = models.squeezenet1_1(pretrained=True)
model = models.alexnet(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
# Replace the model classifier
#model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

# model.fc = nn.Sequential(nn.Linear(num_ftrs, 200),nn.ReLU(True), nn.Linear(200, num_classes))
softmax = nn.Softmax(dim=1)
model.classifier[6] = nn.Sequential(nn.Linear(4096, num_classes), nn.Softmax(dim=1))
model.classifier[6] = nn.Linear(4096, num_classes)

print(model)



if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

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
    outputspace = np.zeros((20,20,20))
    j=0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = F.softmax(model(data), dim=1).detach().numpy()
        outputspace[j] = output
        j += 1
    return outputspace


for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    #model_file = args.experiment + '/model_' + str(epoch) + '.pth'
    #torch.save(model.state_dict(), model_file)
    #print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file')

outputspace = output_space()
np.save('experiment/cnnfeat',outputspace)
# to load : np.load('experiment/cnnfeat.npy')

# everything is in the natural order
