from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

LDA_train = np.load('experiment/lda_train.npy')
LDA_validation = np.load('experiment/lda_val.npy')

#TextNet settings
n_epochs = 80
batch_size = 32
lr = 0.001

# Data parameters
n_cat = 20
n_train = 30
n_val = 20

# Data initialization and loading
tensor_y = torch.zeros(n_cat * n_train, dtype=torch.long)
for i in range(n_train * n_cat):
    tensor_y[i] = int(np.floor(i/n_train))

tensor_x = torch.stack([torch.Tensor(i) for i in LDA_train])

my_dataset = TensorDataset(tensor_x,tensor_y)
train_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True, num_workers=1)



tensor_y_val = torch.zeros(n_cat * n_val, dtype=torch.long)
for i in range(n_val * n_cat):
    tensor_y_val[i] = int(np.floor(i/n_val))

tensor_x = torch.stack([torch.Tensor(i) for i in LDA_validation])

my_dataset_val = TensorDataset(tensor_x,tensor_y_val)
val_loader = DataLoader(my_dataset_val, batch_size=n_val, shuffle=False, num_workers=1)



use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')


# Neural network and optimizer

class TextNet(nn.Module):
    def __init__(self):
        super(TextNet, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, n_cat)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = TextNet()

# optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

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
        if batch_idx==0:
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

l_mult = 0.9
for epoch in range(1, n_epochs + 1):
    if epoch % 4 == 0:
        optimizer.param_groups[0]['lr'] = l_mult*optimizer.param_groups[0]['lr']
    train(epoch)
    validation()


def output_space():
    model.eval()
    outputspace = np.zeros((n_cat,n_val,n_cat))
    j = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = F.softmax(model(data), dim=1).detach().numpy()
        outputspace[j] = output
        j += 1
    return outputspace


outputspace = output_space()
np.save('experiment/text_feat',outputspace)
