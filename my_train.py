import argparse
import network
from dataloader import MNIST_load
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import time
import torch.optim as optim
from sklearn.metrics import precision_score as ps
from sklearn.metrics import recall_score as rs
from sklearn.metrics import f1_score as fs
# import sys
# import visdom
from torch.utils.tensorboard import SummaryWriter   

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--lr',default='0.001',type=str)
parser.add_argument('--op',default='Adam',type=str)
parser.add_argument('--net',default='MNlexNet',type=str)
args = parser.parse_args()

if args.net == 'AlexNet':
    writer = SummaryWriter('./log/AlexNet')

if args.net == 'MNlexNet' and args.op == 'Adam' and args.lr == '0.001':
    writer = SummaryWriter('./log/Adam0.001')
elif args.net == 'MNlexNet' and args.op == 'SGD' and args.lr == '0.001':
    writer = SummaryWriter('./log/SGD0.001')
elif args.net == 'MNlexNet' and args.op == 'SGD' and args.lr == '0.05':
    writer = SummaryWriter('./log/SGD0.05')
elif args.net == 'MNlexNet' and args.op == 'Adagrade' and args.lr == '0.001':
    writer = SummaryWriter('./log/Adagrade0.001')
elif args.net == 'MNlexNet' and args.op == 'RMSprop' and args.lr == '0.005':
    writer = SummaryWriter('./log/RMSprop0.005')
elif args.net == 'MNlexNet' and args.op == 'AdamW' and args.lr == '0.05':
    writer = SummaryWriter('./log/Adam0.05')
elif args.net == 'MNlexNet' and args.op == 'Adam' and args.lr == '0.005':
    writer = SummaryWriter('./log/Adam0.005')
elif args.net == 'MNlexNet' and args.op == 'AdamW' and args.lr == '0.05':
    writer = SummaryWriter('./log/AdamW0.05')
elif args.net == 'MNlexNet' and args.op == 'SGDW' and args.lr == '0.05':
    writer = SummaryWriter('./log/1000SGDW0.05')



n_epochs = 3
learning_rate = 0.01
momentum = 0.5
log_interval = 10

train_loader, test_loader = MNIST_load()



def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x


def train(train_loader, model, criterion, optimizer):
    model.train()
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []

    for i, data in enumerate(train_loader, 0):

        inputs, labels = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)

        _, pred = outputs.max(1)

        num_correct = (pred == labels).sum().item()
        acc = num_correct/len(labels)
        pre = ps(labels.cpu().numpy(),pred.cpu().numpy(),average='weighted')
        rec = rs(labels.cpu().numpy(),pred.cpu().numpy(),average='weighted')
        f1 = fs(labels.cpu().numpy(),pred.cpu().numpy(),average='weighted')

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_acc.append(acc)
        train_precision.append(pre)
        train_recall.append(rec)
        train_f1.append(f1)

    return np.mean(train_loss), np.mean(train_acc), np.mean(train_precision), np.mean(train_recall), np.mean(train_f1)


def test(test_loader, model):
    model.eval()
    test_loss = []
    test_acc = []
    test_precision = []
    test_recall = []
    test_f1 = []

    for i, data in enumerate(test_loader, 0):

        inputs, labels = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, pred = outputs.max(1)

        num_correct = (pred == labels).sum().item()
        acc = num_correct/len(labels)
        pre = ps(labels.cpu().numpy(),pred.cpu().numpy(),average='weighted')
        rec = rs(labels.cpu().numpy(),pred.cpu().numpy(),average='weighted')
        f1 = fs(labels.cpu().numpy(),pred.cpu().numpy(),average='weighted')

        test_loss.append(loss.item())
        test_acc.append(acc)
        test_precision.append(pre)
        test_recall.append(rec)
        test_f1.append(f1)

    return np.mean(test_loss), np.mean(test_acc), np.mean(test_precision), np.mean(test_recall), np.mean(test_f1)

if args.net == 'MNlexNet':
    model = network.MNlexNet()
elif args.net == 'AlexNet':
    model = network.AlexNet()
criterion = nn.CrossEntropyLoss()

if args.lr == '0.001':
    learning_rate = 0.001
elif args.lr == '0.05':
    learning_rate = 0.05
elif args.lr == '0.01':
    learning_rate = 0.01
elif args.lr == '0.005':
    learning_rate = 0.005
elif args.lr == '1e-5':
    learning_rate = 1e-5


if args.op == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
elif args.op == 'SGDW':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005)
elif args.op == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # best
elif args.op == 'Adagrade':
    optimizer = optim.Adagrade(model.parameters(), lr=learning_rate)
elif args.op == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
elif args.op == 'AdamW':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.005)
elif args.op == 'AdamDelta':
    optimizer = optim.AdamDelta(model.parameters(), lr=learning_rate)

best_loss = 3
if torch.cuda.is_available():
    model = model.cuda()

for epoch in range(0, 100):
    time_all = 0
    start_time = time.time()
    train_loss, train_acc, train_precision, train_recall,train_f1 = train(train_loader, model, criterion, optimizer)
    time_all = time.time()-start_time
    test_loss, test_acc, test_precision, test_recall, test_f1 = test(test_loader, model)
    if (epoch+1) % 10 == 0:
        print('Epoch: %d,  Train_loss: %.5f, Train_acc: %.5f  Val_loss: %.5f, T_Time: %.3f, Val Acc: %5f' % (
        epoch+1, train_loss, train_acc, test_loss, time_all, test_acc))
    writer.add_scalars('Accuracy',{'Train_acc':train_acc,'Val_acc':test_acc},epoch)
    writer.add_scalar('Train_loss',train_loss,epoch)
    writer.add_scalar('Val_loss',test_loss,epoch)
    writer.add_scalars('Precision',{'train_precision':train_precision,'Val_precision':test_precision},epoch)
    writer.add_scalars('Recall',{'train_recall':train_recall,'Val_recall':test_recall},epoch)
    writer.add_scalars('F1',{'train_f1':train_f1,'Val_f1':test_f1},epoch)

    # save best model
    if test_loss < best_loss:
        best_loss = test_loss
        # print('Find better model in Epoch {0}, saving model.'.format(epoch))
        torch.save(model.state_dict(), './model/MNIST_model.pt')


# for id, data in enumerate(train_loader, 0):
#     inputs, labels = data
#     print(inputs.size(0))





