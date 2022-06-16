import torch
import torch.nn as nn
import network
from dataloader import MNIST_load
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score as ps
from sklearn.metrics import recall_score as rs
from sklearn.metrics import f1_score as fs

model = network.MNlexNet()
model.load_state_dict(torch.load('./model/MNIST_model.pt'))

train_loader, test_loader = MNIST_load()

criterion = nn.CrossEntropyLoss()
def test(test_loader, model):
    model.eval()
    test_loss = []
    test_acc = []
    test_precision = []
    test_recall = []
    test_f1 = []

    examples = enumerate(test_loader)
    batch_idx, (inputs, labels) = next(examples)

    # if torch.cuda.is_available():
    #     inputs = inputs.cuda()
    #     labels = labels.cuda()

    with torch.no_grad():
        outputs = model(inputs)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(inputs[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            outputs.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
    plt.yticks([])
    plt.savefig('./img/test.png')


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

if __name__ == '__main__':
    test_loss,test_acc,test_precision,test_recall,test_f1=test(test_loader,model)
    print("Accuracy:",test_acc)
    print("Precision:",test_precision)
    print("Recall:",test_recall)
    print("f1:",test_f1)
    print("loss",test_loss)

