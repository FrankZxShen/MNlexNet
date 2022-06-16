# MNIST-AlexNet(MNlexNet-pytorch)

This is the PyTorch version repository for MNIST dataset identification.

## Prerequisites

This project uses Anaconda as the environment. The code was tested on Ubuntu 18.04, with Python 3.6, CUDA 10.2 and PyTorch v1.6.0. NVIDIA GPUs are needed for both training and testing.

Install PyTorch==1.6 and torchvision==0.7:

```
conda create -n MNlexNet python=3.6 pytorch=1.6 torchvision -c pytorch
```

Install package dependencies:

```
pip install -r requirements.txt
```

## Datasets

During the training process, the MNIST dataset will be downloaded automatically. If you need to download it, please put the [file](http://yann.lecun.com/exdb/mnist/) in the data folder.

## Training Process

If only using the best model, run:

```
python my_train.py
```

If using SGD or another optimizer, run:

```
python my_train.py --op SGD
```

If using SGD with a learning rate of 0.005, run:

```
python my_train.py --op SGD --lr 0.005
```

For AlexNet, run:

```
python my_train.py --net AlexNet
```

For traditional machine learning methods, run:

```
python othermodels.py
```

You can see the accuracy, precision, recall and f1 scores of SVM, Decision Tree (ID3), Polynomial Bayes and KNN compared.

The best trained models are in the model folder.

## Testing Process

If MNIST dataset images are randomly selected for validation, run:

```
python my_test.py
```

## Visualization

Open log file:

```
cd log
```

Start tensorboard:

```
tensorboard --logdir yourdir --port 6006
```

You can enter `localhost:6006` on your browser to see the visualization.