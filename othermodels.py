import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score as ps
from sklearn.metrics import recall_score as rs
from sklearn.metrics import f1_score as fs

import warnings
warnings.filterwarnings("ignore")


mnist = load_digits()
x,test_x,y,test_y = train_test_split(mnist.data,mnist.target,test_size=0.2,random_state=42)


modelsvm = svm.LinearSVC()
modelsvm.fit(x,y)
z=modelsvm.predict(test_x)
print('SVM_accuracy:',np.sum(z==test_y)/z.size)
print('SVM_precision:',ps(test_y,z,average="weighted"))
print('SVM_recall:',ps(test_y,z,average="weighted"))
print('SVM_f1:',ps(test_y,z,average="weighted"))

from sklearn.tree import DecisionTreeClassifier, export_graphviz
modeltree = DecisionTreeClassifier(criterion="entropy")
modeltree.fit(x,y)
z1=modeltree.predict(test_x)
print('DecisionTree_accuracy:',np.sum(z1==test_y)/z1.size)
print('DecisionTree_precision:',ps(test_y,z1,average="weighted"))
print('DecisionTree_recall:',ps(test_y,z1,average="weighted"))
print('DecisionTree_f1:',ps(test_y,z1,average="weighted"))

from sklearn.naive_bayes import MultinomialNB
modelNB = MultinomialNB()
modelNB.fit(x,y)
z2=modelNB.predict(test_x)
print('MultinomialNB_accuracy:',np.sum(z2==test_y)/z2.size)
print('MultinomialNB_precision:',ps(test_y,z2,average="weighted"))
print('MultinomialNB_recall:',ps(test_y,z2,average="weighted"))
print('MultinomialNB_f1:',ps(test_y,z2,average="weighted"))

# KNN
from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier(n_neighbors=3)
modelKNN.fit(x,y)
z3=modelKNN.predict(test_x)
print('KNN_acc:',np.sum(z3==test_y)/z3.size)
print('KNN_precision:',ps(test_y,z3,average="weighted"))
print('KNN_recall:',ps(test_y,z3,average="weighted"))
print('KNN_f1:',ps(test_y,z3,average="weighted"))



