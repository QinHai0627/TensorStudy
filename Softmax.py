# -*- coding: utf-8 -*-


import numpy as np  
import matplotlib.pyplot as plt  
#import random

#载入数据：斯坦福CS231n课程之螺旋数据集
N = 100 # number of points per class  
D = 2 # dimensionality  
K = 3 # number of classes  
X = np.zeros((N * K, D)) # data matrix (each row = single example)  
y = np.zeros(N * K, dtype='uint8') # class labels  
for j in range(K):  
    ix = list(range(N*j, N*(j + 1)))  
    r = np.linspace(0.0, 1, N) # radius  
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 # theta  
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]  
    y[ix] = j
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.gist_rainbow)  
plt.show() 

# 初始化线性分类器y=wx+b的参数：随机赋值  
W = 0.01 * np.random.randn(D,K)  
b = np.zeros((1,K)) 

# 线性分类器，用矩阵乘法得到各个类别的分数
scores = np.dot(X, W) + b #300*3矩阵，每行代表一个数据点，每列代表三个属性的一个

#交叉熵损失
num_examples = X.shape[0]  
# get unnormalized probabilities  
exp_scores = np.exp(scores)  
# normalize them for each example  
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  

#-log-log映射
corect_logprobs = -np.log(probs[range(num_examples),y])
reg=1.5 #初始任意赋值

# compute the loss: average cross-entropy loss and regularization  
data_loss = np.sum(corect_logprobs)/num_examples  
reg_loss = 0.5 * reg * np.sum(W * W)  
loss = data_loss + reg_loss  

# 梯度下降
dscores = probs  
dscores[range(num_examples),y] -= 1  
dscores /= num_examples

dW = np.dot(X.T, dscores)  
db = np.sum(dscores, axis=0, keepdims=True)  
dW += reg*W # don't forget the regularization gradient

#Train a Linear Classifier  
  
# initialize parameters randomly  
W = 0.01 * np.random.randn(D,K)  
b = np.zeros((1,K))  
  
# some hyperparameters  
step_size = 1e-0  
reg = 1e-3 # regularization strength  
  
# gradient descent loop  
num_examples = X.shape[0]  
for i in range(200):  
    scores = np.dot(X, W) + b  
# evaluate class scores, [N x K]  


# compute the class probabilities  
exp_scores = np.exp(scores)  
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]  
  
# compute the loss: average cross-entropy loss and regularization  
corect_logprobs = -np.log(probs[range(num_examples),y])  
data_loss = np.sum(corect_logprobs)/num_examples  
reg_loss = 0.5 * reg * np.sum(W * W)  
loss = data_loss + reg_loss  
if i % 10 == 0:  
    print ("iteration %d: loss %f" % (i, loss))  
  
# compute the gradient on scores  
dscores = probs  
dscores[range(num_examples),y] -= 1  
dscores /= num_examples  
  
# backpropate the gradient to the parameters (W,b)  
dW = np.dot(X.T, dscores)  
db = np.sum(dscores, axis=0, keepdims=True)  
  
dW += reg*W # regularization gradient  
  
# perform a parameter update  
W += -step_size * dW  
b += -step_size * db  

# evaluate training set accuracy  
scores = np.dot(X, W) + b  
predicted_class = np.argmax(scores, axis=1)  
print('training accuracy: %.2f' % (np.mean(predicted_class == y))) 