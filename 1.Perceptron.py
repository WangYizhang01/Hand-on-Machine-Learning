# perceptron.py
#
# Created by 王一张 on 2021/3/22.

import numpy as np

def load_data(filename):
    """导入数据"""
    train_data = []; train_labels = []
    f = open(filename,'r')
    for line in f.readlines():
        data = line.strip().split(',')
        train_data.append([int(i)/255 for i in data[1:]])
        train_labels.append(1 if int(data[0]) >= 5 else -1)

    return train_data,train_labels

def perceptron(train_data,train_labels,iters=50,n=0.0001):
    """感知机训练"""
    train_data = np.mat(train_data)
    train_labels = np.mat(train_labels).T
    w = np.zeros((1,np.shape(train_data)[1]))
    b = 0

    for i in range(iters):
        print("epoch: ",i)
        for j in range(len(train_data)):
            if train_labels[j] * (np.dot(w,train_data[j].T)+b) < 0:
                w = w + n*train_labels[j]*train_data[j]
                b = b + n*train_labels[j]
    return w,b

def accuracy(filename,w,b):
    """返回准确度"""
    test_data,test_labels = load_data(filename)
    test_data = np.mat(test_data)
    test_labels = np.mat(test_labels).T
    count = 0

    for j in range(len(test_data)):
        if test_labels[j] * (np.dot(w, test_data[j].T) + b) < 0:
            count += 1
    return 1-count/len(test_labels)

if __name__ == '__main__':
    train_data,train_labels = load_data('/Users/wangyizhang/PycharmProjects/Statistical-Learning-Method_Code/Mnist/mnist_train.csv')
    w, b = perceptron(train_data,train_labels,30)

    print("accuracy is : ")
    print(accuracy('/Users/wangyizhang/PycharmProjects/Statistical-Learning-Method_Code/Mnist/mnist_test.csv',w,b))
