import numpy as np
import time

def load_data(filename):
    data_train = []
    data_labels = []
    with open(filename,'r') as f:
        for line in f.readlines():
            data = line.strip().split(',')
            data_labels.append(int(data[0]))
            data_train.append([int(i) for i in data[1:]])

    return data_train,data_labels

def compute_dist(l1,l2):
    return np.sqrt(np.sum(np.square(l1-l2)))

def KNN(data_train,data_labels,topK,x):
    data_train = np.mat(data_train)
    # data_labels = np.mat(data_labels)

    dist = []
    for i in range(len(data_train)):
        dist.append(compute_dist(data_train[i],x))

    d_temple = sorted(dist)[:topK]

    labels = []
    for item in d_temple:
        labels.append(data_labels[dist.index(item)])

    l = -1;max = -1
    for j in set(labels):
        if labels.count(j) > max:
            l = j

    return l

def accuracy(data_train,data_labels,data_test,test_labels):
    data_train = np.mat(data_train)
    # data_labels = np.mat(data_labels).T
    data_test = np.mat(data_test)
    # test_labels = np.mat(test_labels).T

    # # 运行一遍KNN
    # for i in range(len(data_train)):
    #     if i%500 == 0:
    #         print("train "+ str(i) + "samples.")
    #     data_labels[i] = KNN(data_train,data_labels,25,data_train[i])

    err_count = 0
    for j in range(len(data_test)):
        if j%5 == 0:
            print("test "+ str(j) + "samples.")
        label = KNN(data_train,data_labels,25,data_test[j])
        if label != test_labels[j]:
            err_count += 1

    print("the accuracy is : ")
    print(1-err_count/200)


if __name__ == "__main__":
    start = time.time()
    data_train, data_labels = load_data('/Users/wangyizhang/PycharmProjects/Statistical-Learning-Method_Code/Mnist/mnist_train.csv')
    data_test, test_labels = load_data('/Users/wangyizhang/PycharmProjects/Statistical-Learning-Method_Code/Mnist/mnist_test.csv')
    data_test = data_test[:200]
    test_labels = test_labels[:200]
    accuracy(data_train,data_labels,data_test,test_labels)
    print("time is : " + str(time.time()-start))
