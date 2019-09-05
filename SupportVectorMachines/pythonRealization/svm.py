import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import math

# https://ljvmiranda921.github.io/notebook/2017/02/11/multiclass-svm/
# https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
# https://www.pyimagesearch.com/2016/09/05/multi-class-svm-loss/


class LogisticRegression:
    def __init__(self, file_name):
        self.file_name = file_name

    def get_dataset(self, train_size = 0.9):
        df = pd.read_csv(self.file_name)
        df = df.drop(['Id'],axis=1)
        target = df['Species']
        s = set()
        for val in target:
            s.add(val)
        s = list(s)
        rows = list(range(100,150))
        df = df.drop(df.index[rows])

        x = df['SepalLengthCm']
        y = df['PetalLengthCm']

        self.setosa_x = x[:50]
        self.setosa_y = y[:50]

        self.versicolor_x = x[50:]
        self.versicolor_y = y[50:]


        ## Drop rest of the features and extract the target values
        df = df.drop(['SepalWidthCm','PetalWidthCm'],axis=1)
        Y = []
        target = df['Species']
        for val in target:
            if(val == 'Iris-setosa'):
                Y.append(0)
            else:
                Y.append(1)
        df = df.drop(['Species'],axis=1)
        X = df.values.tolist()

        ## Shuffle and split the data into training and test set
        X, Y = shuffle(X,Y)
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=train_size)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        self.x_train = x_train.reshape(90,2)
        self.x_test = x_test.reshape(10,2)
        self.y_train = y_train.reshape(90,1)
        self.y_test = y_test.reshape(10,1)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def _countCostFunc(self, lambda_= 0.001):

        y = self._sigmoid(self.x_train.dot(self.W))

        self.m = len(self.y_train)
        j = -(1 / self.m) * (np.asscalar(np.transpose(self.y_train).dot(np.log(y)))
          + np.asscalar(np.transpose((1 - self.y_train)).dot(np.log(1 - y)))) + lambda_ * np.sum(self.W ** 2) / (2 * self.m)

        return j
        

    def train(self, epochs = 500, learning_rate = 0.001, lambda_ = 0.001):
        self.m = len(self.y_train)
        w_size = (self.x_train[0].shape[0], 1)
        self.W = np.random.randn(w_size[0], w_size[1])*0.01
        epochs_count = 1
        cost_x = []
        cost_y = []
        while(epochs_count < epochs):
            y = self._sigmoid(self.x_train.dot(self.W))

            pen = np.transpose(self.x_train).dot(y - self.y_train) + lambda_ * self.W / self.m

            self.W = self.W - learning_rate * pen
            epochs_count += 1
            cost_x.append(epochs_count)
            cost = self._countCostFunc(lambda_)
            cost_y.append(cost)
        plt.figure(figsize=(8,6))
        plt.plot(cost_x, cost_y, 'k-')
        plt.title("cost for logistic regression")
        plt.xlabel('epochs')
        plt.ylabel('cost')
        plt.show()
        return self.W

    def predict(self):
        y_pred = self._sigmoid(self.x_test.dot(self.W))
        predictions = []
        for val in y_pred:
            if(val > 0.5):
                predictions.append(1)
            else:
                predictions.append(0)
        print(len(predictions), ' ', len(self.y_test))
        accuracy_score_ans = accuracy_score(self.y_test,predictions)

        return accuracy_score_ans
    
    def plotDecisionBoundary(self):
        plt.figure(figsize=(8,6))
        plt.scatter(self.setosa_x,self.setosa_y,marker='+',color='green')
        plt.scatter(self.versicolor_x,self.versicolor_y,marker='_',color='red')

        number_of_points = len(self.W)
        a = -self.W[0] / self.W[1]

        xx = np.linspace(4, 7, num = number_of_points)
        xx = xx.reshape(number_of_points,1)

        yy = a * xx
        plt.plot(xx, yy, 'k-')
        plt.title("decision boundary for logistic regression")
        plt.xlabel('x1')
        plt.ylabel('x2')
        return plt


class Svm:
    def __init__(self, file_name):
        self.file_name = file_name

    # binary_svm
    def getDataset(self, train_size = 0.9, flagMulti = False):
        df = pd.read_csv(self.file_name)
        df = df.drop(['Id'],axis=1)
        target = df['Species']
        s = set()
        for val in target:
            s.add(val)
        s = list(s)
        rows = list(range(100,150))
        df = df.drop(df.index[rows])

        x = df['SepalLengthCm']
        y = df['PetalLengthCm']

        self.setosa_x = x[:50]
        self.setosa_y = y[:50]

        self.versicolor_x = x[50:]
        self.versicolor_y = y[50:]


        ## Drop rest of the features and extract the target values
        df = df.drop(['SepalWidthCm','PetalWidthCm'],axis=1)
        Y = []
        target = df['Species']
        for val in target:
            if (flagMulti):
                if(val == 'Iris-setosa'):
                    Y.append(0)
                else:
                    Y.append(1)                
            else:
                if(val == 'Iris-setosa'):
                    Y.append(-1)
                else:
                    Y.append(1)
        df = df.drop(['Species'],axis=1)
        X = df.values.tolist()

        ## Shuffle and split the data into training and test set
        X, Y = shuffle(X,Y)
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=train_size)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        self.x_train = x_train.reshape(90,2)
        self.x_test = x_test.reshape(10,2)
        self.y_train = y_train.reshape(90,1)
        self.y_test = y_test.reshape(10,1)

    def _countCostFunc(self, C = 1):
        y = self.x_train.dot(self.W) + self.b
        prod = 1 - y * self.y_train

        for i in range(len(prod)):
            prod[i] = max(0, prod[i])

        j = 0.5 * np.sum(self.W ** 2) + C * np.sum(prod)
        return j
        

    def train(self, epochs = 500, learning_rate = 0.001, C = 1):
        w_size = (self.x_train[0].shape[0], 1)
        self.W = np.random.randn(w_size[0], w_size[1])*0.01
        self.b = np.zeros((1, 1))

        epochs_count = 1
        cost_x = []
        cost_y = []
        while(epochs_count < epochs):
            y = self.x_train.dot(self.W) + self.b
            prod = y * self.y_train
            count = 0
            self.W -= -learning_rate * (C * self.x_train.T.dot(self.y_train * ((1 - prod) > 0))  + self.W)
            self.b -= -learning_rate * (C * self.y_train.T.dot(self.y_train * (1 - prod) > 0))
            epochs_count += 1
            cost_x.append(epochs_count)
            cost = self._countCostFunc(C)
            cost_y.append(cost)
        plt.figure(figsize=(8,6))
        plt.plot(cost_x, cost_y, 'k-')
        plt.title("cost for binary svm")
        plt.xlabel('epochs')
        plt.ylabel('cost')
        plt.show()
        return self.W, self.b

    def predict(self):
        y_pred = self.x_test.dot(self.W)  + self.b
        predictions = []
        for val in y_pred:
            if(val > 0):
                predictions.append(1)
            else:
                predictions.append(-1)

        accuracy_score_ans = accuracy_score(self.y_test,predictions)

        return accuracy_score_ans
    
    def plotDecisionBoundary(self):
        plt.figure(figsize=(8,6))
        plt.scatter(self.setosa_x,self.setosa_y,marker='+',color='green')
        plt.scatter(self.versicolor_x,self.versicolor_y,marker='_',color='red')

        number_of_points = len(self.W)
        a = -self.W[0] / self.W[1]

        xx = np.linspace(4, 7, num = number_of_points)
        xx = xx.reshape(number_of_points,1)

        yy = a * xx - self.b / self.W[1]
        plt.plot(xx, yy, 'r-')
        plt.title("decision boundary for binary svm")
        plt.xlabel('x1')
        plt.ylabel('x2')

        return plt

    # multi_svm

    def _countCostFuncMulti(self, delta = 1, lambda_= 0.001):

        num_train = len(self.y_train)
        scores = self.x_train.dot(self.W)
        correct_class_score = scores[np.arange(num_train), self.y_train.T].T
        margins = np.maximum(0, scores - correct_class_score + delta)

        margins[np.arange(num_train), self.y_train.T] = 0
        cost = np.sum(margins)
        cost /= num_train
        cost += 0.5 * lambda_ * np.sum(self.W * self.W)
        return cost
        

    def trainMulti(self, epochs = 500, learning_rate = 0.001, lambda_ = 0.001, delta = 1):
        w_size = (self.x_train[0].shape[0], np.max(self.y_train) + 1)
        self.W = np.random.randn(w_size[0], w_size[1])*0.01
        num_train = len(self.y_train)
        epochs_count = 1
        cost_x = []
        cost_y = []
        
        while(epochs_count < epochs):
            scores = self.x_train.dot(self.W)
            correct_class_score = scores[np.arange(num_train), self.y_train.T].T
            margins = np.maximum(0, scores - correct_class_score + delta)
            margins[np.arange(num_train), self.y_train.T] = 0
            X_mask = np.zeros(margins.shape)
            X_mask[margins > 0] = 1
            count = np.sum(X_mask,axis=1)

            X_mask[np.arange(num_train),self.y_train.T] = -count
            dW = self.x_train.T.dot(X_mask)

            dW /= num_train

            dW += lambda_* self.W

            self.W += -learning_rate * dW
            epochs_count += 1
            cost_x.append(epochs_count)
            cost = self._countCostFuncMulti(delta, lambda_)
            cost_y.append(cost)
        plt.figure(figsize=(8,6))
        plt.title("cost for multi svm")
        plt.xlabel('epochs')
        plt.ylabel('cost')
        plt.plot(cost_x, cost_y, 'k-')
        plt.show()
        return self.W

    def predictMulti(self):
        y_pred = self.x_test.dot(self.W) 
        predictions = []
        for val in y_pred:
            predictions.append(np.argmax(val))

        accuracy_score_ans = accuracy_score(self.y_test,predictions)

        return accuracy_score_ans
    
    def plotDecisionBoundaryMulti(self):
        plt.figure(figsize=(8,6))
        plt.scatter(self.setosa_x,self.setosa_y,marker='+',color='green')
        plt.scatter(self.versicolor_x,self.versicolor_y,marker='_',color='red')

        number_of_points = len(self.W)
        a = -(self.W[0, 0] - self.W[0, 1]) / (self.W[1, 0] - self.W[1, 1])

        xx = np.linspace(4, 7, num = number_of_points)
        xx = xx.reshape(number_of_points,1)

        yy = a * xx 
        plt.title("decision boundary for multi svm")
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.plot(xx, yy, 'r-')


        return plt

def main():
    lr = LogisticRegression('Iris.csv')
    lr.get_dataset()
    W = lr.train(epochs = 300, learning_rate = 0.0005, lambda_ = 0)
    accuracy_score = lr.predict()
    print('accuracy_score lr = ', accuracy_score)

    my_test = [[6, 1.5], [5, 3], [4.5, 1], [6, 5]]
    my_test = np.array(my_test)
    ans = lr._sigmoid(my_test.dot(W))
    ans2 = []
    for val in ans:
        if(val > 0.5):
            ans2.append('red')
        else:
            ans2.append('green')
    for i in range(len(my_test)):   
        print(my_test[i], " ", ans2[i])

    plt1 = lr.plotDecisionBoundary()
    plt1.show()


    svm = Svm('Iris.csv')
    svm.getDataset(flagMulti = True)
    W = svm.trainMulti(epochs = 10000, learning_rate = 0.0005, lambda_= 0.01)
    accuracy_score = svm.predictMulti()
    print('accuracy_score svm = ', accuracy_score)

    my_test = [[6, 1.5], [5, 3], [4.5, 1], [6, 5]]
    my_test = np.array(my_test)
    ans = my_test.dot(W) 
    ans2 = []

    for val in ans:
        if (np.argmax(val) == 1):
            ans2.append('red')
        else:
            ans2.append('green')
    for i in range(len(my_test)):   
        print(my_test[i], " ", ans2[i])

    plt2 = svm.plotDecisionBoundaryMulti()

    plt2.show()

    # # binary svm
    # svm = Svm('Iris.csv')
    # svm.getDataset(flagMulti = False)
    # W, b = svm.train(epochs = 100, learning_rate = 0.0005, C = 1)
    # accuracy_score = svm.predict()
    # print('accuracy_score svm = ', accuracy_score)

    # my_test = [[6, 1.5], [5, 3], [4.5, 1], [6, 5]]
    # my_test = np.array(my_test)
    # ans = my_test.dot(W) + b
    # ans2 = []

    # for val in ans:
    #     if (val > 0):
    #         ans2.append('red')
    #     else:
    #         ans2.append('green')
    # for i in range(len(my_test)):   
    #     print(my_test[i], " ", ans2[i])

    # plt3 = svm.plotDecisionBoundary()

    # plt3.show()

    # # framework realization
    # clf = SVC(kernel='linear',C = 100)
    # clf.fit(self.x_train,self.y_train.ravel())
    # y_pred = clf.predict(self.x_test)
    # print(accuracy_score(self.y_test,y_pred), ' framework')

    # w = clf.coef_[0]
    # a = -w[0] / w[1]
    # xx = np.linspace(4, 7, num = number_of_points)
    # yy = a * xx - (clf.intercept_[0]) / w[1]
    # plt.plot(xx, yy, 'b-')
    

main()






