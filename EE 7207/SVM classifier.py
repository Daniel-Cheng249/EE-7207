# Editor Name: Daniel Cheng
# Edit Time: 19:20 2021/9/26

from scipy.io import loadmat
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

PATH = 'Data/'
data_train = loadmat(PATH + 'data_train.mat')['data_train']
label_train = loadmat(PATH + 'label_train.mat')['label_train']
data_test = loadmat(PATH + 'data_test.mat')['data_test']

# separate training data into 2 sets: training set/validation set = 8/2
x_train, x_val, x_train_label, x_val_label = train_test_split(data_train, label_train, random_state=1, train_size=0.8, test_size=0.2)

# C=1, kernel='rbf', gamma=0.5
model_train = svm.SVC(C=1, kernel='rbf', gamma=0.5, decision_function_shape='ovo')
model_train.fit(x_train, x_train_label.ravel()) # convert x_train_label into 1-D ndarray

train_score = model_train.score(x_train, x_train_label)
print("The score of training set is：", train_score)

val_score = model_train.score(x_val, x_val_label)
print("The score of validation set is：", val_score)

test_result = model_train.predict(data_test)
print("Test data result is：", test_result)

# print('train_decision_function:\n', model_train.decision_function(x_train))
