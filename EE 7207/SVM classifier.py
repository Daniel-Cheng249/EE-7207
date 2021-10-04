# Editor Name: Daniel Cheng
# Edit Time: 19:20 2021/9/26

from scipy.io import loadmat
from sklearn import svm
from sklearn.model_selection import train_test_split

file1 = 'C:\\Users\\程泽\\Desktop\\Continuous Assessment\\data_train.mat'
file2 = 'C:\\Users\\程泽\\Desktop\\Continuous Assessment\\label_train.mat'
file3 = 'C:\\Users\\程泽\\Desktop\\Continuous Assessment\\data_test.mat'
data_train = loadmat(file1, mat_dtype=True)
label_train = loadmat(file2, mat_dtype=True)
data_test = loadmat(file3, mat_dtype=True)

x = data_train.get('data_train')
y = label_train.get('label_train')
x_test=data_test.get('data_test')
# print(len(x_test))

# separate training data into 2 sets: training set/validation set = 9/1
x_train, x_val, x_train_label, x_val_label = train_test_split(x, y, random_state=1, train_size=0.8, test_size=0.2)

model_train = svm.SVC(C=1, kernel='rbf', gamma=0.1, decision_function_shape='ovo')
model_train.fit(x_train, x_train_label.ravel()) # convert x_train_label into 1-D ndarray

train_score = model_train.score(x_train, x_train_label)
print("The score of training set is：", train_score)

val_score = model_train.score(x_val, x_val_label)
print("The score of validation set is：", val_score)

test_result = model_train.predict(x_test)
print("Test data result is：", test_result)

# print('train_decision_function:\n', model_train.decision_function(x_train))

