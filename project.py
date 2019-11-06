# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset = dataset.dropna()
x = dataset.iloc[:, [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]].values
y = dataset.iloc[:, 12].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_0 = LabelEncoder()
x[:, 0] = labelencoder_x_0.fit_transform(x[:, 0])
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
labelencoder_x_3 = LabelEncoder()
x[:, 3] = labelencoder_x_3.fit_transform(x[:, 3])
labelencoder_x_9 = LabelEncoder()
x[:, 9] = labelencoder_x_9.fit_transform(x[:, 9])
onehotencoder = OneHotEncoder(categorical_features = [9])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, :11]

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)

# Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(x_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc = (cm[0][0] + cm[1][1])*100/len(y_pred)
print('Accuracy of Logistic Regression:')
print(acc)

# Fitting K-NN classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski')
classifier.fit(x_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(x_test)

# Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc = (cm[0][0] + cm[1][1])*100/len(y_pred)
print('Accuracy of K-NN:')
print(acc)

# Fitting Naive Bayes classifier to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(x_test)

# Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc = (cm[0][0] + cm[1][1])*100/len(y_pred)
print('Accuracy of Naive Bayes:')
print(acc)
