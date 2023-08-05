# Importing libraries
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import pickle

dataFile = 'Datas/diabetsData.pickle';
modelFile = 'Models/diabetsModel.sav';

# ================================================================================================

dataset = pd.read_csv('Datasets/diabetes.csv');

target = 'Outcome';
columns = dataset.columns.values.tolist();
columns.remove(target);

# Extract Features
X = dataset[columns];

# Extract Class Labels
y = dataset[target];

# ================================================================================================

# Split Dataset
# 75% for training model and 25% for testing model
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0);

# Normalize features
scaler = StandardScaler();
scaler.fit(X_train);
X_train = scaler.transform(X_train);

# ================================================================================================

# Now we create the SVM model
# OBS: The Sci-kit Learn library has four SVM kernels (linear, poly, rbf, sigmoid kernels)

# We iterate through the kernels and see which one gives us the best decision boundary for the dataset.
# The decision boundary is the hyperplane or curve that separates the positive class and the negative class. 
# It could be linear or non-linear.

# OBS: The polynomial and RBF kernels are suitable when the classes are not linearly separable.

#kernel = {'type': '', 'acc': 0.0}

#for k in ('linear', 'poly', 'rbf', 'sigmoid'):
#    model = svm.SVC(kernel=k)
#    model.fit(X_train, y_train)
#    y_pred = model.predict(X_train)
#    acc = round(accuracy_score(y_train, y_pred) * 100, 2);
#    print(k + ': ' + str(acc) + '%')
#    if acc > kernel['acc']:
#        kernel['acc'] = acc;
#        kernel['type'] = k;

#Choose the best kernel for create model
#model = svm.SVC(kernel=kernel['type']);
#model.fit(X_train, y_train);

#model = {'kernel': kernel['type'], 'model': model}

#pick = open(modelFile, 'wb');
#pickle.dump(model, pick);
#pick.close();

#print('Model salvo com sucesso!');

# ================================================================================================

# Ler arquivo do modelo
pick = open(modelFile, 'rb');
model = pickle.load(pick);
pick.close();

kernel = model['kernel'];
model = model['model'];

print('Tipo de Kernel utilizado: ' + kernel);

# ================================================================================================

# Making a single prediction
patient = pd.DataFrame(np.array([[ 1., 150., 70., 45., 0., 40., 1.5, 25]]), columns=columns)
#patient = pd.DataFrame(np.array([[ 1., 50., 70., 45., 0., 40., 1.5, 25]]), columns=columns)

# Normalize the data with the values used in the training set
patient = scaler.transform(patient);

result = model.predict(patient);
print('\n' + "Resultado: " + str(result));

# ================================================================================================

# Checking the third patient in the test set with index 2
#pos = 2;
#print(X_test.iloc[pos]);

# Convert dataframe to a numpy array
#patient = pd.DataFrame(np.array([ X_test.iloc[pos]]), columns=columns);

# Predicting on third patient in Test Set
#patient = scaler.transform(patient);
    
#print("Predição do Modelo:", model.predict(patient));
#print("Predição atual:", y_test.iloc[pos]);

# ================================================================================================

# Accuracy on Testing Set
X_test = scaler.transform(X_test);
y_pred = model.predict(X_test);
acc = round(accuracy_score(y_test, y_pred) * 100, 2);
print('\nAcurácia do Modelo: ' + str(acc) + '%');

# Compute precision, recall and f1 score
precision = round(precision_score(y_test, y_pred) * 100, 2);
recall = round(recall_score(y_test, y_pred) * 100, 2);
f1 = round(f1_score(y_test, y_pred) * 100, 2);

print("\nPrecision: " + str(precision) + '%');
print('Recall: ' + str(recall) + '%');
print('F1: ' + str(f1) + '%\n');

# Generate classification report
#print(classification_report(y_test, y_pred))