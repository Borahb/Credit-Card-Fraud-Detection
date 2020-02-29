# Credit Card fraud detection using ANN


"""
Created on Fri Feb 28 15:15:48 2020

@author: Bhaskar
"""
#importing of libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#importing the dataset
dataset = pd.read_csv('creditcard.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state = 0)

#feature scaling the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# applying PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components= 3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
variance = pca.explained_variance_ratio_

#Constructing ANN


#importing the keras library

from keras.models import Sequential
from  keras.layers import Dense

#initializing the ANN
classifier = Sequential()

#adding the input layer and first hidden layer

classifier.add(Dense(output_dim=8,init='uniform',activation = 'relu',input_dim = 3))

#adding the second hidden layer
classifier.add(Dense(output_dim=8,init='uniform',activation = 'relu'))

#adding the third hidden layer
classifier.add(Dense(output_dim=8,init='uniform',activation = 'relu'))

#Adding the output layer

classifier.add(Dense(output_dim=1,init='uniform',activation = 'sigmoid'))

#compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN  to the training set
classifier.fit(X_train,y_train, batch_size = 20, nb_epoch=50)

#Predicting the results
y_pred = classifier.predict(X_test)


 
# save model
pickle.dump(classifier, open('fraud_detect.pickle', 'wb'))
