# -*- coding: utf-8 -*-

from sklearn.neural_network import MLPClassifier
import numpy as np


# Load the cats and dogs 
import load_data


X_train, y_train = load_data.loadData(_set='train',batch_iter=0,batch_size=100)

clf = MLPClassifier(solver='lbfgs', activation='logistic',alpha=1e-2,hidden_layer_sizes=(1000, 1000), random_state=1)

clf.fit(X_train.T, np.ravel(y_train.T))   

out = clf.predict(X_train.T)

print(np.mean(out == np.ravel(y_train.T)) * 100)

X_test, y_test = load_data.loadData(32,_set='test')

out2 = clf.predict(X_test.T)




print(np.mean(out2 == np.ravel(y_train.T)) * 100)
