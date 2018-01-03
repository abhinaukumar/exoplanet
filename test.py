# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 09:38:25 2018

@author: nownow
"""

import pandas
import numpy as np
import scipy
import keras
from sklearn.externals import joblib

test_data = pandas.read_csv('../data/Final_Test.csv')
test_data = test_data.values

x_test = test_data[:,1:]
y_test = test_data[:,0]


print "Loaded data"


print "Partitioned data"

m = np.mean(x_test,axis=1)
s = np.std(x_test,axis=1)

x_test = (x_test - np.tile(m,(3197,1)).T)/np.tile(s,(3197,1)).T

print "Normalized data"

x_test = scipy.ndimage.filters.gaussian_filter(x_test,13)


m_s_data = np.load(file('../m_s_3_1_20_04.npy','r'))

m = m_s_data['m']
s = m_s_data['s']

pca = joblib.load('../pca_3_1_20_04.pkl')

encoder = keras.models.load_model('../encoder_3_1_20_19.json')

x_test = (x_test - m)/s

x_test = pca.transform(x_test)

pred_test = encoder.predict(x_test)

kl_test = - 0.5 * np.mean(1 + pred_test[1] - pred_test[0]**2 - np.exp(pred_test[1]), axis=-1)

label_test = 1 + np.ndarray.astype(kl_test > 3.2,'int')

np.savetxt('predictions.csv',label_test,delimiter=',')
