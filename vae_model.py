# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 20:30:24 2017

@author: nownow
"""

import pandas
import numpy as np
from sklearn.decomposition import PCA
import scipy
import matplotlib.pyplot as plt
import scipy.ndimage

import keras
from keras.layers import Input, Dense, Merge
from keras.models import Model
import keras.backend as K
    
plt.ion()

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(1, 400), mean=0.,
                              stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon
    
data = pandas.read_csv('../data/ExoTrain.csv')
data = data.values

x_data = data[:,1:]
y_data = data[:,0]


print "Loaded data"

x_1 = x_data[y_data == 1]
ind = np.random.permutation(len(x_1))
x_1 = x_1[ind]

x_2 = x_data[y_data == 2]
ind = np.random.permutation(len(x_2))
x_2 = x_2[ind]

print "Partitioned data"

m = np.mean(x_1,axis=1)
s = np.std(x_1,axis=1)

x_1 = (x_1 - np.tile(m,(3197,1)).T)/np.tile(s,(3197,1)).T

m = np.mean(x_2,axis=1)
s = np.std(x_2,axis=1)

x_2 = (x_2 - np.tile(m,(3197,1)).T)/np.tile(s,(3197,1)).T


print "Normalized data"

x_1 = scipy.ndimage.filters.gaussian_filter(x_1,13)
x_2 = scipy.ndimage.filters.gaussian_filter(x_2,13)

print "Filtered data"

ind = np.random.permutation(len(x_1))
x_1 = x_1[ind]

x_train = x_1[:int(len(x_1)*0.8),:]

x_test = x_1[int(len(x_1)*0.8):,:]

m = np.mean(x_train,axis=0)
s = np.std(x_train,axis=0)

x_train = (x_train - m)/s
x_test = (x_test - m)/s

x_2 = (x_2 - m)/s


print "Applied PCA"

pca = PCA(n_components=800)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
x_2 = pca.transform(x_2)

print "Training neural network"

l = 0.5

x = Input((800,))

h_mean = Dense(400,activation='linear',kernel_initializer=keras.initializers.zeros())(x)
h_log_var = Dense(400,activation='linear',kernel_initializer=keras.initializers.zeros())(x)

h = Merge(mode=sampling,output_shape=(400,))([h_mean,h_log_var])

y = Dense(800,activation='linear',kernel_initializer=keras.initializers.zeros() )(h)

def vae_loss(x_, x_decoded_mean):
    
    x_loss = 400*(keras.metrics.mean_squared_error(x_, x_decoded_mean))    
    kl_loss = - 0.5 * K.mean(1 + h_log_var - K.square(h_mean) - K.exp(h_log_var), axis=-1)
    return K.mean(x_loss + kl_loss)
    
model = Model(x,y)
model.compile(optimizer=keras.optimizers.adadelta(2),loss=vae_loss)

model.fit(x_train, x_train, validation_data=(x_test,x_test),
          batch_size=1,
          epochs=250,
          verbose=1,
          callbacks=[keras.callbacks.ReduceLROnPlateau(patience=3),
                     keras.callbacks.EarlyStopping(patience=7), 
                     keras.callbacks.ModelCheckpoint('../pca_13_model_weights_3_1_18_08.h5',save_best_only=True,save_weights_only=True)])
          
encoder = Model(x,[h_mean,h_log_var])

pred_train = encoder.predict(x_train)
pred_test = encoder.predict(x_test)
pred_pos = encoder.predict(x_2)

kl_train = - 0.5 * np.mean(1 + pred_train[1] - pred_train[0]**2 - np.exp(pred_train[1]), axis=-1)
kl_test = - 0.5 * np.mean(1 + pred_test[1] - pred_test[0]**2 - np.exp(pred_test[1]), axis=-1)
kl_pos = - 0.5 * np.mean(1 + pred_pos[1] - pred_pos[0]**2 - np.exp(pred_pos[1]), axis=-1)

plt.scatter(kl_train,0*np.ones_like(kl_train))
plt.scatter(kl_test,1*np.ones_like(kl_test))
plt.scatter(kl_pos,2*np.ones_like(kl_pos))
