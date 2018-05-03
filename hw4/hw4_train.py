
# coding: utf-8

# In[ ]:


import sys
import csv
import numpy as np

from keras.layers import Input, Dense
from keras.models import Model, load_model
from sklearn.cluster import KMeans


# In[ ]:


def readTestData():
    txt_data = open('./data/test_case.csv', 'r', encoding = 'big5')
    row = csv.reader(txt_data, delimiter=',')
    data = []
    num = 0
    for r in row:
        if num > 0:
            data.append([int(r[0]), int(r[1]), int(r[2])])
        num = num + 1
    data = np.array(data)
    return data


# In[ ]:


def autoencoder(data, dim, epoch):
    input_img = Input(shape=(784,))
    encoded = Dense(256, activation='relu')(input_img)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoder_output = Dense(dim, activation='relu')(encoded)
    
    decoded = Dense(64, activation='relu')(encoder_output)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(784, activation='relu')(decoded)
    
    autoencoder = Model(input = input_img, output = decoded)
    encoder = Model(input = input_img, output = encoder_output)
    
    autoencoder.compile(optimizer = 'adam', loss = 'mse')
    
    autoencoder.fit(x, x, epochs = epoch, batch_size = 256, shuffle = True)
    
    encoder.save('model.h5')


# In[ ]:


img_data = np.load('./data/image.npy')


# In[ ]:


x = img_data/255


# In[ ]:


autoencoder(x, 32, 100)

