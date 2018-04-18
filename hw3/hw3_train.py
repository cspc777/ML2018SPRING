
# coding: utf-8

# In[1]:


import sys
import numpy as np
import csv
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator


# In[2]:


def readData():
    data = []
    label = []
    txt_data = open(str(sys.argv[1]), 'r', encoding = 'big5')
    row = csv.reader(txt_data, delimiter=',')
    num = 0
    for r in row:
        if num > 0:
            label.append([int(r[0])])
            
            word = r[1].strip().split(' ')
            element = list(map(float, word))
            data.append(element)            
        num = num + 1
    data = np.array(data)
    label = np.array(label)
    return data, label


# In[3]:


def shuffle(x, y):
    np.random.seed(11)
    random_idx = np.arange(x.shape[0])
    np.random.shuffle(random_idx)
    x = x[random_idx,:]
    y = y[random_idx]    
    return x, y


# In[4]:


def splitdata(x,y,size):
    x_r,y_r = shuffle(x,y)
    x_train, y_train = x_r[:size,:], y_r[:size]
    x_val, y_val = x_r[size:,:], y_r[size:]
    return x_train, y_train, x_val, y_val


# In[5]:


x, y = readData()
x = x / 255

x_train, y_train, x_val, y_val = splitdata(x,y,len(x)-3500)


# In[6]:


x_train = np.reshape(x_train,(-1,48,48,1))
x_val = np.reshape(x_val,(-1,48,48,1))


# In[7]:


# Image PreProcessing
gen_img = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True)
gen_img.fit(x_train)


# In[8]:


y_train_OneHot = np_utils.to_categorical(y_train)
y_val_OneHot = np_utils.to_categorical(y_val)


# In[9]:


model = Sequential()


# In[10]:


model.add(Conv2D(filters = 32, kernel_size = (3, 3),input_shape=(48,48,1)
                 , activation='relu', padding='same'))
model.add(Conv2D(filters = 32, kernel_size = (3, 3)
                 , activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = (3, 3)
                 ,activation='relu', padding='same'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3)
                 ,activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 128, kernel_size = (3, 3)
                 ,activation='relu', padding='same'))
model.add(Conv2D(filters = 128, kernel_size = (3, 3)
                 ,activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.4))

model.add(Conv2D(filters = 256, kernel_size = (3, 3)
                 ,activation='relu', padding='same'))
model.add(Conv2D(filters = 256, kernel_size = (3, 3)
                 ,activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))


# In[11]:


print(model.summary())


# In[12]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[13]:


train_history = model.fit_generator(gen_img.flow(x_train, y_train_OneHot, batch_size=50), samples_per_epoch=50*x_train.shape[0]//50, epochs = 160, validation_data=(x_val, y_val_OneHot)) 


# In[14]:


model.save('model.h5')

