
# coding: utf-8

# In[2]:


import sys
import csv
import numpy as np
from keras import backend as K
from keras.layers import Input, Embedding, Flatten, Dot, Add, Concatenate, Dense, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import load_model


# In[ ]:


def readtrainingData(filename):
    users = []
    movies = []
    rating = []
    txt_data = open(filename, 'r', encoding = 'big5')
    row = csv.reader(txt_data, delimiter=',')
    num = 0
    for r in row:
        if num > 0:
            users.append(int(r[1]))
            movies.append(int(r[2]))
            rating.append(float(r[3]))           
        num = num + 1
    users = np.array(users)
    movies = np.array(movies)
    rating = np.array(rating)
    idx = np.arange(users.shape[0])
    np.random.shuffle(idx)
    users = users[idx]
    movies = movies[idx]
    rating = rating[idx]
    return users, movies, rating


# In[ ]:


def readtestData(filename):
    users = []
    movies = []
    txt_data = open(filename, 'r', encoding = 'big5')
    row = csv.reader(txt_data, delimiter=',')
    num = 0
    for r in row:
        if num > 0:
            users.append(int(r[1]))
            movies.append(int(r[2]))          
        num = num + 1
    users = np.array(users)
    movies = np.array(movies)
    return users, movies


# In[6]:


def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1., 5.)
    return K.sqrt(K.mean(K.square((y_true - y_pred))))


# In[4]:


dim = 16


# In[7]:


user_input = Input(shape=[1])
movie_input = Input(shape=[1])
user_emb = Embedding(6041, dim, embeddings_initializer='random_normal')(user_input)
user_emb = Flatten()(user_emb)

movie_emb = Embedding(3953, dim, embeddings_initializer='random_normal')(movie_input)
movie_emb = Flatten()(movie_emb)

user_bias = Embedding(6041, 1, embeddings_initializer='zeros')(user_input)
user_bias = Flatten()(user_bias)

movie_bias = Embedding(3953, 1, embeddings_initializer='zeros')(movie_input)
movie_bias = Flatten()(movie_bias)

R = Dot(axes=1)([user_emb, movie_emb])
R = Add()([R, user_bias, movie_bias])
model = Model([user_input, movie_input], R)
model.compile(loss='mse', optimizer='adamax', metrics=[rmse])
model.summary()


# In[9]:


model.load_weights('model.h5')


# In[ ]:


users_test , movies_test = readtestData(str(sys.argv[1]))


# In[ ]:


predict = model.predict([users_test, movies_test], batch_size=512)


# In[ ]:


predict = np.clip(predict, 1, 5)
predict = predict.reshape(-1)


# In[ ]:


ans = []
for i in range(len(predict)):
    ans.append([str(i+1)])
    ans[i].append(predict[i])
filename = str(sys.argv[2])
text = open(filename, 'w+')
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['TestDataID','Rating'])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()

