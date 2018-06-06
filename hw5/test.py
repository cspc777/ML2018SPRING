
# coding: utf-8

# In[1]:


import numpy as np
import pickle as pk
import sys, argparse, os, csv
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, GRU, LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[2]:


def read_test_Data(file_path):
    test=[]
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for idx in range(len(lines)):
            if idx != 0:
                line = lines[idx].strip().split(',',1)
                test.append(line[1])
    return test


# In[3]:


model = Sequential()


# In[4]:


model.add(Embedding(20000+1, 256, input_length=32, embeddings_initializer=keras.initializers.random_normal(stddev=1.0))) #256
model.add(GRU(128, return_sequences=True, dropout=0.4, recurrent_dropout=0.2)) #bid(lstm 128=>64)
model.add(GRU(64, dropout=0.4, recurrent_dropout=0.2))
model.add(Dense(32))
model.add(LeakyReLU(alpha=0.03))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()


# In[5]:


model.load_weights('model.h5')


# In[ ]:


x_test = read_test_Data(str(sys.argv[1]))


# In[ ]:


with open('token.pk', 'rb') as f:
    tokenizer = pk.load(f)
x_test = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, maxlen=32)


# In[ ]:


predict = model.predict(x_test, batch_size = 512)


# In[ ]:


ans = []
for i in range(len(predict)):
    ans.append([str(i)])
    ans[i].append(int(predict[i]>=0.5))
filename = str(sys.argv[2])
text = open(filename, 'w+')
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['id','label'])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()

