
# coding: utf-8

# In[ ]:


import numpy as np
import pickle as pk
import sys, argparse, os, csv
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, GRU, LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[ ]:


def readData(file_path):
    x=[]
    y=[]
    with open(file_path, 'r') as f:
        for line in f:
            lines = line.strip().split(' +++$+++ ')
            x.append(lines[1])
            y.append(int(lines[0]))
    return x, y


# In[ ]:


def read_nolabel_Data(file_path):
    x=[]
    with open(file_path, 'r') as f:
        for line in f:
            x.append(line)
    return x


# In[ ]:


def read_test_Data(file_path):
    test=[]
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for idx in range(len(lines)):
            if idx != 0:
                line = lines[idx].strip().split(',',1)
                test.append(line[1])
    return test


# In[ ]:


def token(data, vocab_size, maxlen):
    tokenizer = Tokenizer(num_words = vocab_size, filters='\n')
    tokenizer.fit_on_texts(data)
    pk.dump(tokenizer, open('token.pk', 'wb'))
    data = tokenizer.texts_to_sequences(data)
    #print(len(tokenizer.word_index))
    data = pad_sequences(data, maxlen=maxlen)
    return data


# In[ ]:


training_file = str(sys.argv[1])


# In[ ]:


x_train, y_train = readData(training_file)
x_train = token(x_train, 20000, 32) #82945


# In[ ]:


model = Sequential()


# In[ ]:


model.add(Embedding(20000+1, 256, input_length=32, embeddings_initializer=keras.initializers.random_normal(stddev=1.0))) #256
model.add(GRU(128, return_sequences=True, dropout=0.4, recurrent_dropout=0.2)) #bid(lstm 128=>64)
model.add(GRU(64, dropout=0.4, recurrent_dropout=0.2))
model.add(Dense(32))
model.add(LeakyReLU(alpha=0.03))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
save_path = os.path.join('model.h5')
earlystopping = EarlyStopping(monitor='val_acc', patience=1, mode='max')
checkpoint = ModelCheckpoint(filepath=save_path, save_best_only=True, 
                             save_weights_only=True, monitor='val_acc', mode='max')


# In[ ]:


model.fit(x_train, y_train, batch_size = 512, epochs = 20 , validation_split=0.1, callbacks=[checkpoint, earlystopping]) #20


# In[ ]:


model.save_weights('model.h5')


# In[ ]:


#semi-supervised
# nolabel_file = str(sys.argv[2])
# no_label = read_nolabel_Data(nolabel_file)
# with open('token.pk', 'rb') as f:
#     tokenizer = pk.load(f)
# no_label_to = tokenizer.texts_to_sequences(no_label)
# no_label_to = pad_sequences(no_label_to, maxlen=32)
# predict = model.predict(no_label_to, batch_size = 512)
# new_data = []
# new_label = []

# for i in range(len(predict)):
#     if(predict[i]>=0.9):
#         new_data.append(no_label[i])
#         new_label.append(int(1))
#     elif(predict[i]<=0.1):
#         new_data.append(no_label[i])
#         new_label.append(int(0))


# In[ ]:


# new_data  = token(new_data, 20000, 32) #82945
# model.fit(new_data, new_label, batch_size = 512, epochs = 10 , validation_split=0.1, callbacks=[checkpoint, earlystopping]) #20
# model.save('model.h5')

