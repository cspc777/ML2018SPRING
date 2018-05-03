
# coding: utf-8

# In[1]:


import sys
import numpy as np
import csv
from keras.models import load_model
from keras.utils import np_utils


# In[2]:


def readTestData():
    data = []
    txt_data = open('test.csv', 'r', encoding = 'big5')
    row = csv.reader(txt_data, delimiter=',')
    num = 0
    for r in row:
        if num > 0:
            word = r[1].strip().split(' ')
            element = list(map(float, word))
            data.append(element)            
        num = num + 1 
    data = np.array(data)
    return data


# In[3]:


test = readTestData()
test = test /255
test = np.reshape(test,(-1,48,48,1))


# In[4]:


model1 = load_model('model1.h5')
model2 = load_model('model2.h5')
model3 = load_model('model3.h5')
model4 = load_model('model.h5')


# In[5]:


models = [model1, model2, model3, model4]


# In[6]:


#ensemble
weight = 0.0
for i in range(len(models)):
    weight += models[i].predict(test)


# In[7]:


weight = weight / len(models)


# In[8]:


predict = np.argmax(weight, 1)


# In[9]:


predict = np.reshape(predict,(-1,1))
ans = []
for i in range(len(test)):
    ans.append([str(i)])
    ans[i].append(int(predict[i][0]))
filename = 'predict.csv'


# In[10]:


text = open(filename, 'w+')
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['id','label'])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()

