
# coding: utf-8

# In[1]:


import sys
import numpy as np
import csv


# In[ ]:


def readTestData():
    data_list = []
    with open(str(sys.argv[5]), 'r') as f:
        lines = f.readlines()
        for idx in range(1, len(lines)):
            word = lines[idx].strip().split(',')
            element = list(map(int, word))
            data_list.append(element)
    data = np.array(data_list)

    return data


# In[ ]:


def sigmoid(z):
    ans = np.clip( 1 / (1.0+np.exp(-z)), 0.00000000000001, 0.99999999999999 )
    return ans


# In[ ]:

def random_pick(seq, probabilities):
    prob = np.array(probabilities)
    sum_all = np.sum(prob)
    prob = prob / sum_all
    x = np.random.uniform()
    cumulative_prob = 0.0
    for item, item_prob in zip(seq, prob):
        cumulative_prob+=item_prob 
        if x < cumulative_prob:
            return item

def preprocess_test(x_test):
    workclass_idx = list(np.load('./model/workclass_idx.npy'))
    workclass_count_list = list(np.load('./model/workclass_count_list.npy'))
    occupation_idx = list(np.load('./model/occupation_idx.npy'))
    occupation_count_list = list(np.load('./model/occupation_count_list.npy'))
    native_country_idx = list(np.load('./model/native_country_idx.npy'))
    native_country_count_list = list(np.load('./model/native_country_count_list.npy'))
    for i in range(len(x_test)):
        if x_test[i,7] == 1:
            idx = random_pick(workclass_idx, workclass_count_list)
            x_test[i,idx] = 1
        if x_test[i,54] == 1:
            idx = random_pick(occupation_idx, occupation_count_list)
            x_test[i,idx] = 1
        if x_test[i,116] == 1:
            idx = random_pick(native_country_idx, native_country_count_list)
            x_test[i,idx] = 1
    x_test = np.delete(x_test, (7,54,116), 1)
    return x_test


# In[ ]:


def re_arrange(data):      
    tmp = []
    age = data[:,0]
    fnlwgt = data[:,9]
    capital_gain = data[:,76]
    capital_loss = data[:,77]
    hours_per_week = data[:,78]
    
    tmp.append(age)
    tmp.append(fnlwgt)
    tmp.append(capital_gain)
    tmp.append(capital_loss)
    tmp.append(hours_per_week)
    tmp = np.array(tmp)
    tmp = tmp.transpose()
    data = np.delete(data, (0,9,76,77,78), 1)
    data = np.concatenate((tmp, data), axis=1)
    return data


# In[ ]:


def extra_process(data):
    data = np.concatenate((data, (data[:, :5])**2), axis=1)
    data = np.concatenate((data, (data[:, :5])**3), axis=1)
    data = np.concatenate((data, np.log(data[:, :5]+1)), axis=1)

    return data


# In[ ]:


test = readTestData()

#load mean std
mean = np.load('./model/mean.npy')
std = np.load('./model/std.npy')

#load x, x_label

test = preprocess_test(test)
test = re_arrange(test)
test = extra_process(test)
test = (test-mean)/std


# In[ ]:


#read model
w = np.load('./model/model_w.npy')
b = np.load('./model/model_b.npy')

#get ans.csv
ans = []
result = sigmoid(np.dot(test, w) + b)
result = np.round(result)
for i in range(len(test)):
    ans.append([str(i+1)])
    ans[i].append(int(result[i][0]))

filename = str(sys.argv[6])
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['id','label'])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()

