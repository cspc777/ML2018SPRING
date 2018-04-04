
# coding: utf-8

# In[44]:


import sys
import numpy as np
import csv
import math
import os
# In[45]:


#age #workclass 9(+?) #fnlwgt #education 16 #education_num 16 #marital_status 7 #occupation: 15(+?) #relationship 6
#race 5 #sex 2 #capital_gain #capital_loss #hours_per_week #native_country 42(+?)


# In[46]:


def readData():
    data_list = []
    with open(str(sys.argv[3]), 'r') as f:
        lines = f.readlines()
        for idx in range(1, len(lines)):
            word = lines[idx].strip().split(',')
            element = list(map(int, word))
            data_list.append(element)
    data = np.array(data_list)
    #word = lines[0].strip().split(',')
    #print(word[43])

    label_list=[]
    with open(str(sys.argv[4]), 'r') as f:
        lines = f.readlines()
        for line in lines:
            word = line.strip()
            element = list(map(int, word))
            label_list.append(element)
        label = np.array(label_list)
    return data, label


# In[47]:


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


# In[48]:


def shuffle(x, y):
    np.random.seed(11)
    random_idx = np.arange(x.shape[0])
    np.random.shuffle(random_idx)
    x = x[random_idx,:]
    y = y[random_idx]    
    return x, y


# In[49]:


def splitdata(x,y,size):
    x_r,y_r = shuffle(x,y)
    x_train, x_label_train = x_r[:size,:], y_r[:size]
    x_val, x_label_val = x_r[size:,:], y_r[size:]
    return x_train, x_label_train, x_val, x_label_val


# In[50]:


def normalize(x_train):
    mean = np.mean(x_train, 0)
    std = np.std(x_train, 0)
    x_train = (x_train - mean)/std
    return x_train, mean, std


# In[51]:


def generative_train(x, x_label):
    class0 = []
    class1 = []
    for i in range(len(x_label)):
            if x_label[i] == 0:
                class0.append(x[i])
            else:
                class1.append(x[i])
    class0 = np.array(class0)
    class1 = np.array(class1)
    PC0 = len(class0) / (len(class1) + len(class0))
    PC1 = len(class1) / (len(class1) + len(class0))
    mu0 = np.mean(class0, 0)
    mu1 = np.mean(class1, 0)
    sigma = PC0 * np.cov(class0.T) + PC1 * np.cov(class1.T)
    det = np.linalg.det(sigma) + 1e-20
    inv = np.linalg.pinv(sigma)
    return sigma, mu0, mu1, det, inv


# In[52]:


def gauss_dist(sigma, mu, det, inv, x):
    prob = (1 / np.sqrt((2 * math.pi)**len(x) * det)) * np.exp(-(1 / 2) * np.dot(np.dot((x - mu).T, inv), (x - mu)))
    return prob


# In[53]:


def accuracy(sigma, mu0, mu1, det, inv, x, y):
    count = 0
    for i in range(len(x)):
        c1 = gauss_dist(sigma, mu1, det, inv, x[i])
        c0 = gauss_dist(sigma, mu0, det, inv, x[i])
        class_result = (c1 > c0)
        if class_result == y[i]:
            count = count + 1
    return count / len(x)


# In[54]:


def classfication(sigma, mu0, mu1, det, inv, x_test):
    result = []
    for i in range(len(x_test)):
        c1 = gauss_dist(sigma, mu1, det, inv, x_test[i])
        c0 = gauss_dist(sigma, mu0, det, inv, x_test[i])
        if c1 > c0:
            result.append([1])
        else:
            result.append([0])
    return result


# In[55]:


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


# In[56]:


def getclass(feature_start, feature_end, no_count, x, x_label, flag):
    idx = []
    count_list = []
    for feature_idx in range(feature_start, feature_end):
        if feature_idx == no_count:
            continue
        else:
            count = 0
            for i in range(len(x_label)):
                if x_label[i] == flag:
                    continue
                else:
                    if x[i,feature_idx] == 1:
                        count += 1
            idx.append(feature_idx)
            count_list.append(count)
    return idx, count_list


# In[57]:


def preprocess_test_prob(x, x_label, x_test):
    workclass_idx, workclass_count_list = getclass(1, 10, 7, x, x_label, -2)
    occupation_idx, occupation_count_list = getclass(50, 65, 54, x, x_label, -2)
    native_country_idx, native_country_count_list = getclass(81, 122, 116, x, x_label, -2)
    np.save('./model/workclass_idx.npy', np.array(workclass_idx))
    np.save('./model/workclass_count_list', np.array(workclass_count_list))
    np.save('./model/occupation_idx.npy', np.array(occupation_idx))
    np.save('./model/occupation_count_list', np.array(occupation_count_list))
    np.save('./model/native_country_idx.npy', np.array(native_country_idx))
    np.save('./model/native_country_count_list', np.array(native_country_count_list))
    x = np.delete(x, (7,54,116), 1)
    return x


# In[58]:


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


# In[59]:


def preprocess(x, x_label):
    workclass_idx, workclass_count_list = getclass(1, 10, 7, x, x_label, 0)
    occupation_idx, occupation_count_list = getclass(50, 65, 54, x, x_label, 0)
    native_country_idx, native_country_count_list = getclass(81, 122, 116, x, x_label, 0)
    for i in range(len(x)):
        if x_label[i] == 0:
            if x[i,7] == 1:
                idx = 4
                x[i,idx] = 1
            if x[i,54] == 1:
                idx = np.array(occupation_count_list).argmin()
                x[i,idx] = 1
            if x[i,116] == 1:
                idx = np.array(native_country_count_list).argmin()
                x[i,idx] = 1
        else:
            if x[i,7] == 1:
                idx = random_pick(workclass_idx, workclass_count_list)
                x[i,idx] = 1
            if x[i,54] == 1:
                idx = random_pick(occupation_idx, occupation_count_list)
                x[i,idx] = 1
            if x[i,116] == 1:
                idx = random_pick(native_country_idx, native_country_count_list)
                x[i,idx] = 1
    return x, x_label


# In[60]:


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


# In[61]:


def extra_process(data):
    data = np.concatenate((data, (data[:, :5])**2), axis=1)
    data = np.concatenate((data, (data[:, :5])**3), axis=1)
    data = np.concatenate((data, np.log(data[:, :5]+1)), axis=1)

    return data


# In[62]:
if not os.path.isdir('./model'): 
    os.mkdir('./model')

x, x_label = readData()
#print(len(x)) #32561
test = readTestData()

x, x_label = preprocess(x, x_label)
x = preprocess_test_prob(x, x_label, test)

x = re_arrange(x)
x = extra_process(x)

x, mean , std = normalize(x)

x_train, y_train, x_val, y_val = splitdata(x,x_label,30000)

sigma, mu0, mu1, det, inv = generative_train(x_train, y_train)
print('accuracy: %f' %(accuracy(sigma, mu0, mu1, det, inv, x_train, y_train)))
print('val_accuracy: %f' %(accuracy(sigma, mu0, mu1, det, inv, x_val, y_val)))

test = preprocess_test(test)
test = re_arrange(test)
test = extra_process(test)
test = (test-mean)/std

#get ans.csv
ans = []
result = classfication(sigma, mu0, mu1, det, inv, test)
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

