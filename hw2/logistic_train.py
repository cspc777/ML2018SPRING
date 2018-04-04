
# coding: utf-8

# In[1]:


import sys
import numpy as np
import csv
import os


# In[2]:


#age #workclass 9(+?) #fnlwgt #education 16 #education_num 16 #marital_status 7 #occupation: 15(+?) #relationship 6
#race 5 #sex 2 #capital_gain #capital_loss #hours_per_week #native_country 42(+?)


# In[3]:


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


# In[4]:


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


# In[5]:


def shuffle(x, y):
    np.random.seed(11)
    random_idx = np.arange(x.shape[0])
    np.random.shuffle(random_idx)
    x = x[random_idx,:]
    y = y[random_idx]    
    return x, y


# In[6]:


def splitdata(x,y,size):
    x_r,y_r = shuffle(x,y)
    x_train, x_label_train = x_r[:size,:], y_r[:size]
    x_val, x_label_val = x_r[size:,:], y_r[size:]
    return x_train, x_label_train, x_val, x_label_val


# In[7]:


def normalize(x_train):
    mean = np.mean(x_train, 0)
    std = np.std(x_train, 0)
    x_train = (x_train - mean)/std
    return x_train, mean, std


# In[8]:


def sigmoid(z):
    ans = np.clip( 1 / (1.0+np.exp(-z)), 0.00000000000001, 0.99999999999999 )
    return ans


# In[9]:


def accuracy(x, label, w, b):
    correct = 0 
    y = sigmoid(np.dot(x, w) + b)
    y = np.round(y)
    for i in range(len(y)):
        if y[i] == label[i]:
            correct = correct + 1
    out = correct/len(y)
    return out


# In[10]:


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


# In[11]:


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


# In[12]:


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


# In[13]:


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


# In[14]:


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


# In[15]:


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


# In[16]:


def extra_process(data):
    data = np.concatenate((data, (data[:, :5])**2), axis=1)
    data = np.concatenate((data, (data[:, :5])**3), axis=1)
    data = np.concatenate((data, np.log(data[:, :5]+1)), axis=1)

    return data


# In[17]:
if not os.path.isdir('./model'): 
    os.mkdir('./model')

x, x_label = readData()
#print(len(x)) #32561
test = readTestData()
#print(x[0,123])

x, x_label = preprocess(x, x_label)
x = preprocess_test_prob(x, x_label, test)

x = re_arrange(x)
x = extra_process(x)

x, mean , std = normalize(x)
#save mean, std
np.save('./model/mean.npy', mean)
np.save('./model/std.npy', std)

x_train, y_train, x_val, y_val = splitdata(x,x_label,30000)
#print(x_train[0])
#print(y_train[0])

#traning
w = np.ones((len(x_train[0]),1))
b = 0
landa = 0
learning_rate = 0.05
iteration = 30000
x_t = x_train.transpose()

sum_gra_w = np.zeros((len(x_train[0]),1))
sum_gra_b = 0

for i in range(iteration):
    sig = sigmoid(np.dot(x_train, w) + b)
    
    loss = -(y_train*np.log(sig) + (1-y_train)*np.log(1-sig))   

    gra_w = -np.dot(x_t, y_train - sig) + 2 * landa * w
    gra_b = -np.sum(y_train - sig)
    
    sum_gra_w += gra_w**2
    sum_gra_b += gra_b**2
    ada_w = np.sqrt(sum_gra_w)
    ada_b = np.sqrt(sum_gra_b)
    
    w = w - learning_rate * gra_w/ada_w
    b = b - learning_rate * gra_b/ada_b
    
    #kaggle score evaluation
    cost = np.mean(loss)
    if(i%1000 == 0):
        print('iteration: %d,  loss:%f' %(i, cost))
print('accuracy:%f' %(accuracy(x_train, y_train, w, b)))
#kaggle score evaluation val
print('cal,  accuracy:%f' %(accuracy(x_val, y_val, w, b)))

#save/read model
#save model
np.save('./model/model_w.npy', w)
np.save('./model/model_b.npy', b)

