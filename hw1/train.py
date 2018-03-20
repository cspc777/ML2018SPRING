
# coding: utf-8

# In[3]:


import csv
import numpy as np
import sys


# In[4]:


def readData():
    #read data
    data = []
    for i in range(18):
        data.append([])
    
    txt_data = open('train.csv', 'r', encoding = 'big5')
    row = csv.reader(txt_data, delimiter=',')
    num_category = 0
    for r in row:
        if num_category > 0:    #remove first row
            for c in range(3, 27):
                if r[c] == 'NR':
                    data[(num_category-1)%18].append(float(0))
                else:
                    data[(num_category-1)%18].append(float(r[c]))
        num_category = num_category + 1
    
    for i in range(len(data[9])):
        if(data[9][i] < 0):
            data[9][i] = 0
    #print(data[9])
    
    #add exp term
    #tmp = []
    #for exp in range(2, 3):
    #    for i in range(18):
    #        tmp.append(list(np.array(data[i])**exp))
    tmp = []
    tmp.append(list(np.array(data[0])**2))    #AMB_TEMP
    tmp.append(list(np.array(data[7])**2))    #O3
    tmp.append(list(np.array(data[8])**2))    #PM10
    tmp.append(list(np.array(data[9])**2))    #PM2.5
    tmp.append(list(np.array(data[10])**2))    #RAINFALL
    tmp.append(list(np.array(data[14])**2))    #WD_HR
    tmp.append(list(np.array(data[15])**2))    #WIND_DIREC
    tmp.append(list(np.array(data[16])**2))    #WIND_SPEED
    tmp.append(list(np.array(data[17])**2))    #WS_HR
    
    #print(np.array(tmp[2]))
    data = list(np.concatenate((np.array(data),np.array(tmp)), axis=0))
    #print(data[20])
    
    txt_data.close()
    
    #make training data array x: input, y:output
    x = []
    y = []
    for month in range(12):
        for hour in range(471):
            flag = 1
            x.append([])
            y.append([])
            idx = len(x) - 1
            for num_category in range(18+18*0+9): # category
                for i in range(9):
                    x[idx].append(data[num_category][month*480+hour+i])
                    if data[9][month*480+hour+i] > 200 or data[9][month*480+hour+9] > 200:
                        del x[idx]
                        del y[idx]
                        flag = 0
                        break
                if flag == 0:
                    break
            if flag == 0:
                continue
            y[idx].append(data[9][month*480+hour+9])
    x = np.array(x)
    y = np.array(y)
    #print(x[0])
    #print(y[2])

    return x, y


# In[6]:


def normalize(x):
    mean = np.mean(x, 0)
    std = np.std(x, 0) + 1e-20
    return (x-mean)/std, mean, std


# In[7]:


def shuffle(x, y):
    np.random.seed(11)
    random_idx = np.arange(x.shape[0])
    np.random.shuffle(random_idx)
    x = x[random_idx,:]
    y = y[random_idx]    
    return x, y


# In[8]:


def splitdata(x,y,size):
    x_r,y_r = shuffle(x,y)
    x_train, y_train = x_r[:size,:], y_r[:size]
    x_val, y_val = x_r[size:,:], y_r[size:]
    return x_train, y_train, x_val, y_val


# In[9]:


def main(argv):
    x, y = readData()
    #X, Y = readData()

    x, mean, std = normalize(x)

    #print(len(x)) 5624
    x_train, y_train, x_val, y_val = splitdata(x,y, len(x) - len(x)//3)

    x_train = x
    y_train = y

    w = np.ones((len(x_train[0]),1))
    b = 1
    learning_rate = 1
    landa = 0
    iteration = 200000
    x_t = x_train.transpose()

    sum_gra_w = np.zeros((len(x_train[0]),1))
    sum_gra_b = 0


    for i in range(iteration):
        loss = y_train - b - np.dot(x_train, w) # + landa * np.sum(w**2)  #add regularization 
    
        gra_w = -2*np.dot(x_t, loss) + 2 * landa * w
        gra_b = -2*np.sum(loss)
        sum_gra_w += gra_w**2
        sum_gra_b += gra_b**2
        ada_w = np.sqrt(sum_gra_w)
        ada_b = np.sqrt(sum_gra_b)
    
        w = w - learning_rate * gra_w/ada_w 
        b = b - learning_rate * gra_b/ada_b 
    
        #kaggle score evaluation
        cost = np.sqrt(np.mean(loss**2))
        if(i%1000 == 0):
            print('iteration: %d,  cost:%f' %(i, cost))
        
    #kaggle score evaluation val
    loss_val = y_val - b - np.dot(x_val, w)
    cost_val = np.sqrt(np.mean(loss_val**2))
    print('cal,  cost:%f' %(cost_val))
    
    #save/read model
    #save model
    np.save('model_w.npy', w)
    np.save('model_b.npy', b)
    np.save('mean.npy', mean)
    np.save('std.npy', std)


# In[ ]:


if __name__ == '__main__':
    main(sys.argv)

