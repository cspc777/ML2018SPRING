
# coding: utf-8

# In[1]:


import csv
import numpy as np
import sys


# In[2]:


def readTestData():
    input_file = str(sys.argv[1])
    data = []
    for i in range(18):
        data.append([])
    
    txt_data = open(input_file, 'r')
    row = csv.reader(txt_data, delimiter=',')
    num_category = 0
    for r in row:
        for c in range(2, 11):
            if r[c] == 'NR':
                data[(num_category)%18].append(float(0))
            else:
                data[(num_category)%18].append(float(r[c]))
        num_category = num_category + 1

    for i in range(len(data[9])):
        if(data[9][i] < 0):
            data[9][i] = 0

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
    
    data = list(np.concatenate((np.array(data),np.array(tmp)), axis=0))
    
    txt_data.close()
    
    #make test data array x
    test_x = []
    idx_r = 0
    idx_c = 0
    while(idx_c < len(data[0])):
        test_x.append([])
        for r in range(len(data)):
            for c in range(9):
                test_x[idx_r].append(data[r][c+idx_c])
        idx_r = idx_r + 1
        idx_c = idx_c + 9
    #print(len(test_x[0]))
    test_x = np.array(test_x)
    return test_x


# In[3]:


def main(argv):
    output_file = str(sys.argv[2])
    #read model
    w = np.load('model_w.npy')
    b = np.load('model_b.npy')
    mean = np.load('mean.npy')
    std = np.load('std.npy')
    
    test = readTestData()
    test = (test-mean)/std

    #get ans.csv
    ans = []
    result = np.dot(test, w) + b
    for i in range(len(test)):
        ans.append(['id_'+str(i)])
        ans[i].append(result[i][0])

    filename = output_file
    text = open(filename, "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(['id','value'])
    for i in range(len(ans)):
        s.writerow(ans[i]) 
    text.close()


# In[ ]:


if __name__ == '__main__':
    main(sys.argv)

