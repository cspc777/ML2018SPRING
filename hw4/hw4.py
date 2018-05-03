
# coding: utf-8

# In[ ]:


import sys
import csv
import numpy as np

from keras.layers import Input, Dense
from keras.models import Model, load_model
from sklearn.cluster import KMeans


# In[ ]:


def readTestData():
    txt_data = open(str(sys.argv[2]), 'r', encoding = 'big5')
    row = csv.reader(txt_data, delimiter=',')
    data = []
    num = 0
    for r in row:
        if num > 0:
            data.append([int(r[0]), int(r[1]), int(r[2])])
        num = num + 1
    data = np.array(data)
    return data


# In[ ]:


img_data = np.load(str(sys.argv[1]))


# In[ ]:


x = img_data/255


# In[ ]:


encoder = load_model('model.h5')

x_reduction = encoder.predict(x)
kmeans = KMeans(n_clusters=2).fit(x_reduction)


# In[ ]:


test = readTestData()


# In[ ]:


ans = []
for i in range(len(test)):
    ans.append([int(test[i][0])])
    if(kmeans.labels_[test[i][1]] == kmeans.labels_[test[i][2]]):
        ans[i].append(int(1))
    else:
        ans[i].append(int(0))

filename = str(sys.argv[3])

text = open(filename, 'w+')
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['ID','Ans'])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()

