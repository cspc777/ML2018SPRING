
# coding: utf-8

# In[ ]:


import numpy as np
import sys
from skimage import io


# In[ ]:


img = []
for i in range(415):
    img.append(io.imread(str(sys.argv[1]) + '/' + str(i) +'.jpg').flatten())
img = np.array(img)


# In[ ]:


U, s, V = np.linalg.svd((img - np.mean(img, 0)).T, full_matrices=False)


# In[ ]:


inputimg = io.imread(str(sys.argv[1]) + '/' + str(sys.argv[2])).flatten()
ans = np.mean(img, 0)
for i in range(4):
    ans = ans + np.dot((inputimg-np.mean(img, 0)), U.T[i]) * U.T[i]
ans -= np.min(ans)
ans /= np.max(ans)
ans = (ans*255).astype(np.uint8)
io.imsave('reconstruction.jpg', ans.reshape(600,600,3))

