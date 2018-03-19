
# coding: utf-8

# ### Playing arund with Iris ###
# 
# We will use Iris in class to practice some attribute transformations and computing similarities.
# 

# In[1]:


import matplotlib.pyplot as plt
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, 2:4]  # we only take petal length and petal width.
Y = iris.target

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('Petal length')
plt.ylabel('Petal width')

plt.show()


# In[2]:


import numpy as np
A = iris.data
a = A[0,:]
b = A[-1,:]
print a,b


# In[3]:


c = np.log(a)


# In[4]:


d = np.abs(c)
print d


# In[5]:


for i in xrange(A.shape[1]):
    print np.min(A[:,i]), np.max(A[:,i])


# In[6]:


c = A[:,0]
c_mean = np.mean(c)
c_std = np.std(c)
d = (c-c_mean)/c_std
print c_mean, c_std
print np.min(d), np.max(d), np.mean(d), np.std(d)

