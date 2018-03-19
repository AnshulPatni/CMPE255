
# coding: utf-8

# # NumPy

# NumPy is the fundamental package for scientific computing with Python. It contains among other things:
# 
# - a powerful N-dimensional array object
# - sophisticated (broadcasting) functions
# - tools for integrating C/C++ and Fortran code
# - useful linear algebra, Fourier transform, and random number capabilities
# 
# Besides its obvious scientific uses, NumPy can also be used as an efficient multi-dimensional container of generic data. Arbitrary data-types can be defined. This allows NumPy to seamlessly and speedily integrate with a wide variety of databases.
# 
# Library documentation: <a>http://www.numpy.org/</a>
# 
# Reference: <a>https://github.com/jdwittenauer/ipython-notebooks</a>

# In[1]:


from numpy import *


# In[2]:


# declare a vector using a list as the argument
v = array([1,2,3,4])
v


# In[3]:


# declare a matrix using a nested list as the argument
M = array([[1,2],[3,4]])
M


# In[4]:


# still the same core type with different shapes
type(v), type(M)


# In[5]:


M.size


# In[6]:


# arguments: start, stop, step
x = arange(0, 10, 1)
x


# In[7]:


linspace(0, 10, 25)


# In[8]:


logspace(0, 10, 10, base=e)


# In[9]:


x, y = mgrid[0:5, 0:5]
x


# In[10]:


y


# In[11]:


from numpy import random


# In[12]:


random.rand(5,5)


# In[13]:


# normal distribution
random.randn(5,5)


# In[14]:


diag([1,2,3])


# In[15]:


M.itemsize


# In[16]:


M.nbytes


# In[17]:


M.ndim


# In[18]:


v[0], M[1,1]


# In[19]:


M[1]


# In[20]:


# assign new value
M[0,0] = 7
M


# In[21]:


M[0,:] = 0
M


# In[22]:


# slicing works just like with lists
A = array([1,2,3,4,5])
A[1:3]


# In[23]:


A = array([[n+m*10 for n in range(5)] for m in range(5)])
A


# In[24]:


row_indices = [1, 2, 3]
A[row_indices]


# In[25]:


# index masking
B = array([n for n in range(5)])
row_mask = array([True, False, True, False, False])
B[row_mask]


# ### Linear Algebra

# In[26]:


v1 = arange(0, 5)


# In[27]:


v1 + 2


# In[28]:


v1 * 2


# In[29]:


v1 * v1


# In[30]:


dot(v1, v1)


# In[31]:


dot(A, v1)


# In[32]:


# cast changes behavior of + - * etc. to use matrix algebra
M = matrix(A)
M * M


# In[33]:


# inner product
v.T * v


# In[34]:


C = matrix([[1j, 2j], [3j, 4j]])
C


# In[35]:


conjugate(C)


# In[36]:


# inverse
C.I


# ### Statistics

# In[37]:


mean(A[:,3])


# In[38]:


std(A[:,3]), var(A[:,3])


# In[39]:


A[:,3].min(), A[:,3].max()


# In[40]:


d = arange(1, 10)
sum(d), prod(d)


# In[41]:


cumsum(d)


# In[42]:


cumprod(d)


# In[43]:


# sum of diagonal
trace(A)


# In[44]:


m = random.rand(3, 3)
m


# In[45]:


# use axis parameter to specify how function behaves
m.max(), m.max(axis=0)


# In[46]:


A


# In[47]:


# reshape without copying underlying data
n, m = A.shape
B = A.reshape((1,n*m))

B


# In[48]:


# modify the array
B[0,0:5] = 5
B


# In[49]:


# also changed
A


# In[50]:


# creates a copy
B = A.flatten()
B


# In[51]:


# can insert a dimension in an array
v = array([1,2,3])
v[:, newaxis], v[:,newaxis].shape, v[newaxis,:].shape


# In[52]:


repeat(v, 3)


# In[53]:


tile(v, 3)


# In[54]:


w = array([5, 6])


# In[55]:


concatenate((v, w), axis=0)


# In[56]:


# deep copy
B = copy(A)

