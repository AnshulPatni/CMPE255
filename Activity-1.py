
# coding: utf-8

# In[1]:


# The usual preamble
import pandas as pd
# Open graphs in new cells in the page rather than in a separate window
get_ipython().magic(u'matplotlib inline')
# Make the graphs a bit prettier, and bigger
pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60) 


# We're going to use a new dataset here, to demonstrate how to deal with larger datasets. This is a subset of the of 311 service requests from [NYC Open Data](https://nycopendata.socrata.com/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9). 

# In[2]:


complaints = pd.read_csv('data/311-requests.csv')


# # 1.1 What's even in it? (the summary)

# When you look at a large dataframe, instead of showing you the contents of the dataframe, it'll show you a *summary*. This includes all the columns, and how many non-null values there are in each column.

# In[3]:


complaints


# # 1.2 Selecting columns and rows

# To select a column, we index with the name of the column, like this:

# In[4]:


complaints['Complaint Type']


# To get the first 5 rows of a dataframe, we can use a slice: `df[:5]`.
# 
# This is a great way to get a sense for what kind of information is in the dataframe -- take a minute to look at the contents and get a feel for this dataset.

# In[5]:


complaints[:5]


# We can combine these to get the first 5 rows of a column:

# In[6]:


complaints['Complaint Type'][:5]


# and it doesn't matter which direction we do it in:

# In[7]:


complaints[:5]['Complaint Type']


# # 1.3 Selecting multiple columns

# What if we just want to know the complaint type and the borough, but not the rest of the information? Pandas makes it really easy to select a subset of the columns: just index with list of columns you want.

# In[8]:


complaints[['Complaint Type', 'Borough']]


# That showed us a summary, and then we can look at the first 10 rows:

# In[9]:


complaints[['Complaint Type', 'Borough']][:10]


# # 1.4 What's the most common complaint type?

# This is a really easy question to answer! There's a `.value_counts()` method that we can use:

# In[10]:


complaints['Complaint Type'].value_counts()


# If we just wanted the top 10 most common complaints, we can do this:

# In[11]:


complaint_counts = complaints['Complaint Type'].value_counts()
complaint_counts[:10]


# But it gets better! We can plot them!

# In[12]:


complaint_counts[:10].plot(kind='bar')

