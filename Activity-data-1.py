
# coding: utf-8

# ### Check out some data ###
# 
# In this activity we will check out ipython install and the availability of some important modules. Furthermore, we'll visualize a dataset, learning a bit about the power of Pandas.
# 
# <sub>Note: Activity adapted from [nbviewer](http://nbviewer.jupyter.org/github/jvns/pandas-cookbook/blob/v0.1/cookbook/Chapter%205%20-%20Combining%20dataframes%20and%20scraping%20Canadian%20weather%20data.ipynb). Montréal cycling data from [Données Ouvertes Montréal](http://donnees.ville.montreal.qc.ca/dataset/velos-comptage)</sub>

# First, let's make sure you have the right modules installed to run activities for this class.

# In[1]:


import pandas as pd
import numpy as np
import scipy as sp
# Open graphs in new cells in the page rather than in a separate window
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = (15, 5)  # set a default figure size


# You can read data from a CSV file using the `read_csv` function. By default, it assumes that the fields are comma-separated.
# 
# We're going to be looking some cyclist data from Montréal. Here's the [original page](http://donnees.ville.montreal.qc.ca/dataset/velos-comptage) (in French), but it's already included in this repository. We're using the data from 2012.
# 
# This dataset is a list of how many people were on 7 different bike paths in Montreal, each day.

# In[4]:


broken_df = pd.read_csv('data/bikes.csv')


# In[5]:


# Look at the first 3 rows
broken_df[:3]


# You'll notice that this is totally broken! `read_csv` has a bunch of options that will let us fix that, though. Here we'll
# 
# * change the column separator to a `;`
# * Set the encoding to `'latin1'` (the default is `'utf8'`)
# * Parse the dates in the 'Date' column
# * Tell it that our dates have the date first instead of the month first
# * Set the index to be the 'Date' column

# In[6]:


fixed_df = pd.read_csv('data/bikes.csv', sep=';', encoding='latin1', parse_dates=['Date'], dayfirst=True, index_col='Date')
fixed_df[:3]


# When you read a CSV, you get a kind of object called a `DataFrame`, which is made up of rows and columns. You get columns out of a DataFrame the same way you get elements out of a dictionary.
# 
# Here's an example:

# In[7]:


fixed_df['Berri 1']


# Just add `.plot()` to the end to plot! How could it be easier? =)
# 
# We can see that, unsurprisingly, not many people are biking in January, February, and March, 

# In[8]:


fixed_df['Berri 1'].plot()


# We can also plot all the columns just as easily. We'll make it a little bigger, too.
# You can see that it's more squished together, but all the bike paths behave basically the same -- if it's a bad day for cyclists, it's a bad day everywhere.

# In[9]:


fixed_df.plot(figsize=(15, 10))

