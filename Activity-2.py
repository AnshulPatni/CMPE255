
# coding: utf-8

# In[1]:


# The usual preamble
import pandas as pd
# Open graphs in new cells in the page rather than in a separate window
get_ipython().magic(u'matplotlib inline')
# Always display all the columns
pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60) 


# Let's continue with our NYC 311 service requests example.

# In[3]:


complaints = pd.read_csv('data/311-requests.csv')


# # 2.1 Selecting only noise complaints

# I'd like to know which borough has the most noise complaints. First, we'll take a look at the data to see what it looks like:

# In[4]:


complaints[:5]


# To get the noise complaints, we need to find the rows where the "Complaint Type" column is "Noise - Street/Sidewalk". I'll show you how to do that, and then explain what's going on.

# In[5]:


noise_complaints = complaints[complaints['Complaint Type'] == "Noise - Street/Sidewalk"]
noise_complaints[:3]


# If you look at `noise_complaints`, you'll see that this worked, and it only contains complaints with the right complaint type. But how does this work? Let's deconstruct it into two pieces

# In[6]:


complaints['Complaint Type'] == "Noise - Street/Sidewalk"


# This is a big array of `True`s and `False`s, one for each row in our dataframe. When we index our dataframe with this array, we get just the rows where.
# 
# You can also combine more than one condition with the `&` operator like this:

# In[7]:


is_noise = complaints['Complaint Type'] == "Noise - Street/Sidewalk"
in_brooklyn = complaints['Borough'] == "BROOKLYN"
complaints[is_noise & in_brooklyn][:5]


# Or if we just wanted a few columns:

# In[8]:


complaints[is_noise & in_brooklyn][['Complaint Type', 'Borough', 'Created Date', 'Descriptor']][:10]


# # 2.2 A digression about numpy arrays

# On the inside, the type of a column is `pd.Series`

# In[9]:


pd.Series([1,2,3])


# and pandas Series are internally numpy arrays. If you add `.values` to the end of any `Series`, you'll get its internal numpy array

# In[10]:


import numpy as np
np.array([1,2,3])


# In[11]:


pd.Series([1,2,3]).values


# So this binary-array-selection business is actually something that works with any numpy array:

# In[12]:


arr = np.array([1,2,3])


# In[13]:


arr != 2


# In[14]:


arr[arr != 2]


# # 2.3 So, which borough has the most noise complaints?

# In[15]:


is_noise = complaints['Complaint Type'] == "Noise - Street/Sidewalk"
noise_complaints = complaints[is_noise]
noise_complaints['Borough'].value_counts()


# It's Manhattan! But what if we wanted to divide by the total number of complaints, to make it make a bit more sense? That would be easy too:

# In[16]:


noise_complaint_counts = noise_complaints['Borough'].value_counts()
complaint_counts = complaints['Borough'].value_counts()


# In[17]:


noise_complaint_counts / complaint_counts


# Oops, why was that zero? That's no good. This is because of integer division in Python 2. Let's fix it, by converting `complaint_counts` into an array of floats.

# In[18]:


noise_complaint_counts / complaint_counts.astype(float)


# In[19]:


(noise_complaint_counts / complaint_counts.astype(float)).plot(kind='bar')


# So Manhattan really does complain more about noise than the other boroughs! Neat.

# <style>
#     @font-face {
#         font-family: "Computer Modern";
#         src: url('http://mirrors.ctan.org/fonts/cm-unicode/fonts/otf/cmunss.otf');
#     }
#     div.cell{
#         width:800px;
#         margin-left:16% !important;
#         margin-right:auto;
#     }
#     h1 {
#         font-family: Helvetica, serif;
#     }
#     h4{
#         margin-top:12px;
#         margin-bottom: 3px;
#        }
#     div.text_cell_render{
#         font-family: Computer Modern, "Helvetica Neue", Arial, Helvetica, Geneva, sans-serif;
#         line-height: 145%;
#         font-size: 130%;
#         width:800px;
#         margin-left:auto;
#         margin-right:auto;
#     }
#     .CodeMirror{
#             font-family: "Source Code Pro", source-code-pro,Consolas, monospace;
#     }
#     .text_cell_render h5 {
#         font-weight: 300;
#         font-size: 22pt;
#         color: #4057A1;
#         font-style: italic;
#         margin-bottom: .5em;
#         margin-top: 0.5em;
#         display: block;
#     }
#     
#     .warning{
#         color: rgb( 240, 20, 20 )
#         }  
