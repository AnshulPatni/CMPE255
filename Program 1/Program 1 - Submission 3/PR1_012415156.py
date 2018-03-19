
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("train.dat", sep="\t", header=None)


# In[2]:


from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer()
counts = vector.fit_transform(df[1])
counts.shape


# In[3]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()


# In[4]:


tfidf_freq = tfidf_transformer.fit_transform(counts)
tfidf_freq.shape


# In[5]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(tfidf_freq, df[0])


# In[6]:


from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('svc', LinearSVC()),
])


# In[7]:


text_clf = text_clf.fit(df[1], df[0])


# In[8]:


import nltk
import csv

test_data = open("test.dat")

predict_class = text_clf.predict(test_data)


# In[10]:


result = pd.DataFrame(predict_class)
result.to_csv('myPrediction1.dat', index=False, header=None)

