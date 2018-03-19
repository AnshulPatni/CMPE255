
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("train.dat", sep="\t", header=None)
print df


# In[2]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df[1])
X_train_counts.shape


# In[3]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[4]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, df[0])


# In[5]:


from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])


# In[8]:


text_clf = text_clf.fit(df[1], df[0])


# In[11]:


import nltk
import csv

t = open("test.dat")

predicted = text_clf.predict(t)


# In[12]:


result = pd.DataFrame(predicted)


# In[13]:


result.to_csv('result.dat', index=False, header=None)

