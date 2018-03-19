
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("train.dat", sep="\t", header=None)


# In[2]:


import nltk
nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[3]:


stop_words = set(stopwords.words('english'))


# In[4]:


for i in range(0, len(df)):
    word_tokens = word_tokenize(df[1][i])
    sentence_nostop = [w for w in word_tokens if not w in stop_words]
    str = ""
    for i in range(0, len(sentence_nostop)):
        str = str + sentence_nostop[i] + " "
    df[1][i] = str


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer()
counts = vector.fit_transform(df[1])
counts.shape


# In[6]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()


# In[7]:


tfidf_freq = tfidf_transformer.fit_transform(counts)
tfidf_freq.shape


# In[8]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(tfidf_freq, df[0])


# In[9]:


from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('svc', LinearSVC()),
])


# In[10]:


text_clf = text_clf.fit(df[1], df[0])


# In[11]:


import nltk
import csv

test_data = open("test.dat")

predict_class = text_clf.predict(test_data)


# In[12]:


result = pd.DataFrame(predict_class)


# In[13]:


result.to_csv('myPrediction.dat', index=False, header=None)

