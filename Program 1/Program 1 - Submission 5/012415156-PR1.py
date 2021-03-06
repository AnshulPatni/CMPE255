
# coding: utf-8

# In[1]:


import pandas as pd
train_data = pd.read_csv("train.dat", sep="\t", header=None)


# In[2]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[3]:


s_words = set(stopwords.words('english'))
print(len(train_data))


# In[4]:


for i in range(0, len(train_data)):
    word_tokens = word_tokenize(train_data[1][i])
    sentence_nostop = [w for w in word_tokens if not w in s_words]
    str = ""
    for i in range(0, len(sentence_nostop)):
        if(i < len(sentence_nostop) - 1):
            str = str + sentence_nostop[i] + " "
        else:
            str = str + sentence_nostop[i]
    train_data[1][i] = str


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer(ngram_range=(1,4))
counts = vector.fit_transform(train_data[1])   #df[1] contains values without the class values 
counts.shape


# In[8]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()


# In[9]:


tfidf_freq = tfidf_transformer.fit_transform(counts)
tfidf_freq.shape


# In[10]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(tfidf_freq, train_data[0])


# In[12]:


from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
])


# In[13]:


text_clf = text_clf.fit(train_data[1], train_data[0])


# In[14]:


import csv
test_data = open("test.dat")

predict_class = text_clf.predict(test_data)


# In[15]:


result = pd.DataFrame(predict_class)
result.to_csv('myPrediction3.dat', index=False, header=None)

