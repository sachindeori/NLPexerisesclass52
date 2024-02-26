#!/usr/bin/env python
# coding: utf-8

# In[19]:


from sklearn.datasets import fetch_20newsgroups


# In[20]:


twenty_train = fetch_20newsgroups(subset='train', shuffle=True)


# In[21]:


twenty_train.target_names #prints all the categories


# In[22]:


print("\n".join(twenty_train.data[0].split("\n")[:3])) #prints firstline of the first data file


# In[23]:


from sklearn.feature_extraction.text import CountVectorizer


# In[24]:


count_vect = CountVectorizer()


# In[25]:


X_train_counts = count_vect.fit_transform(twenty_train.data)


# In[26]:


X_train_counts.shape


# In[27]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[28]:


tfidf_transformer = TfidfTransformer()


# In[29]:


X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# In[30]:


X_train_tfidf.shape


# In[31]:


from sklearn.naive_bayes import MultinomialNB


# In[32]:


clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)


# In[33]:


from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', MultinomialNB()),
])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)


# In[34]:


import numpy as np
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
np.mean(predicted == twenty_test.target)


# In[2]:


a = ['circket','football','hockey']
a.append('swiming')
print(a)


# In[ ]:




