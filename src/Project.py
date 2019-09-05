#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:31:04 2019

@author: Ayca Begum Tascioglu

Data Seperation
"""
import pandas as pd

df = pd.read_csv("query_result.csv",escapechar = '\\')
formIDs = df.form_id
titles = df.title
targets = df.target
infos = df['info']

#%%
"""
Tokenize

Reference: StackOverFlow
"""
import nltk
import re

infos = infos.str.replace('[^A-z ]','').str.replace(' +',' ').str.strip()
infos = infos.str.replace('^',' ').str.replace(' +',' ').str.strip()
infos = infos.str.replace('[',' ').str.replace(' +',' ').str.strip()
infos = infos.str.replace('[^\w\s]',' ').str.replace(' +',' ').str.strip()

infos = infos.str.replace(']',' ').str.replace(' +',' ').str.strip()
infos = infos.str.lower()
for i in infos:
    i= re.sub('[\W_]+', '', str(i))
df['sentence'] = infos
df['cleaned'] = [ nltk.word_tokenize( str(sentence) ) for sentence in infos ]
#%%
"""
DROP NAN VALUES
"""
df = df.dropna()
"""
NAN
"""
#%% 
"""
Data analysis

Reference: Towards Data Science
"""
import numpy as np
import matplotlib as plt
spam_forms = df[df.target == 'SPAM']
safe_forms = df[df.target != 'SPAM']

#------> undo comment here <------#
#spam_forms['cleaned'] = spam_forms['cleaned'].str.replace('[^\w\s]','').str.replace(' +',' ').str.strip()
table = pd.Series(spam_forms['cleaned']).apply(pd.value_counts).fillna(0).astype(int)
#table.plot()
#plt.show()

#%%
#replace method str
def punc_rep(s):
    s = s.replace('[^\w\s]',' ').strip()
    s = s.replace('        ','').strip()
    s = s.replace(',','').strip()
    s = s.replace('.','').strip()
    s = s.replace('[','').strip()
    s = s.replace(']','').strip()
    s = s.replace(':','').strip()
    s = s.replace('        ','').strip()
    s = s.lower()
    return s
#%%
"""
Removing digits, and other characters
Creating a dictionary for cleaned words and their occurancies
Spam Froms
"""
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

from string import digits
spam_str = str(spam_forms['cleaned'])
spam_str = punc_rep(spam_str)
remove_digits = str.maketrans('','',digits)
spam_str = spam_str.translate(remove_digits)

split_spam = spam_str.split() 
spam_dct = dict()
tmp = []
for i in split_spam:
    if i not in stopWords:
        tmp.append(i)
split_spam = tmp
for i in split_spam:
    if i in spam_dct:
        spam_dct[i] += 1
    else:
        spam_dct[i] = 1
#%%
"""
Removing digits, and other characters
Creating a dictionary for cleaned words and their occurancies
Safe Froms
"""

safe_str = str(safe_forms['cleaned'])
safe_str = punc_rep(safe_str)
remove_digits = str.maketrans('','',digits)

split_safe = safe_str.split() 
safe_dct = dict()
former = split_safe
tmp = []
for i in split_safe:
    if i not in stopWords:
        tmp.append(i)
split_safe = tmp
for i in split_safe:
    if i in safe_dct:
        safe_dct[i] += 1
    else:
        safe_dct[i] = 1 
#%%
#Z score
"""
Dictionary to Dataframe

"""
from scipy import stats
safe_df = pd.DataFrame(list(safe_dct.items()))
safe_df.columns=['word','count']
safe_df['zscore'] = stats.zscore(safe_df['count'])
safe_df['label'] = 'SAFE'

spam_df = pd.DataFrame(list(spam_dct.items()))
spam_df.columns=['word','count']
spam_df['zscore'] = stats.zscore(spam_df['count'])
spam_df['label'] = 'SPAM'

#%%
"""
Normalization of data with ZScore
spam_df>3: email,submit,form,name,password,student - <-3 empty df
safe_df>3: submit,name,email - <-3 empty df
"""
spam_df.drop(spam_df[ spam_df['zscore'] >3.35  ].index , inplace=True)
safe_df.drop(safe_df[ safe_df['zscore'] >3 ].index , inplace=True)
#%%
"""
Plots
"""
#import matplotlib.pyplot as plt
#plt.scatter(safe_df.word, safe_df.zscore)
#plt.xlabel("word")
#plt.ylabel("zscore")
#plt.show()
#%%
"""
Algortihm trial for tdif
n: total no of words in spam forms
"""
#def tf_idf(i,j):
#   return tf(i,j)*idf(i,j)
#
#def tf(i,j):
#    return i/j
#
#def idf(i,j):
#    return np.log(j/i)
#
#n = sum(spam_df['count'])
#m = sum(safe_df['count'])
#
#spam_df['tf_idf'] = tf_idf(spam_df['count'],n)
#safe_df['tf_idf'] = tf_idf(safe_df['count'],n)

#%%
"""
Vectorizing with Tfidf Vectorizer

"""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import datasets
from sklearn import svm
from sklearn.linear_model import LogisticRegression

new_df = spam_df.append(safe_df)
X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['target'], test_size = 0.1, random_state = 1)
#X_train, X_test, y_train, y_test = train_test_split(new_df['word'].sample(n = 146),spam_df['word'], test_size = 0.1, random_state = 1)

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)

svm = svm.SVC(kernel='linear')
svm.fit(X_train, y_train)

#%%
from sklearn.metrics import confusion_matrix
X_test = vectorizer.transform(X_test)
y_pred = svm.predict(X_test)
print(confusion_matrix(y_test, y_pred))

def pred(msg):
    msg = vectorizer.transform([msg])
    prediction = svm.predict(msg)
    #svm.predict(vectorizer.transform(['password']))
    return prediction[0]
