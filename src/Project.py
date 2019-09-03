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
df['cleaned'] = [ nltk.word_tokenize( str(sentence) ) for sentence in infos ]
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
safe_str = safe_str.translate(remove_digits)

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
safe_df = pd.DataFrame.from_dict(safe_dct, orient='index')
safe_df.columns=['count']
safe_df['zscore'] = stats.zscore(safe_df['count'])

spam_df = pd.DataFrame.from_dict(spam_dct, orient='index')
spam_df.columns=['count']
spam_df['zscore'] = stats.zscore(spam_df['count'])

#%%
"""
Normalization of data with ZScore
spam_df>3: email,submit,form,name,password,student - <-3 empty df
safe_df>3: submit,name,email - <-3 empty df
"""
spam_df.drop(spam_df[ spam_df['zscore'] >3 ].index , inplace=True)
safe_df.drop(safe_df[ safe_df['zscore'] >3 ].index , inplace=True)
#%%


