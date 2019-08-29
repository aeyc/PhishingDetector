#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:31:04 2019

@author: Ayca Begum Tascioglu

Data Seperation
"""
import pandas as pd
#import csv

#with open('query_result.csv') as fd:
#    cr = csv.reader(fd, escapechar='/')
#    for idx,i in enumerate(cr):
#        if idx==20:
#            break
#        print(i)


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

infos = infos.str.replace('[^A-z ]','').str.replace(' +',' ').str.strip()
infos = infos.str.replace('^',' ').str.replace(' +',' ').str.strip()

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

table = pd.Series(spam_forms['cleaned']).apply(pd.value_counts).fillna(0).astype(int)
#table.plot()
#plt.show()

#%%
"""
Most used Words in Spam Forms

Reference: GeeksForGeeks
"""
from collections import Counter

spam_str = str(spam_forms['cleaned'])
split_it = spam_str.split() 
Counter = Counter(split_it) 
  
most_occur_spam = Counter.most_common(10) #k = 10
#print(most_occur_spam)
#%%
"""
Most used Words in Safe Forms

Reference: GeeksForGeeks
"""
from collections import Counter
safe_str = str(safe_forms['cleaned'])

split_safe = safe_str.split() 
Counter5 = Counter(split_safe) 
  
most_occur_safe = Counter5.most_common(15) #k = 10
#split_it_safe = safe_str.split() 
#Counter1 = Counter(safe_str.split()) 
  
#most_occur_safe = (Counter(safe_str.split())).most_common(10) #k = 10
#print(most_occur_safe)