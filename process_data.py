# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:05:39 2022

@author: group77
"""

import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

# Performing preprocessing of the query (Query Processing)
# As well as the dataset (including tokenization, stop words removal, stemming)

# Methods below are utilised in the BM25 and VSM classes

# Perform preprocessing of the datasets
def preprocess(data):
    preprocessed_data = []
    ps = PorterStemmer()
    stop_words = stopwords.words('english')
    punctuation_list = set(string.punctuation)
    # Original data is stored to return it to end-user
    original_data = []
    for sent in data:
        # Split on .W symbol to split the documents
        sent = sent.split(".W")
        sent = sent[1]
        original_data.append(sent)
        # Convert to lower case
        new_sent = sent.lower()
        # Remove digits
        new_sent = re.sub(r'\d+','', new_sent)
        # Remove punctuation
        new_sent = " ".join("".join([" " if ch in punctuation_list else ch for ch in new_sent]).split())
        # Tokenize
        tokens = word_tokenize(new_sent)
        # Remove stop words            
        tokens = [word for word in tokens if word not in stop_words]
        # Apply stemming            
        tokens = [ps.stem(i) for i in tokens]
        # Return pre-processed data
        document_text = ' '.join(tokens)
        preprocessed_data.append(document_text)
    
    return preprocessed_data, original_data

# Perform pre-processing of the query
def preprocess_query(query):        
    # Turn to lower case
    query = query.lower()
    # Remove digits
    query = re.sub(r'\d+','', query)
    # Remove punctuation
    punctuation_list = set(string.punctuation)
    query = " ".join("".join([" " if ch in punctuation_list else ch for ch in query]).split())
    # Tokenize
    tokens = word_tokenize(query)
    stop_words = stopwords.words('english')
    tokens = [word for word in tokens if word not in stop_words]
    # Apply stemming
    ps = PorterStemmer()
    tokens = [ps.stem(i) for i in tokens]
    # Return pre-processed data
    query_txt = ' '.join(tokens)
    return query_txt      
    

