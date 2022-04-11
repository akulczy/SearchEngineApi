# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 19:17:51 2022

@author: group77
"""

import nltk
import re
import math
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Method to construct vector of the previous query
def construct_query_vector(query, vocabulary):
    q = []
    # Construct query vector based on occurrence of tokens
    for token in vocabulary:
        if token in query:
            q.append(1)
        else:
            q.append(0)

    q = pd.DataFrame(q)
    q = q.T
    # Return query vector
    return q


# Method to get relevant and non-relevant documents
def get_rel_nonrel_docs(docNo, vsm_scores, relevant_docs):
    relevant_list = []
    nonrelevant_list = []
   

    for i in range(docNo):
        if relevant_docs[i] == 1 or relevant_docs[i] == "1":
            relevant_list.append(np.array(list(vsm_scores.values)[i]))
        else:
            nonrelevant_list.append(np.array(list(vsm_scores.values)[i]))
    return relevant_list, nonrelevant_list


# Apply Rocchio algorithm for query expansion
def relevance_feedback_vsm(vsm_scores, vocabulary, relevant_docs, docNo, query):
    # Parameters for the algorithm
    alpha = 1
    beta = 0.75
    gamma = 0.25    

    # All BM25 predictions
    q_vector = construct_query_vector(query, vocabulary)
    #q_vector = q_vector
    # Get list of relevant and irrelevant documents
    relevant_list, nonrelevant_list = get_rel_nonrel_docs(docNo, vsm_scores, relevant_docs)
    
    # Apply Rocchio algorithm to calculate new query vector (q_next)
    term1 = alpha * q_vector
    
    # Avoid division by zero
    relevant_list_len = len(relevant_list)
    nonrelevant_list_len = len(nonrelevant_list)
    
    if relevant_list_len == 0:
        relevant_list_len = 1
    if nonrelevant_list_len == 0:
        nonrelevant_list_len = 1    
    
    term2 = beta * (1/relevant_list_len) * np.sum(relevant_list, axis=0) - gamma * (1/nonrelevant_list_len) * np.sum(nonrelevant_list, axis=0)
    
    q_next = term1.T + term2  
    
    # Return q_next
    # New predictions are output based on q_next within the VSM class
    return q_next

