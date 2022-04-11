# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:07:32 2022

@author: group 77
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
def construct_query_vector(query, vocabulary, bm25_idf):
    empty = np.zeros(4177) 
    empty2 = np.zeros(2777)
    # Tokenize
    query = word_tokenize(query)
    # Construct vector based on BM25 scores
    q_terms = []
    for token in query:
        if token in bm25_idf:
            q_term_df = np.append(np.array(bm25_idf[token]), empty2)
            q_terms.append(q_term_df)
        else:
            q_terms.append(empty)
    # Return query vector
    return q_terms

# Method to get relevant and non-relevant documents
def get_rel_nonrel_docs(docNo, bm25_scores, relevant_docs, scores_list):
    relevant_list = []
    nonrelevant_list = []

    for i in range(docNo):
        if relevant_docs[i] == 1 or relevant_docs[i] == "1":
            relevant_list.append(np.array(scores_list[i]).reshape(1, 4177))
        else:
            nonrelevant_list.append(np.array(scores_list[i]).reshape(1, 4177))
    return relevant_list, nonrelevant_list

# Apply Rocchio algorithm for query expansion
def relevance_feedback_bm25(BM25_score, vocabulary, relevant_docs, docNo, query, scores_list):
    # Parameters for the algorithm
    alpha = 1
    beta = 0.75
    gamma = 0.25    
   
    bm25_idf = BM25_score
    # Construct previous query vector
    q_vector = construct_query_vector(query, vocabulary, bm25_idf)
    q_terms_df = pd.DataFrame(q_vector)
    q_terms_only_df = q_terms_df.T
    
    relevant_list, nonrelevant_list = get_rel_nonrel_docs(docNo, BM25_score, relevant_docs, scores_list)
    
    # Apply Rocchio algorithm to calculate new query vector (q_next)
    term1 = alpha * q_terms_only_df
    
    # Avoid division by zero
    relevant_list_len = len(relevant_list)
    nonrelevant_list_len = len(nonrelevant_list)
    
    if relevant_list_len == 0:
        relevant_list_len = 1
    if nonrelevant_list_len == 0:
        nonrelevant_list_len = 1    
    
    term2 = beta * (1/relevant_list_len) * np.sum(relevant_list, axis=0) - gamma * (1/nonrelevant_list_len) * np.sum(nonrelevant_list, axis=0)
    
    q_next = term1 + term2.T
    
    # Return q_next
    # New predictions are output based on q_next within the BM25 class
    return q_next