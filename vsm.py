# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 09:53:25 2022

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
import logging

from relevance_feedback_vsm import *
from process_data import preprocess, preprocess_query

# Implementation of the Vector Space IR Model

class VSM:
    def __init__(self):
        self.VSM_scores = []
        self.vocabulary = []
        self.documents = []
        self.scores_list = []
        self.original_docs = []
        self.dataframe = []
    
    # Dataset Loader
    # Open and preprocess the datasets
    def return_docs_queries(self):
        text = open("cran.all.1400", "r")
        text = text.read()
        text = text.split('.I')
        text = [sent.replace('\n',' ') for sent in text]
        text = text[1:]
        documents, original_documents = preprocess(text)
        
        qtext = open("cran.qry", "r")
        qtext = qtext.read()
        qtext = qtext.split('.I')
        qtext = [sent.replace('\n',' ') for sent in qtext]
        qtext = qtext[1:]
        queries, original_queries = preprocess(qtext)
        
        self.documents = documents
        self.original_docs = original_documents
        
        return documents, queries, original_documents
    
    # Method accepts dataframe, query, documents and vocalubary lists
    # And calculates RSV scores
    def calculate_vsm_score(self, dataframe, documents, vocabulary, query):
        #DFs
        dfs = (dataframe > 0).sum(axis=0)
        # IDFs
        N = dataframe.shape[0]
        idfs = np.log10(N/dfs)
        d = dataframe*idfs
        
        # Tokenize query
        query = preprocess_query(query)
        query = word_tokenize(query)
        q = []
        for token in vocabulary:
            if token in query:
                q.append(1)
            else:
                q.append(0)
        
        q = pd.DataFrame(q)
        q = q.T
        
        numerator = []
        for row in d.values:
            numerator.append(q.values*row)
        
        for row in range(len(numerator)):
            numerator[row] = numerator[row].sum()
        numerator = pd.DataFrame(numerator)
        
        denominator = []
        for row in d.values:
            denominator.append(row**2)
        
        for row in range(len(denominator)):
            denominator[row] = math.sqrt(denominator[row].sum())*math.sqrt(q.values.sum())
        
        denominator = pd.DataFrame(denominator)
        
        #Calculate RSV
        RSV = numerator/denominator
        RSV = RSV.replace(np.nan, 0)
        
        self.VSM_scores = RSV
        
        return RSV
    
    
    # Rank documents and returned IDs of the top x ones
    def rank_documents(self, documents, RSV, doc2idx):
        ranked = sorted(zip(documents,RSV.values), key = lambda tup:tup[1], reverse=True)
    
        ranked_ids = [doc2idx[ranked[doc][0]] for doc in range(len(documents))]
        ranked_ids[:20]
        return ranked_ids
    
    
    # Utilise methods above to return the results
    def return_results_vsm(self, query, docNo):
        documents, queries, original_documents = self.return_docs_queries()
        queries = [query]
        
        doc2idx = {}
        idx = 1
        # Create document to index dictionary
        for doc in documents:
            doc2idx[doc] = idx
            idx += 1   
            
        # Vectorise
        vectorizer = CountVectorizer(stop_words='english')
        documents_vectorized = vectorizer.fit_transform(documents)
        # Construct vocabulary
        vocabulary = vectorizer.get_feature_names_out()
        self.vocabulary = vocabulary
        # Store in dataframe
        dataframe = pd.DataFrame(documents_vectorized.toarray(), columns=vocabulary)
        self.dataframe = dataframe
        # Calculate RSV
        RSV = self.calculate_vsm_score(dataframe, documents, vocabulary, query)
        # Get X highest ranked documents
        ranked_docs_ids = self.rank_documents(documents, RSV, doc2idx)
        ranked_docs_ids = ranked_docs_ids[:docNo]
        # Get documents to be returned
        docs_to_return = []
        for doc_id in ranked_docs_ids:
            docs_to_return.append(original_documents[doc_id-1])
        return docs_to_return, ranked_docs_ids
    
       
    # Rank documents and return IDs of the top ranked ones
    # Method adjusted for relevance feedback     
    def rank_documents_feedback(self, q_next, documents, vsm_scores, doc2idx):
        q = q_next.T
        
        dfs = (self.dataframe > 0).sum(axis=0)
        N = self.dataframe.shape[0]
        idfs = np.log10(N/dfs)
        
        d = self.dataframe*idfs
    
        predictions = []
        scores = []

        numerator = []
        for row in d.values:
            numerator.append(q.values*row)
    
        for row in range(len(numerator)):
            numerator[row] = numerator[row].sum()
        numerator = pd.DataFrame(numerator)
    
        denominator = []
        for row in d.values:
            denominator.append(row**2)
    
        for row in range(len(denominator)):
            denominator[row] = math.sqrt(denominator[row].sum())*math.sqrt(q.values.sum())
    
        denominator = pd.DataFrame(denominator)
    
        RSV = numerator/denominator
        RSV = RSV.replace(np.nan, 0)
    
        ranked = sorted(zip(documents,RSV.values), key = lambda tup:tup[1], reverse=True)
        ranked_ids = [doc2idx[ranked[doc][0]] for doc in range(len(documents))]
        return ranked_ids, RSV
    
    # Return new list of ranked documents, based on relevance feedback
    def return_results_vsm_feedback(self, query, docNo, relevant_docs):
        VSM_score = self.VSM_scores
        
        doc2idx = {}
        idx = 1
        # Create document to index dictionary
        for doc in self.documents:
            doc2idx[doc] = idx
            idx += 1
                        
        # Get the q next
        q_next = relevance_feedback_vsm(VSM_score, self.vocabulary, relevant_docs, docNo, query)
        
        # Get x highest ranked documents
        ranked_docs_ids = self.rank_documents_feedback(q_next, self.documents, VSM_score, doc2idx)[0]
        ranked_docs_ids = ranked_docs_ids[:docNo]

        # Get documents to be returned
        docs_to_return = []
        for doc_id in ranked_docs_ids:
            docs_to_return.append(self.original_docs[doc_id-1])
        return docs_to_return, ranked_docs_ids

