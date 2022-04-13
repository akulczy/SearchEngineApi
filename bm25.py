# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 19:20:20 2022

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

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from relevance_feedback_bm25 import *
from process_data import preprocess, preprocess_query

# Implementation of the BM25 IR Model

class BM25:
    def __init__(self):
        self.BM25_score = 0
        self.vocabulary = []
        self.documents = []
        self.scores_list = []
        self.original_docs = []
    
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
    
    # Method accepts dataframe, documents and vocalubary lists
    # And calculates BM25 scores
    def calculate_bm25_score(self, dataframe, documents, vocabulary):
        dfs = (dataframe > 0).sum(axis=0)
    
        N = dataframe.shape[0]
        # Calculate IDF
        idfs = np.log10(N/dfs)
        
        k_1 = 1.2 
        b = 0.8 
        
        # Get words from the documents
        dls = [len(d.split(' ')) for d in documents] 
        dls = dataframe.sum(axis=1).tolist()
        avgdl = np.mean(dls) 
        # Calculate numerator and denominator
        numerator = np.array((k_1 + 1) * dataframe)
        denominator = np.array(k_1 *((1 - b) + b * (dls / avgdl))).reshape(N,1) + np.array(dataframe)
        
        BM25_tf = numerator / denominator
        
        idfs = np.array(idfs)
        
        BM25_score = BM25_tf * idfs
        
        self.scores_list = BM25_score
        
        bm25_idf = pd.DataFrame(BM25_score, columns=vocabulary)
        
        return bm25_idf
        
    
    # Rank documents and return IDs of the top ranked ones    
    def rank_documents(self, queries, documents, bm25_idf, doc2idx):
        empty = np.zeros(4177)
        query = word_tokenize(queries[0])
        q_terms = []
        
        for token in query:
            if token in bm25_idf:
                q_term_df = list(bm25_idf[token])
                q_terms.append(q_term_df)
            else:
                q_terms.append(empty)    
        
        q_terms_df = pd.DataFrame(q_terms)
        q_terms_only_df = q_terms_df.T
        
        score_q_d = q_terms_only_df.sum(axis=1)
        ranked = sorted(zip(documents,score_q_d.values), key = lambda tup:tup[1], reverse=True)
        
        ranked_ids = [doc2idx[ranked[doc][0]] for doc in range(len(documents))]
        return ranked_ids
    
    
    # Utilise methods above to obtain scores, rank documents, and return results
    def return_results_bm25(self, query, docNo):
        documents, queries, original_documents = self.return_docs_queries()
        query = preprocess_query(query)
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
        # Calculate BM25 score
        bm25_idf = self.calculate_bm25_score(dataframe, documents, vocabulary)
        self.BM25_score = bm25_idf
        # Get x highest ranked documents
        ranked_docs_ids = self.rank_documents(queries, documents, bm25_idf, doc2idx)
        ranked_docs_ids = ranked_docs_ids[:docNo]
        # Get documents to be returned
        docs_to_return = []
        for doc_id in ranked_docs_ids:
            docs_to_return.append(original_documents[doc_id-1])
        return docs_to_return, ranked_docs_ids
    
    
    # Rank documents and return IDs of the top ranked ones
    # Method adjusted for relevance feedback     
    def rank_documents_feedback(self, q_next, documents, bm25_idf, doc2idx):
        # Q Next calculated by query expansion
        q_terms_only_df = q_next
        # Calculate scores
        score_q_d = q_terms_only_df.sum(axis=1)
        ranked = sorted(zip(documents,score_q_d.values), key = lambda tup:tup[1], reverse=True)
        # Rank documents
        ranked_ids = [doc2idx[ranked[doc][0]] for doc in range(len(documents))]
        return ranked_ids
    
    # Return new list of ranked documents, based on relevance feedback
    def return_results_bm25_feedback(self, query, docNo, relevant_docs):
        BM25_idf = self.BM25_score
        doc2idx = {}
        idx = 1
        # Create document to index dictionary
        for doc in self.documents:
            doc2idx[doc] = idx
            idx += 1
            
        # Get the q next
        q_next = relevance_feedback_bm25(BM25_idf, self.vocabulary, relevant_docs, docNo, query, self.scores_list)
        
        # Get x highest ranked documents
        ranked_docs_ids = self.rank_documents_feedback(q_next, self.documents, BM25_idf, doc2idx)
        ranked_docs_ids = ranked_docs_ids[:docNo]
        # Get documents to be returned
        docs_to_return = []
        for doc_id in ranked_docs_ids:
            docs_to_return.append(self.original_docs[doc_id-1])
        return docs_to_return, ranked_docs_ids


