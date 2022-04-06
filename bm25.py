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

# Perform preprocessing of the datasets
def preprocess(data):
    preprocessed_data = []
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
        punctuation_list = set(string.punctuation)
        new_sent = " ".join("".join([" " if ch in punctuation_list else ch for ch in new_sent]).split())
        # Tokenize
        tokens = word_tokenize(new_sent)
        # Remove stop words
        stop_words = stopwords.words('english')
        tokens = [word for word in tokens if word not in stop_words]
        # Apply stemming
        ps = PorterStemmer()
        tokens = [ps.stem(i) for i in tokens]
        # Return pre-processed data
        document_text = ' '.join(tokens)
        preprocessed_data.append(document_text)
    
    return preprocessed_data, original_data


def return_docs_queries():
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
    
    return documents, queries, original_documents

def calculate_bm25_score(dataframe, documents, vocabulary):
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
    
    bm25_idf = pd.DataFrame(BM25_score, columns=vocabulary)
    
    return bm25_idf
    
    
def rank_documents(queries, documents, bm25_idf, doc2idx):
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
    ranked_ids[:20]
    return ranked_ids


def return_results_bm25(query):
    documents, queries, original_documents = return_docs_queries()
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
    # Store in dataframe
    dataframe = pd.DataFrame(documents_vectorized.toarray(), columns=vocabulary)
    # Calculate BM25 score
    bm25_idf = calculate_bm25_score(dataframe, documents, vocabulary)
    # Get x highest ranked documents
    ranked_docs_ids = rank_documents(queries, documents, bm25_idf, doc2idx)
    ranked_docs_ids = ranked_docs_ids[:20]
    # Get documents to be returned
    docs_to_return = []
    for doc_id in ranked_docs_ids:
        docs_to_return.append(original_documents[doc_id-1])
    return docs_to_return, ranked_docs_ids
    



#gold = open("cranqrel", "r")
#gold = gold.read()
#gold = gold.split('\n')

