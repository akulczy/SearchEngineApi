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

# Open and preprocess the datasets
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

# Method accepts dataframe, query, documents and vocalubary lists
# And calculates RSV scores
def calculate_vsm_score(dataframe, documents, vocabulary, query):
    #DFs
    dfs = (dataframe > 0).sum(axis=0)
    # IDFs
    N = dataframe.shape[0]
    idfs = np.log10(N/dfs)
    d = dataframe*idfs
    
    # Tokenize query
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
    return RSV


# Rank documents and returned IDs of the top x ones
def rank_documents(documents, RSV, doc2idx):
    ranked = sorted(zip(documents,RSV.values), key = lambda tup:tup[1], reverse=True)

    ranked_ids = [doc2idx[ranked[doc][0]] for doc in range(len(documents))]
    ranked_ids[:20]
    return ranked_ids


# Utilise methods above to return the results
def return_results_vsm(query):
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
    # Calculate RSV
    RSV = calculate_vsm_score(dataframe, documents, vocabulary, query)
    # Get X highest ranked documents
    ranked_docs_ids = rank_documents(documents, RSV, doc2idx)
    ranked_docs_ids = ranked_docs_ids[:20]
    # Get documents to be returned
    docs_to_return = []
    for doc_id in ranked_docs_ids:
        docs_to_return.append(original_documents[doc_id-1])
    return docs_to_return, ranked_docs_ids

