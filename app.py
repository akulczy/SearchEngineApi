# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 20:46:07 2021

@author: group77
"""


import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import numpy as np

app = Flask(__name__)
api = Api(app)

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

import bm25
import vsm

BM25_model = bm25.BM25()
VSM_model = vsm.VSM()

@app.route('/', methods=['GET'])
def home():
   return "<h1>An API created for the Search Engine Project - Group 77.</h1>"

class BM25(Resource):
    @staticmethod
    def post():
        
        # Retrieve query and the no of documents to retrieve from the request
        data = request.get_json()
        queryVal = data['queryVal']
        docNo = int(data['docNo'])
        docs_to_return, ranked_docs_ids = BM25_model.return_results_bm25(queryVal, docNo) 
        
            
        return jsonify(
            results = docs_to_return,
            ranked_docs_ids = ranked_docs_ids
        )
    
    
class VSM(Resource):
    @staticmethod
    def post():
        
        # Retrieve query and the no of documents to retrieve from the request
        data = request.get_json()
        queryVal = data['queryVal']
        docNo = int(data['docNo'])
        docs_to_return, ranked_docs_ids = VSM_model.return_results_vsm(queryVal, docNo) 
        
            
        return jsonify(
            results = docs_to_return,
            ranked_docs_ids = ranked_docs_ids
        )
    
    
class BM25_Relevance_Feedback(Resource):
    @staticmethod
    def post():
        
        # Retrieve query and the no of documents to retrieve from the request
        data = request.get_json()
        queryVal = data['queryVal']
        docNo = int(data['docNo'])
        relevant_docs = data['relevanceList']
        docs_to_return, ranked_docs_ids = BM25_model.return_results_bm25_feedback(queryVal, docNo, relevant_docs) 
        
            
        return jsonify(
            results = docs_to_return,
            ranked_docs_ids = ranked_docs_ids
        )
    
class VSM_Relevance_Feedback(Resource):
    @staticmethod
    def post():
        
        # Retrieve query and the no of documents to retrieve from the request
        data = request.get_json()
        queryVal = data['queryVal']
        docNo = int(data['docNo'])
        relevant_docs = data['relevanceList']
        
        docs_to_return, ranked_docs_ids = VSM_model.return_results_vsm_feedback(queryVal, docNo, relevant_docs) 
        
            
        return jsonify(
            results = docs_to_return,
            ranked_docs_ids = ranked_docs_ids
        )
    
api.add_resource(BM25, '/bm25')
api.add_resource(VSM, '/vsm')
api.add_resource(BM25_Relevance_Feedback, '/bm25/feedback')
api.add_resource(VSM_Relevance_Feedback, '/vsm/feedback')

if __name__ == '__main__':
    app.run(host="localhost", port=5000, debug=True)