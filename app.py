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

from bm25 import return_results_bm25

@app.route('/', methods=['GET'])
def home():
   return "<h1>An API created for the Search Engine Project - Group 77.</h1>"

class BM25(Resource):
    @staticmethod
    def post():
        
        # Retrieve query from the request
        data = request.get_json()
        queryVal = data['queryVal']
        docs_to_return, ranked_docs_ids = return_results_bm25(queryVal) 
        
            
        return jsonify(
            results = docs_to_return,
            ranked_docs_ids = ranked_docs_ids
        )
    
api.add_resource(BM25, '/bm25')
if __name__ == '__main__':
    app.run(host="localhost", port=5000)