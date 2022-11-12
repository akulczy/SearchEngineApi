# Search Engine API
Search Engine developed as part of the coursework for the ECS736P Information Retrieval module at Queen Mary University of London. Two IR models are implemented: BM25 (Okapi BM25) and VSM (Vector Space Model).

## Endpoints
> /bm25

Process the query data and retrieve results of the BM25 model.
> /vsm

Process the query data and retrieve results of the VSM model.
> /bm25/feedback

Process the query data and retrieve results of the BM25 model, along with the relevance feedback results.
> /vsm/feedback

Process the query data and retrieve results of the VSM model, along with the relevance feedback results.

## Dataset
The dataset utilised for the project is Cranfield 1400 from the University of Glasgow Collection, available from http://ir.dcs.gla.ac.uk/resources/test_collections/cran/.
