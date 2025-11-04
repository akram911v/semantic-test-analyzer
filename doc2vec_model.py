import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import TruncatedSVD 
from sklearn.metrics.pairwise import cosine_similarity 
 
class Doc2VecModel: 
    def __init__(self, vector_size=100): 
        self.vector_size = vector_size 
        self.document_vectors = {} 
        self.vectorizer = TfidfVectorizer(max_features=1000) 
        self.svd = TruncatedSVD(n_components=vector_size) 
 
    def train(self, tokenized_docs, documents): 
        doc_strings = [' '.join(tokens) for tokens in tokenized_docs] 
        tfidf_matrix = self.vectorizer.fit_transform(doc_strings) 
        doc_vectors = self.svd.fit_transform(tfidf_matrix) 
        for doc_id in range(len(documents)): 
            self.document_vectors[doc_id] = doc_vectors[doc_id] 
 
    def get_document_similarity(self, doc1_id, doc2_id): 
        vec1 = self.document_vectors.get(doc1_id) 
        vec2 = self.document_vectors.get(doc2_id) 
        if vec1 is not None and vec2 is not None: 
            return cosine_similarity([vec1], [vec2])[0][0] 
        return 0.0 
 
    def find_similar_documents(self, query_tokens, documents, top_n=5): 
        query_string = ' '.join(query_tokens) 
        query_tfidf = self.vectorizer.transform([query_string]) 
        query_vector = self.svd.transform(query_tfidf)[0] 
        similarities = [] 
        for doc_id, doc_vector in self.document_vectors.items(): 
            sim = cosine_similarity([query_vector], [doc_vector])[0][0] 
            similarities.append((doc_id, sim, documents[doc_id])) 
        similarities.sort(key=lambda x: x[1], reverse=True) 
        return similarities[:top_n] 
