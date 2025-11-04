import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
from word2vec_model import Word2VecModel
from doc2vec_model import Doc2VecModel

# Download required NLTK data with error handling
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    print("NLTK data download completed or already exists")

class LSASemanticAnalyzer:
    def __init__(self, n_components=100):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.svd = TruncatedSVD(n_components=n_components)
        self.documents = []
        self.processed_docs = []
        self.tokenized_docs = []
        self.tfidf_matrix = None
        self.lsa_matrix = None
        self.word2vec_model = None
        self.doc2vec_model = None
        
    def preprocess_text(self, text, return_tokens=False):
        """Text preprocessing using techniques from Practical Work 1"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Simple tokenization (fallback if NLTK fails)
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback: simple whitespace tokenization
            tokens = text.split()
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        except:
            # Fallback: common English stopwords
            common_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            tokens = [token for token in tokens if token not in common_stopwords]
        
        # Lemmatization
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        except:
            # Fallback: no lemmatization
            pass
        
        # Remove short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        if return_tokens:
            return tokens
        return ' '.join(tokens)
    
    def fit(self, documents):
        """Train LSA model on documents"""
        self.documents = documents
        
        # Preprocess all documents - store both tokens and strings
        self.tokenized_docs = [self.preprocess_text(doc, return_tokens=True) for doc in documents]
        self.processed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_docs)
        
        # Apply LSA (SVD)
        self.lsa_matrix = self.svd.fit_transform(self.tfidf_matrix)
        
        print(f"LSA model trained with {len(documents)} documents")
        print(f"Original TF-IDF dimensions: {self.tfidf_matrix.shape}")
        print(f"LSA dimensions: {self.lsa_matrix.shape}")
        print(f"Explained variance ratio: {self.svd.explained_variance_ratio_.sum():.4f}")
    
    # Word2Vec methods
    def initialize_word2vec(self, vector_size=100, window=5, min_count=2):
        self.word2vec_model = Word2VecModel(
            vector_size=vector_size,
            window=window,
            min_count=min_count
        )
    
    def train_word2vec(self, vector_size=100, window=5, min_count=2):
        if not hasattr(self, 'word2vec_model') or self.word2vec_model is None:
            self.initialize_word2vec(vector_size, window, min_count)
        self.word2vec_model.train(self.tokenized_docs, self.documents)
        print("Word2Vec model trained successfully")
    
    # Doc2Vec methods - FIXED: removed unused parameters
   def initialize_doc2vec(self, vector_size=100, window=5, min_count=2, epochs=20):
    self.doc2vec_model = Doc2VecModel(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs
    )

def train_doc2vec(self, vector_size=100, window=5, min_count=2, epochs=20):
    if not hasattr(self, 'doc2vec_model') or self.doc2vec_model is None:
        self.initialize_doc2vec(vector_size, window, min_count, epochs)
    self.doc2vec_model.train(self.tokenized_docs, self.documents)
    print("Doc2Vec model trained successfully")
    
    # Existing LSA methods remain the same...
    def document_similarity(self, doc1, doc2):
        """Calculate semantic similarity between two documents"""
        processed_doc1 = self.preprocess_text(doc1)
        processed_doc2 = self.preprocess_text(doc2)
        
        # Transform documents to LSA space
        doc1_tfidf = self.vectorizer.transform([processed_doc1])
        doc2_tfidf = self.vectorizer.transform([processed_doc2])
        
        doc1_lsa = self.svd.transform(doc1_tfidf)
        doc2_lsa = self.svd.transform(doc2_tfidf)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(doc1_lsa, doc2_lsa)[0][0]
        return similarity
    
    def query_similarity(self, query, top_k=3):
        """Find most similar documents to a query"""
        processed_query = self.preprocess_text(query)
        
        # Transform query to LSA space
        query_tfidf = self.vectorizer.transform([processed_query])
        query_lsa = self.svd.transform(query_tfidf)
        
        # Calculate similarities with all documents
        similarities = cosine_similarity(query_lsa, self.lsa_matrix)[0]
        
        # Get top_k most similar documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'similarity': similarities[idx],
                'index': idx
            })
        
        return results
    
    def get_topic_terms(self, n_terms=10):
        """Get most important terms for each topic"""
        terms = self.vectorizer.get_feature_names_out()
        topic_terms = []
        
        for i, topic in enumerate(self.svd.components_):
            top_indices = topic.argsort()[-n_terms:][::-1]
            top_terms = [(terms[idx], topic[idx]) for idx in top_indices]
            topic_terms.append({
                'topic_id': i,
                'terms': top_terms
            })
        
        return topic_terms
    
    def transform_document(self, document):
        """Transform a single document to LSA space"""
        processed_doc = self.preprocess_text(document)
        doc_tfidf = self.vectorizer.transform([processed_doc])
        doc_lsa = self.svd.transform(doc_tfidf)
        return doc_lsa[0]

def main():
    # Sample documents for demonstration
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers for complex pattern recognition.",
        "Natural language processing helps computers understand and process human language.",
        "Computer vision enables machines to interpret and understand visual information.",
        "Data science involves extracting insights from structured and unstructured data.",
        "Python is a popular programming language for data analysis and machine learning.",
        "TensorFlow and PyTorch are widely used frameworks for deep learning projects.",
        "Supervised learning uses labeled datasets to train machine learning models.",
        "Unsupervised learning finds patterns in data without predefined labels.",
        "Reinforcement learning trains models through rewards and punishments system."
    ]
    
    # Initialize and train LSA model
    analyzer = LSASemanticAnalyzer(n_components=5)
    analyzer.fit(documents)
    
    # Train Word2Vec and Doc2Vec models
    analyzer.train_word2vec()
    analyzer.train_doc2vec()  # FIXED: no parameters needed
    
    print("\n" + "="*50)
    print("EXTENDED SEMANTIC ANALYSIS DEMONSTRATION")
    print("="*50)
    
    # Show topic terms from LSA
    print("\nLSA TOPIC TERMS:")
    topic_terms = analyzer.get_topic_terms(n_terms=5)
    for topic in topic_terms:
        print(f"Topic {topic['topic_id']}: {[term[0] for term in topic['terms']]}")
    
    # Test document similarity with LSA
    print("\nLSA DOCUMENT SIMILARITY:")
    doc1 = "Machine learning algorithms learn from data"
    doc2 = "Artificial intelligence systems can make predictions"
    similarity = analyzer.document_similarity(doc1, doc2)
    print(f"Similarity between '{doc1}' and '{doc2}': {similarity:.4f}")
    
    # Test Word2Vec document similarity
    print("\nWORD2VEC DOCUMENT SIMILARITY:")
    if analyzer.word2vec_model:
        tokens1 = analyzer.preprocess_text(doc1, return_tokens=True)
        tokens2 = analyzer.preprocess_text(doc2, return_tokens=True)
        vec1 = analyzer.word2vec_model._get_document_vector(tokens1)
        vec2 = analyzer.word2vec_model._get_document_vector(tokens2)
        w2v_similarity = cosine_similarity([vec1], [vec2])[0][0]
        print(f"Word2Vec similarity: {w2v_similarity:.4f}")
    
    # Test query similarity with all three methods
    print("\nQUERY SIMILARITY SEARCH:")
    query = "neural networks and deep learning"
    
    # LSA results
    lsa_results = analyzer.query_similarity(query, top_k=3)
    print(f"\nLSA Results for '{query}':")
    for i, result in enumerate(lsa_results):
        print(f"{i+1}. Similarity: {result['similarity']:.4f}")
        print(f"   Document: {result['document']}")
    
    # Word2Vec results
    if analyzer.word2vec_model:
        query_tokens = analyzer.preprocess_text(query, return_tokens=True)
        w2v_results = analyzer.word2vec_model.find_similar_documents(query_tokens, documents, top_n=3)
        print(f"\nWord2Vec Results for '{query}':")
        for i, (doc_id, sim, doc_text) in enumerate(w2v_results):
            print(f"{i+1}. Similarity: {sim:.4f}")
            print(f"   Document: {doc_text}")
    
    # Doc2Vec results
    if analyzer.doc2vec_model:
        query_tokens = analyzer.preprocess_text(query, return_tokens=True)
        d2v_results = analyzer.doc2vec_model.find_similar_documents(query_tokens, documents, top_n=3)
        print(f"\nDoc2Vec Results for '{query}':")
        for i, (doc_id, sim, doc_text) in enumerate(d2v_results):
            print(f"{i+1}. Similarity: {sim:.4f}")
            print(f"   Document: {doc_text}")

if __name__ == "__main__":
    main()

