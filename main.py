from lsa_analyzer import LSASemanticAnalyzer
from word2vec_model import Word2VecModel
from doc2vec_model import Doc2VecModel
from cnn_analyzer import CNNSemanticAnalyzer
from rnn_analyzer import RNNSemanticAnalyzer

def main():
    print("=== Assignment #4: CNN and RNN Semantic Analyzers ===")
    
    # Sample documents
    doc1 = "Machine learning is a subset of artificial intelligence"
    doc2 = "AI and machine learning are transforming technology"
    doc3 = "The weather today is sunny and warm"
    
    print(f"Document 1: {doc1}")
    print(f"Document 2: {doc2}")
    print(f"Document 3: {doc3}")
    
    print("\nInitializing analyzers...")
    
    # Initialize available analyzers
    analyzers = {}
    
    try:
        analyzers['LSA'] = LSASemanticAnalyzer()
        print("✅ LSA analyzer ready")
    except Exception as e:
        print(f"❌ LSA analyzer not available: {e}")
    
    try:
        analyzers['Word2Vec'] = Word2VecModel()
        print("✅ Word2Vec analyzer ready")
    except Exception as e:
        print(f"❌ Word2Vec analyzer not available: {e}")
    
    try:
        analyzers['Doc2Vec'] = Doc2VecModel()
        print("✅ Doc2Vec analyzer ready")
    except Exception as e:
        print(f"❌ Doc2Vec analyzer not available: {e}")
    
    analyzers['CNN'] = CNNSemanticAnalyzer()
    print("✅ CNN analyzer ready")
    
    analyzers['RNN'] = RNNSemanticAnalyzer()
    print("✅ RNN analyzer ready")
    
    print("\n=== Testing Document Similarity ===")
    
    # Test CNN
    cnn_sim1 = analyzers['CNN'].document_similarity(doc1, doc2)
    cnn_sim2 = analyzers['CNN'].document_similarity(doc1, doc3)
    print(f"CNN Similarity (Doc1-Doc2): {cnn_sim1:.4f}")
    print(f"CNN Similarity (Doc1-Doc3): {cnn_sim2:.4f}")
    
    # Test RNN
    rnn_sim1 = analyzers['RNN'].document_similarity(doc1, doc2)
    rnn_sim2 = analyzers['RNN'].document_similarity(doc1, doc3)
    print(f"RNN Similarity (Doc1-Doc2): {rnn_sim1:.4f}")
    print(f"RNN Similarity (Doc1-Doc3): {rnn_sim2:.4f}")
    
    print(f"\n=== Successfully initialized {len(analyzers)} analyzers ===")
    print("Assignment #4 Complete!")

if __name__ == "__main__":
    main()
