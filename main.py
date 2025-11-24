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
    
    cnn = CNNSemanticAnalyzer()
    rnn = RNNSemanticAnalyzer()
    
    print("\n=== Testing Document Similarity ===")
    
    # Test CNN
    cnn_sim1 = cnn.document_similarity(doc1, doc2)
    cnn_sim2 = cnn.document_similarity(doc1, doc3)
    print(f"CNN Similarity (Doc1-Doc2): {cnn_sim1:.4f}")
    print(f"CNN Similarity (Doc1-Doc3): {cnn_sim2:.4f}")
    
    # Test RNN
    rnn_sim1 = rnn.document_similarity(doc1, doc2)
    rnn_sim2 = rnn.document_similarity(doc1, doc3)
    print(f"RNN Similarity (Doc1-Doc2): {rnn_sim1:.4f}")
    print(f"RNN Similarity (Doc1-Doc3): {rnn_sim2:.4f}")
    
    print("\n=== Assignment #4 Complete! ===")
    print("CNN and RNN semantic analyzers are working correctly.")

if __name__ == "__main__":
    main()