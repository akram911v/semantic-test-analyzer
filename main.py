
# main.py - Assignment #5: Text Generation with Semantic Analyzers
from lsa_analyzer import LSASemanticAnalyzer
from word2vec_model import Word2VecModel
from doc2vec_model import Doc2VecModel
from cnn_analyzer import CNNSemanticAnalyzer
from rnn_analyzer import RNNSemanticAnalyzer
from lstm_generator import LSTMTemplateGenerator
from seq2seq_generator import Seq2SeqGenerator

def main():
    print("=" * 60)
    print("ASSIGNMENT #5: TEXT GENERATION WITH SEMANTIC ANALYZERS")
    print("=" * 60)
    
    # Sample documents for semantic analysis
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
    
    print("\n1. INITIALIZING ALL COMPONENTS...")
    print("-" * 40)
    
    # Initialize semantic analyzers from Assignment #4
    print("Initializing semantic analyzers...")
    
    lsa_analyzer = LSASemanticAnalyzer(n_components=5)
    print("✅ LSA Semantic Analyzer ready")
    
    word2vec_model = Word2VecModel()
    print("✅ Word2Vec Analyzer ready")
    
    doc2vec_model = Doc2VecModel()
    print("✅ Doc2Vec Analyzer ready")
    
    cnn_analyzer = CNNSemanticAnalyzer()
    print("✅ CNN Semantic Analyzer ready")
    
    rnn_analyzer = RNNSemanticAnalyzer()
    print("✅ RNN Semantic Analyzer ready")
    
    # Initialize new text generators for Assignment #5
    print("\nInitializing text generators...")
    
    lstm_generator = LSTMTemplateGenerator()
    print("✅ LSTM Template Generator ready")
    
    seq2seq_generator = Seq2SeqGenerator()
    print("✅ Sequence-to-Sequence Generator ready")
    
    print("\n" + "=" * 60)
    print("2. SEMANTIC ANALYSIS (From Assignment #4)")
    print("=" * 60)
    
    # Train LSA model
    print("\nTraining LSA model on sample documents...")
    lsa_analyzer.fit(documents[:5])
    
    # Test document similarity
    print("\nTesting Document Similarity:")
    doc1 = "Machine learning algorithms learn from data"
    doc2 = "Artificial intelligence systems can make predictions"
    
    print(f"\nDocument 1: {doc1}")
    print(f"Document 2: {doc2}")
    
    # LSA similarity
    lsa_similarity = lsa_analyzer.document_similarity(doc1, doc2)
    print(f"LSA Similarity: {lsa_similarity:.4f}")
    
    # CNN similarity
    cnn_similarity = cnn_analyzer.document_similarity(doc1, doc2)
    print(f"CNN Similarity: {cnn_similarity:.4f}")
    
    # RNN similarity
    rnn_similarity = rnn_analyzer.document_similarity(doc1, doc2)
    print(f"RNN Similarity: {rnn_similarity:.4f}")
    
    print("\n" + "=" * 60)
    print("3. TEXT GENERATION (New for Assignment #5)")
    print("=" * 60)
    
    print("\nA. LSTM TEMPLATE GENERATION")
    print("-" * 40)
    
    # Show available templates
    templates = lstm_generator.get_available_templates()
    print(f"Available templates: {len(templates)}")
    
    # Generate from templates
    print("\nGenerating text from templates:")
    for i in range(3):
        generated = lstm_generator.generate_from_template()
        print(f"{i+1}. {generated}")
    
    # Generate from specific template
    specific_template = "The [ADJECTIVE] [NOUN] [VERB] [ADVERB] in the context of [TOPIC]"
    print(f"\nGenerating from specific template:")
    print(f"Template: {specific_template}")
    generated = lstm_generator.generate_from_template(specific_template)
    print(f"Generated: {generated}")
    
    print("\nB. SEQUENCE-TO-SEQUENCE GENERATION")
    print("-" * 40)
    
    # Explain architecture
    print("\nSequence-to-Sequence Architecture Explanation:")
    print(seq2seq_generator.explain_architecture())
    
    # Generate from templates
    print("\nGenerating text with Seq2Seq templates:")
    for i in range(2):
        generated = seq2seq_generator.generate_from_template()
        print(f"{i+1}. {generated}")
    
    # Generate sequence from input
    print("\nGenerating sequence from input text:")
    input_text = "semantic analysis of documents"
    generated = seq2seq_generator.generate_sequence(input_text)
    print(generated)
    
    print("\n" + "=" * 60)
    print("4. INTEGRATED EXAMPLE")
    print("=" * 60)
    
    print("\nSemantic Analysis + Text Generation Pipeline:")
    
    # Step 1: Semantic analysis
    query = "natural language processing and text generation"
    print(f"\nStep 1: Semantic analysis of query: '{query}'")
    
    # Find similar documents using LSA
    lsa_results = lsa_analyzer.query_similarity(query, top_k=2)
    print(f"\nSimilar documents found by LSA:")
    for i, result in enumerate(lsa_results):
        print(f"  {i+1}. Similarity: {result['similarity']:.4f}")
        print(f"     {result['document']}")
    
    # Step 2: Text generation based on semantic context
    print(f"\nStep 2: Generating text based on semantic context...")
    
    # Generate LSTM text
    lstm_generated = lstm_generator.generate_from_template()
    print(f"\nLSTM Generated Text:")
    print(f"  {lstm_generated}")
    
    # Generate Seq2Seq text
    seq2seq_generated = seq2seq_generator.generate_sequence(query)
    print(f"\nSeq2Seq Generated Text:")
    print(f"  {seq2seq_generated}")
    
    print("\n" + "=" * 60)
    print("SYSTEM SUMMARY")
    print("=" * 60)
    
    print(f"""
    Total Components: 7
    --------------------
    Semantic Analyzers (5):
      • LSA (Latent Semantic Analysis)
      • Word2Vec
      • Doc2Vec
      • CNN (Convolutional Neural Network)
      • RNN (Recurrent Neural Network)
    
    Text Generators (2):
      • LSTM Template Generator
      • Sequence-to-Sequence Generator
    
    Key Features:
      • All analyzers from Assignment #4 integrated
      • LSTM-based text generation (now allowed)
      • Sequence-to-sequence generation (now allowed)
      • Template-based generation only (as required)
      • Meaningful text generation based on semantic analysis
    """)
    
    print("\n" + "=" * 60)
    print("ASSIGNMENT #5 COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()