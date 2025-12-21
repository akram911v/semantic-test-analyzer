# chatbot_demo.py - Assignment 6: Autonomous Dialog System
from lstm_generator import LSTMTemplateGenerator
from seq2seq_generator import Seq2SeqGenerator
from markov_generator import MarkovChainGenerator, AdvancedGenerator
from char_rnn_generator import CharRNNGenerator, ChatbotGenerator
from chatbot_system import ChatbotSystem

def main():
    print("=" * 70)
    print("ASSIGNMENT 6: AUTONOMOUS DIALOG SYSTEM (CHATBOT)")
    print("=" * 70)
    
    # Training data for non-template generators
    training_texts = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers for complex pattern recognition.",
        "Natural language processing helps computers understand and process human language.",
        "Computer vision enables machines to interpret and understand visual information.",
        "Data science involves extracting insights from structured and unstructured data.",
        "Python is a popular programming language for data analysis and machine learning.",
        "TensorFlow and PyTorch are widely used frameworks for deep learning projects.",
        "Supervised learning uses labeled datasets to train machine learning models.",
        "Unsupervised learning finds patterns in data without predefined labels.",
        "Reinforcement learning trains models through rewards and punishments system.",
        "Semantic analysis extracts meaning from text and identifies relationships between words.",
        "Text generation creates new text based on patterns learned from existing documents.",
        "Chatbots simulate human conversation using natural language processing techniques.",
        "Neural networks are inspired by the human brain and consist of interconnected nodes.",
        "The transformer architecture revolutionized natural language processing tasks."
    ]
    
    # Dialog training pairs for chatbot
    dialog_pairs = [
        ("What is machine learning?", "Machine learning is a branch of artificial intelligence that enables systems to learn from data."),
        ("Explain neural networks", "Neural networks are computing systems inspired by biological neural networks in brains."),
        ("What does NLP stand for?", "NLP stands for Natural Language Processing, which helps computers understand human language."),
        ("Tell me about deep learning", "Deep learning uses multi-layer neural networks to learn complex patterns from large amounts of data."),
        ("How does text generation work?", "Text generation systems create new text by learning patterns from existing text data."),
        ("What is semantic analysis?", "Semantic analysis extracts meaning from text to understand context and relationships between words."),
        ("Can you explain transformers?", "Transformers are neural network architectures that use attention mechanisms for sequence processing."),
        ("What is a chatbot?", "A chatbot is an AI system designed to simulate conversation with human users."),
        ("How do you generate responses?", "I generate responses using different text generation methods, both template-based and non-template."),
        ("What are your capabilities?", "I can discuss AI topics, generate text, and compare different text generation approaches.")
    ]
    
    print("\n1. INITIALIZING GENERATORS")
    print("-" * 40)
    
    # Initialize Template Generators (from Assignment 5)
    print("Initializing Template Generators (Assignment 5)...")
    lstm_gen = LSTMTemplateGenerator()
    seq2seq_gen = Seq2SeqGenerator()
    print("✅ LSTM Template Generator ready")
    print("✅ Seq2Seq Template Generator ready")
    
    # Initialize Non-Template Generators (new for Assignment 6)
    print("\nInitializing Non-Template Generators (Assignment 6)...")
    
    # Markov Chain Generator
    markov_gen = MarkovChainGenerator(n_gram_size=2)
    markov_gen.train(training_texts)
    print("✅ Markov Chain Generator trained and ready")
    
    # Advanced Markov Generator
    advanced_gen = AdvancedGenerator(n_gram_size=3)
    advanced_gen.train(training_texts)
    print("✅ Advanced Markov Generator trained and ready")
    
    # Character-Level RNN Generator
    char_rnn_gen = CharRNNGenerator(sequence_length=30, step=3)
    combined_text = " ".join(training_texts)
    char_rnn_gen.prepare_text(combined_text)
    print("✅ Character-Level RNN Generator prepared and ready")
    
    # Chatbot Generator (trained on dialogs)
    chatbot_gen = ChatbotGenerator(sequence_length=30, step=3)
    chatbot_gen.train_on_dialog(dialog_pairs)
    print("✅ Chatbot Generator trained on dialog pairs")
    
    print("\n2. SETTING UP CHATBOT SYSTEM")
    print("-" * 40)
    
    # Create chatbot system
    chatbot = ChatbotSystem(name="SemanticAI Bot")
    
    # Add template generators (from PW5)
    chatbot.add_generator("LSTM Template", lstm_gen, "template")
    chatbot.add_generator("Seq2Seq Template", seq2seq_gen, "template")
    
    # Add non-template generators (from PW6)
    chatbot.add_generator("Markov Chain", markov_gen, "non_template")
    chatbot.add_generator("Advanced Markov", advanced_gen, "non_template")
    chatbot.add_generator("Character RNN", char_rnn_gen, "non_template")
    chatbot.add_generator("Dialog Chatbot", chatbot_gen, "non_template")
    
    print(f"\n✅ Chatbot system initialized with {len(chatbot.generators['template'])} template generators")
    print(f"✅ and {len(chatbot.generators['non_template'])} non-template generators")
    
    print("\n" + "=" * 70)
    print("3. COMPARISON: TEMPLATE vs NON-TEMPLATE GENERATION")
    print("=" * 70)
    
    # Test phrases for comparison
    test_phrases = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "How does text generation work?",
        "Tell me about natural language processing"
    ]
    
    # Run comparison
    chatbot.compare_generation_methods(test_phrases)
    
    print("\n" + "=" * 70)
    print("4. EXAMPLE CHAT SESSIONS")
    print("=" * 70)
    
    # Example 1: Template-based generation
    print("\nEXAMPLE 1: Template-Based Generation")
    print("-" * 40)
    test_input = "What is machine learning?"
    response = chatbot.get_response(test_input, generator_type="template", specific_generator="LSTM Template")
    print(f"You: {test_input}")
    print(f"Bot (Template): {response}")
    
    # Example 2: Non-template generation
    print("\nEXAMPLE 2: Non-Template Generation")
    print("-" * 40)
    test_input = "How does a neural network work?"
    response = chatbot.get_response(test_input, generator_type="non_template", specific_generator="Markov Chain")
    print(f"You: {test_input}")
    print(f"Bot (Non-Template): {response}")
    
    # Example 3: Dialog-focused generation
    print("\nEXAMPLE 3: Dialog-Focused Generation")
    print("-" * 40)
    test_input = "Can you explain semantic analysis?"
    response = chatbot.get_response(test_input, generator_type="non_template", specific_generator="Dialog Chatbot")
    print(f"You: {test_input}")
    print(f"Bot (Dialog): {response}")
    
    print("\n" + "=" * 70)
    print("5. KEY DIFFERENCES ANALYSIS")
    print("=" * 70)
    
    print("""
    TEMPLATE-BASED GENERATION (Assignment 5):
    • Uses predefined templates with placeholders
    • Fills slots with appropriate words from categories
    • Guarantees grammatical structure
    • Limited creativity and variety
    • Predictable output patterns
    
    NON-TEMPLATE GENERATION (Assignment 6):
    • No predefined templates or patterns
    • Generates text based on learned probabilities
    • More creative and varied output
    • Can produce novel combinations
    • Less predictable, more natural
    
    ADVANTAGES OF NON-TEMPLATE APPROACH:
    1. Greater creativity and novelty
    2. Can generate unexpected but relevant responses
    3. Better handles varied input
    4. More natural conversation flow
    5. Learns patterns from data
    
    ADVANTAGES OF TEMPLATE APPROACH:
    1. Guaranteed grammatical correctness
    2. More controlled output
    3. Easier to debug and modify
    4. Consistent quality
    5. No risk of generating nonsense
    """)
    
    print("\n" + "=" * 70)
    print("6. INTERACTIVE CHAT SESSION")
    print("=" * 70)
    
    print("""
    You can now start an interactive chat session!
    
    The chatbot system supports three modes:
    1. 'template' - Use template-based generation only
    2. 'nontemplate' - Use non-template generation only  
    3. 'auto' - Let the system choose (default)
    
    During the chat, you can use these commands:
    • 'stats' - Show usage statistics
    • 'history' - Show conversation history
    • 'compare' - Compare generation methods
    • 'exit' - End the chat session
    
    Would you like to start the interactive chat? (y/n)
    """)
    
    choice = input("> ").strip().lower()
    if choice in ['y', 'yes']:
        chatbot.start_chat_session(session_name="Assignment 6 Demo")
    else:
        print("\nSkipping interactive chat. Summary of capabilities:")
        chatbot.show_statistics()
    
    print("\n" + "=" * 70)
    print("ASSIGNMENT 6 COMPLETE!")
    print("=" * 70)
    
    print("""
    SUMMARY:
    • Developed autonomous dialog system (chatbot)
    • Integrated template generators from Assignment 5
    • Created new non-template generators for Assignment 6
    • Implemented comparison between generation methods
    • Provided interactive chat interface
    • Demonstrated advantages/disadvantages of each approach
    
    The system now demonstrates meaningful text generation
    using both template-based and non-template approaches
    as required by Assignment 6.
    """)

if __name__ == "__main__":
    main()
