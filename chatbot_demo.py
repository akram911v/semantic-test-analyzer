from markov_generator import MarkovChainGenerator
from char_rnn_generator import ChatbotGenerator
from chatbot_system import ChatbotSystem

def main():
    print("=" * 60)
    print("Assignment 6: Chatbot System Demo")
    print("=" * 60)
    
    # Initialize generators
    markov = MarkovChainGenerator()
    chatbot_gen = ChatbotGenerator()
    
    # Create chatbot system
    chatbot = ChatbotSystem("SemanticBot")
    
    print("\n1. Testing Markov Generator:")
    markov.train(["Sample training text"])
    print(f"Generated: {markov.generate_sentence()}")
    
    print("\n2. Testing Chatbot Generator:")
    chatbot_gen.prepare_text("Training data")
    print(f"Response: {chatbot_gen.generate_response('What is AI?')}")
    
    print("\n3. Testing Chatbot System:")
    print(f"Response: {chatbot.get_response('Hello, how are you?')}")
    
    print("\n4. Starting Interactive Chat:")
    print("=" * 60)
    chatbot.start_chat()
    
    print("\n" + "=" * 60)
    print("Assignment 6 Demo Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()