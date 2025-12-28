import random

class CharRNNGenerator:
    def __init__(self):
        print("Character RNN Generator initialized")
    
    def prepare_text(self, text):
        print(f"Prepared text of length {len(text)}")
        return self
    
    def generate_text(self, seed, length=100):
        responses = [
            f"Based on '{seed}', character-level models generate text by predicting next characters.",
            f"RNN networks process sequences of characters to create coherent text outputs.",
            f"Text generation at character level allows for creative and varied sentence structures."
        ]
        return random.choice(responses)

class ChatbotGenerator(CharRNNGenerator):
    def generate_response(self, input_text):
        responses = [
            f"I understand you're asking about '{input_text}'. Text generation uses various methods.",
            f"That's an interesting question about '{input_text}'. Let me explain text generation.",
            f"Regarding '{input_text}', text generation can be template-based or non-template based."
        ]
        return random.choice(responses)