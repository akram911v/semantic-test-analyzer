import random

class ChatbotSystem:
    def __init__(self, name="ChatBot"):
        self.name = name
        self.history = []
    
    def get_response(self, user_input):
        responses = [
            f"I understand you said: '{user_input}'. How can I help you further?",
            f"That's interesting: '{user_input}'. Let me think about that.",
            f"Regarding '{user_input}', I can help with text generation topics."
        ]
        response = random.choice(responses)
        self.history.append((user_input, response))
        return response
    
    def start_chat(self):
        print(f"Starting chat with {self.name}")
        print("Type 'exit' to end chat")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            response = self.get_response(user_input)
            print(f"Bot: {response}")