import numpy as np
import random
import re

class CharRNNGenerator:
    def __init__(self, sequence_length=50, step=3):
        self.sequence_length = sequence_length
        self.step = step
        self.chars = []
        self.char_indices = {}
        self.indices_char = {}
        self.sentences = []
        self.next_chars = []
        
    def prepare_text(self, text):
        """Prepare text for character-level training"""
        # Clean text
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        
        # Extract character information
        self.chars = sorted(list(set(text)))
        self.char_indices = {char: i for i, char in enumerate(self.chars)}
        self.indices_char = {i: char for i, char in enumerate(self.chars)}
        
        # Build sequences
        self.sentences = []
        self.next_chars = []
        
        for i in range(0, len(text) - self.sequence_length, self.step):
            self.sentences.append(text[i:i + self.sequence_length])
            self.next_chars.append(text[i + self.sequence_length])
        
        print(f"Prepared {len(self.sentences)} sequences")
        print(f"Unique characters: {len(self.chars)}")
        
        return self
    
    def generate_text(self, seed_text, length=100, temperature=0.5):
        """Generate text character by character"""
        if not self.chars:
            return "Model not prepared. Please call prepare_text() first."
        
        generated = seed_text
        
        for i in range(length):
            # Get last sequence
            seq = generated[-self.sequence_length:]
            
            # If sequence is shorter than expected, pad it
            if len(seq) < self.sequence_length:
                seq = ' ' * (self.sequence_length - len(seq)) + seq
            
            # Generate next character (simplified version)
            # In a real implementation, this would use a trained RNN
            # For now, we'll use a probabilistic approach
            
            # Get possible next characters from training patterns
            possible_chars = []
            for j, sentence in enumerate(self.sentences):
                if seq in sentence:
                    possible_chars.append(self.next_chars[j])
            
            if possible_chars:
                # Choose next character based on frequency
                char_counts = {}
                for char in possible_chars:
                    char_counts[char] = char_counts.get(char, 0) + 1
                
                # Apply temperature
                if temperature > 0:
                    # Add some randomness
                    chars = list(char_counts.keys())
                    weights = list(char_counts.values())
                    # Adjust weights by temperature
                    weights = [w ** (1/temperature) for w in weights]
                    next_char = random.choices(chars, weights=weights)[0]
                else:
                    # Choose most common
                    next_char = max(char_counts, key=char_counts.get)
            else:
                # Fallback to random character
                next_char = random.choice(self.chars)
            
            generated += next_char
        
        # Clean up the generated text
        generated = self._format_text(generated)
        return generated
    
    def _format_text(self, text):
        """Format generated text for readability"""
        # Capitalize first letter of each sentence
        sentences = re.split(r'([.!?] )', text)
        formatted = ''
        for i, part in enumerate(sentences):
            if i % 2 == 0:  # The sentence part
                if part:
                    formatted += part[0].upper() + part[1:]
            else:  # The punctuation and space
                formatted += part
        return formatted
    
    def generate_sentence(self, seed=None, max_length=200):
        """Generate a complete sentence"""
        if seed is None:
            # Start with common words
            common_starts = ['the ', 'it ', 'we ', 'this ', 'machine ', 'learning ']
            seed = random.choice(common_starts)
        
        # Generate text
        generated = self.generate_text(seed, length=max_length)
        
        # Extract first complete sentence
        match = re.search(r'^.*?[.!?](?=\s|$)', generated)
        if match:
            return match.group(0)
        else:
            # Return first 100 characters if no sentence end found
            return generated[:100] + '...'
    
    def get_model_info(self):
        """Get information about the model"""
        return {
            'sequence_length': self.sequence_length,
            'unique_chars': len(self.chars),
            'training_sequences': len(self.sentences),
            'chars': ''.join(self.chars) if self.chars else 'None'
        }

class ChatbotGenerator(CharRNNGenerator):
    """Extended generator for chatbot responses"""
    
    def __init__(self, sequence_length=50, step=3):
        super().__init__(sequence_length, step)
        self.dialog_pairs = []
        
    def train_on_dialog(self, dialogs):
        """Train on dialog pairs for better conversation generation"""
        # Dialogs should be list of (input, response) pairs
        self.dialog_pairs = dialogs
        
        # Combine all dialog text for character model
        all_text = ''
        for question, answer in dialogs:
            all_text += question + ' ' + answer + ' '
        
        self.prepare_text(all_text)
        print(f"Trained on {len(dialogs)} dialog pairs")
        
        return self
    
    def generate_response(self, user_input, creativity=0.7):
        """Generate a chatbot response to user input"""
        # Clean input
        user_input = user_input.lower().strip()
        
        # Look for similar dialog patterns
        similar_responses = []
        for question, answer in self.dialog_pairs:
            if any(word in question for word in user_input.split()[:3]):
                similar_responses.append(answer)
        
        if similar_responses:
            # Use similar response as seed
            seed = random.choice(similar_responses)[:20]
        else:
            # Use user input as seed
            seed = user_input[:20]
        
        # Generate response
        response = self.generate_text(seed, length=random.randint(50, 150), temperature=creativity)
        
        # Extract first sentence or reasonable length
        sentences = re.split(r'[.!?]', response)
        if sentences and len(sentences[0].strip()) > 5:
            response = sentences[0].strip() + '.'
        else:
            response = response[:100].strip() + '...'
        
        return response
    
    def chat(self, user_input):
        """Simple chat interface"""
        response = self.generate_response(user_input)
        return {
            'user': user_input,
            'bot': response,
            'method': 'Character-Level RNN'
        }
