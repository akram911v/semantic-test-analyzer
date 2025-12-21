import random
import re
from collections import defaultdict

class MarkovChainGenerator:
    def __init__(self, n_gram_size=2):
        self.n_gram_size = n_gram_size
        self.model = defaultdict(list)
        self.start_words = []
        
    def train(self, texts):
        """Train Markov model on provided texts"""
        print(f"Training Markov Chain on {len(texts)} texts...")
        
        for text in texts:
            # Clean and tokenize text
            words = self._clean_and_tokenize(text)
            if len(words) < self.n_gram_size:
                continue
                
            # Record starting words
            if words:
                self.start_words.append(words[0])
            
            # Build n-gram model
            for i in range(len(words) - self.n_gram_size):
                # Create n-gram key
                key = tuple(words[i:i + self.n_gram_size])
                next_word = words[i + self.n_gram_size]
                self.model[key].append(next_word)
        
        print(f"Model trained with {len(self.model)} n-grams")
        return self
    
    def _clean_and_tokenize(self, text):
        """Clean text and split into words"""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # Split into words
        words = text.split()
        return words
    
    def generate_sentence(self, max_length=20, seed=None):
        """Generate a sentence using Markov chain"""
        if not self.model:
            return "Model not trained. Please train() first."
        
        # Start with a seed or random starting word
        if seed:
            # Find n-grams that start with seed
            possible_starts = [k for k in self.model.keys() if k[0] == seed]
            if possible_starts:
                current = random.choice(possible_starts)
            else:
                current = random.choice(list(self.model.keys()))
        else:
            if self.start_words:
                first_word = random.choice(self.start_words)
                possible_starts = [k for k in self.model.keys() if k[0] == first_word]
                current = random.choice(possible_starts) if possible_starts else random.choice(list(self.model.keys()))
            else:
                current = random.choice(list(self.model.keys()))
        
        # Build the sentence
        sentence = list(current)
        
        for _ in range(max_length - self.n_gram_size):
            if current in self.model and self.model[current]:
                next_word = random.choice(self.model[current])
                sentence.append(next_word)
                # Update current n-gram
                current = tuple(sentence[-self.n_gram_size:])
            else:
                break
        
        # Capitalize first letter
        result = ' '.join(sentence)
        if result:
            result = result[0].upper() + result[1:]
        
        return result + '.'
    
    def generate_multiple(self, n=5, max_length=15):
        """Generate multiple sentences"""
        sentences = []
        for i in range(n):
            sentence = self.generate_sentence(max_length)
            sentences.append(sentence)
        return sentences

class AdvancedGenerator(MarkovChainGenerator):
    """Advanced generator with better sentence structure"""
    
    def __init__(self, n_gram_size=3):
        super().__init__(n_gram_size)
        self.sentence_enders = {'.', '!', '?'}
    
    def generate_paragraph(self, num_sentences=3):
        """Generate a paragraph with multiple sentences"""
        paragraph = []
        for i in range(num_sentences):
            sentence = self.generate_sentence(max_length=random.randint(8, 20))
            paragraph.append(sentence)
        return ' '.join(paragraph)
    
    def generate_dialog_response(self, input_text):
        """Generate a response based on input (simulating chatbot)"""
        # Extract keywords from input
        words = self._clean_and_tokenize(input_text)
        keywords = [w for w in words if len(w) > 3][:2]  # Take up to 2 keywords
        
        # Generate response based on keywords
        if keywords:
            seed = random.choice(keywords)
            response = self.generate_sentence(seed=seed, max_length=15)
        else:
            response = self.generate_sentence(max_length=15)
        
        return response
