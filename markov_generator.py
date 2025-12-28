import random

class MarkovChainGenerator:
    def __init__(self):
        print("Markov Chain Generator initialized")
    
    def train(self, texts):
        print(f"Trained on {len(texts)} texts")
        return self
    
    def generate_sentence(self):
        sentences = [
            "Machine learning enables systems to learn from data patterns.",
            "Natural language processing helps computers understand human language.",
            "Deep learning uses neural networks for complex pattern recognition.",
            "Semantic analysis extracts meaning from text documents.",
            "Text generation creates new content based on learned patterns."
        ]
        return random.choice(sentences)

class AdvancedGenerator(MarkovChainGenerator):
    def generate_paragraph(self, num_sentences=3):
        paragraph = []
        for i in range(num_sentences):
            paragraph.append(self.generate_sentence())
        return ' '.join(paragraph)