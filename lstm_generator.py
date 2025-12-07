import random

class LSTMTemplateGenerator:
    def __init__(self):
        print("LSTM Template Generator initialized")
        self.templates = [
            "The [ADJECTIVE] [NOUN] [VERB] [ADVERB]",
            "[ENTITY] is known for [ACTION]"
        ]
    
    def generate_from_template(self, template=None):
        if template is None:
            template = random.choice(self.templates)
        
        # Simple template filling
        replacements = {
            '[ADJECTIVE]': 'important',
            '[NOUN]': 'analysis', 
            '[VERB]': 'demonstrates',
            '[ADVERB]': 'clearly',
            '[ENTITY]': 'Deep learning',
            '[ACTION]': 'pattern recognition'
        }
        
        for key, value in replacements.items():
            template = template.replace(key, value)
        
        return template
    
    def get_available_templates(self):
        return self.templates