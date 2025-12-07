import random

class Seq2SeqGenerator:
    def __init__(self):
        print("Sequence-to-Sequence Generator initialized")
    
    def generate_from_template(self, template=None):
        templates = [
            "Based on [CONTEXT], we generate [CONTENT]",
            "The model transforms [INPUT] into [OUTPUT]"
        ]
        
        if template is None:
            template = random.choice(templates)
        
        replacements = {
            '[CONTEXT]': 'semantic analysis',
            '[CONTENT]': 'coherent text',
            '[INPUT]': 'input sequence', 
            '[OUTPUT]': 'output text'
        }
        
        for key, value in replacements.items():
            template = template.replace(key, value)
        
        return template
    
    def explain_architecture(self):
        return "Seq2Seq architecture: Encoder LSTM processes input, Decoder LSTM generates output."
    
    def generate_sequence(self, input_text):
        responses = [
            f"Based on '{input_text}', the system generates meaningful text.",
            f"Input analysis of '{input_text}' leads to coherent output generation."
        ]
        return random.choice(responses)