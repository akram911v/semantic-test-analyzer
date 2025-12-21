# chatbot_system.py - Assignment 6: Dialog System
import random
import time
from datetime import datetime

class ChatbotSystem:
    def __init__(self, name="SemanticChatBot"):
        self.name = name
        self.conversation_history = []
        self.user_profile = {}
        self.generators = {
            'template': {},      # Template-based generators (from PW5)
            'non_template': {}   # Non-template generators (new for PW6)
        }
        
    def add_generator(self, name, generator, generator_type):
        """Add a text generator to the system"""
        if generator_type not in ['template', 'non_template']:
            raise ValueError("Generator type must be 'template' or 'non_template'")
        
        self.generators[generator_type][name] = {
            'instance': generator,
            'usage_count': 0,
            'response_times': []
        }
        print(f"Added {generator_type} generator: {name}")
        
        return self
    
    def get_response(self, user_input, generator_type=None, specific_generator=None):
        """Get response using appropriate generator"""
        # Clean and analyze input
        clean_input = user_input.lower().strip()
        
        # Choose generator type if not specified
        if generator_type is None:
            # Simple heuristic: for questions, use non-template
            if clean_input.endswith('?'):
                generator_type = 'non_template'
            else:
                generator_type = random.choice(['template', 'non_template'])
        
        # Choose specific generator
        if specific_generator is None:
            available = list(self.generators[generator_type].keys())
            if not available:
                # Fallback to other type
                other_type = 'non_template' if generator_type == 'template' else 'template'
                available = list(self.generators[other_type].keys())
                if available:
                    generator_type = other_type
            
            if available:
                specific_generator = random.choice(available)
            else:
                return "I don't have any text generators configured."
        
        # Get generator
        if specific_generator not in self.generators[generator_type]:
            return f"Generator '{specific_generator}' not found."
        
        generator_info = self.generators[generator_type][specific_generator]
        generator = generator_info['instance']
        
        # Generate response with timing
        start_time = time.time()
        
        try:
            if generator_type == 'template':
                # Template-based generation
                if hasattr(generator, 'generate_from_template'):
                    response = generator.generate_from_template()
                elif hasattr(generator, 'generate_sentence'):
                    response = generator.generate_sentence()
                else:
                    response = "Template generator not responding."
            else:
                # Non-template generation
                if hasattr(generator, 'generate_response'):
                    response = generator.generate_response(user_input)
                elif hasattr(generator, 'generate_sentence'):
                    response = generator.generate_sentence(seed=clean_input[:20])
                elif hasattr(generator, 'generate_dialog_response'):
                    response = generator.generate_dialog_response(user_input)
                else:
                    response = "Non-template generator not responding."
        
        except Exception as e:
            response = f"Error generating response: {str(e)}"
        
        response_time = time.time() - start_time
        
        # Update generator stats
        generator_info['usage_count'] += 1
        generator_info['response_times'].append(response_time)
        
        # Record conversation
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'response': response,
            'generator_type': generator_type,
            'generator_name': specific_generator,
            'response_time': response_time
        })
        
        return response
    
    def start_chat_session(self, session_name="Default Session"):
        """Start interactive chat session"""
        print("\n" + "="*60)
        print(f"CHATBOT SYSTEM - {self.name}")
        print(f"Session: {session_name}")
        print("="*60)
        
        print(f"\nWelcome to {self.name}!")
        print("I can chat with you using different text generation methods.")
        print("\nCommands:")
        print("  'template' - Use template-based generation")
        print("  'nontemplate' - Use non-template generation")
        print("  'auto' - Let me choose (default)")
        print("  'stats' - Show usage statistics")
        print("  'history' - Show conversation history")
        print("  'compare' - Compare generation methods")
        print("  'exit' - End chat session")
        print("\n" + "-"*40)
        
        session_messages = []
        mode = 'auto'  # Default mode
        
        while True:
            try:
                user_input = input(f"\n[You] ({mode} mode): ").strip()
                
                if user_input.lower() == 'exit':
                    print("Goodbye! Thanks for chatting.")
                    break
                elif user_input.lower() == 'template':
                    mode = 'template'
                    print("Switched to template-based generation mode.")
                    continue
                elif user_input.lower() == 'nontemplate':
                    mode = 'non_template'
                    print("Switched to non-template generation mode.")
                    continue
                elif user_input.lower() == 'auto':
                    mode = 'auto'
                    print("Switched to auto mode (I'll choose the best method).")
                    continue
                elif user_input.lower() == 'stats':
                    self.show_statistics()
                    continue
                elif user_input.lower() == 'history':
                    self.show_conversation_history()
                    continue
                elif user_input.lower() == 'compare':
                    self.compare_generation_methods()
                    continue
                elif user_input.lower() == '':
                    continue
                
                # Get response based on mode
                generator_type = None if mode == 'auto' else mode
                response = self.get_response(user_input, generator_type)
                
                # Display response
                print(f"\n[{self.name}]: {response}")
                
                # Record for session summary
                session_messages.append({
                    'user': user_input,
                    'bot': response,
                    'mode': mode
                })
                
            except KeyboardInterrupt:
                print("\n\nSession interrupted.")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
        
        # Show session summary
        self.show_session_summary(session_messages)
        
    def show_statistics(self):
        """Show usage statistics"""
        print("\n" + "="*40)
        print("GENERATOR STATISTICS")
        print("="*40)
        
        total_uses = 0
        
        for gen_type, generators in self.generators.items():
            print(f"\n{gen_type.upper()} GENERATORS:")
            if not generators:
                print("  No generators configured")
                continue
                
            for name, info in generators.items():
                uses = info['usage_count']
                total_uses += uses
                
                if info['response_times']:
                    avg_time = sum(info['response_times']) / len(info['response_times'])
                    print(f"  {name}: {uses} uses, avg time: {avg_time:.3f}s")
                else:
                    print(f"  {name}: {uses} uses")
        
        print(f"\nTotal responses generated: {total_uses}")
        print(f"Conversation history entries: {len(self.conversation_history)}")
    
    def show_conversation_history(self, limit=10):
        """Show recent conversation history"""
        print("\n" + "="*40)
        print("RECENT CONVERSATION HISTORY")
        print("="*40)
        
        if not self.conversation_history:
            print("No conversation history yet.")
            return
        
        recent = self.conversation_history[-limit:] if len(self.conversation_history) > limit else self.conversation_history
        
        for i, entry in enumerate(recent, 1):
            print(f"\n{i}. [{entry['timestamp'].split('T')[1][:8]}]")
            print(f"   You: {entry['user_input']}")
            print(f"   Bot: {entry['response'][:100]}...")
            print(f"   Method: {entry['generator_type']} ({entry['generator_name']})")
    
    def compare_generation_methods(self, test_phrases=None):
        """Compare template vs non-template generation"""
        if test_phrases is None:
            test_phrases = [
                "What is machine learning?",
                "Tell me about natural language processing",
                "How does text generation work?",
                "Explain semantic analysis"
            ]
        
        print("\n" + "="*60)
        print("GENERATION METHOD COMPARISON")
        print("="*60)
        
        results = {'template': [], 'non_template': []}
        
        for phrase in test_phrases:
            print(f"\nTest phrase: '{phrase}'")
            
            for gen_type in ['template', 'non_template']:
                # Get a generator of this type
                available = list(self.generators[gen_type].keys())
                if not available:
                    print(f"  No {gen_type} generators available")
                    continue
                
                # Use first available generator
                generator_name = available[0]
                response = self.get_response(phrase, gen_type, generator_name)
                
                results[gen_type].append({
                    'input': phrase,
                    'response': response,
                    'generator': generator_name
                })
                
                print(f"\n  {gen_type.upper()} ({generator_name}):")
                print(f"    Response: {response}")
        
        # Analysis
        print("\n" + "-"*60)
        print("COMPARISON ANALYSIS")
        print("-"*60)
        
        for gen_type in ['template', 'non_template']:
            if results[gen_type]:
                responses = [r['response'] for r in results[gen_type]]
                avg_length = sum(len(str(r)) for r in responses) / len(responses)
                
                print(f"\n{gen_type.upper()} GENERATION:")
                print(f"  • Average response length: {avg_length:.1f} characters")
                print(f"  • Response examples: {len(responses)}")
                
                # Check for common issues
                template_count = sum(1 for r in responses if '[' in str(r) and ']' in str(r))
                if template_count > 0:
                    print(f"  • Template markers present: {template_count}/{len(responses)}")
    
    def show_session_summary(self, session_messages):
        """Show summary of the chat session"""
        if not session_messages:
            return
        
        print("\n" + "="*60)
        print("CHAT SESSION SUMMARY")
        print("="*60)
        
        print(f"\nTotal exchanges: {len(session_messages)}")
        
        # Count by mode
        mode_counts = {}
        for msg in session_messages:
            mode_counts[msg['mode']] = mode_counts.get(msg['mode'], 0) + 1
        
        print("\nGeneration modes used:")
        for mode, count in mode_counts.items():
            print(f"  • {mode}: {count} times")
        
        print("\nThank you for using the Semantic Chatbot System!")
