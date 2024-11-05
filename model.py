import json
import re
import os
from tokenizer import Tokenizer;
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

class HALSystem:
    def __init__(self, training_data_file):
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, training_data_file)
        
        try:
            with open(file_path, 'r') as file:
                self.training_data = json.load(file)
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            print(f"Current directory: {current_dir}")
            print(f"Files in current directory: {os.listdir(current_dir)}")
            raise
        
        self.intent_patterns = self._extract_intent_patterns()

    def _extract_intent_patterns(self):
        intent_patterns = {}
        for dialogue in self.training_data:
            for turn in dialogue['turns']:
                if turn['speaker'] == 'astronaut':
                    intent = turn['intent']
                    utterance = turn['utterance'].lower()
                    pattern = re.sub(r'[^\w\s]', '', utterance)
                    words = pattern.split()
                    if intent not in intent_patterns:
                        intent_patterns[intent] = set()
                    intent_patterns[intent].update(words)
        return intent_patterns

    def _get_intent(self, user_input):
        user_words = set(re.sub(r'[^\w\s]', '', user_input.lower()).split())
        best_intent = None
        best_match = 0
        for intent, patterns in self.intent_patterns.items():
            match = len(user_words.intersection(patterns))
            if match > best_match:
                best_match = match
                best_intent = intent
        return best_intent if best_intent else "unknown"

    def generate_response(self, user_input):
        intent = self._get_intent(user_input)
        if intent == "unknown":
            return "I'm sorry, I don't understand. Could you please rephrase your request?"

        # Collect all relevant responses
        responses = []
        for dialogue in self.training_data:
            for i, turn in enumerate(dialogue['turns']):
                if turn['intent'] == intent and turn['speaker'] == 'astronaut':
                    dialogue_responses = []
                    for j in range(i + 1, len(dialogue['turns'])):
                        if dialogue['turns'][j]['speaker'] == 'system':
                            dialogue_responses.append(dialogue['turns'][j]['utterance'])
                        else:
                            break
                    if dialogue_responses:
                        responses.extend(dialogue_responses)

        # If we have responses, join them with proper punctuation
        if responses:
            return " ".join(responses)
        
        return "I'm having trouble processing that request. Could you try again?"

hal = HALSystem('C:\\Users\\Ashwin Rajhans\\SIH\\SIH\\data.json')

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {e}")
        return "Error loading page", 500

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json['input']
    response = hal.generate_response(user_input)
    
    # Split long responses into sentences
    sentences = re.split('(?<=[.!?]) +', response)
    
    return jsonify({
        'response': response,
        'sentences': sentences
    })

if __name__ == '__main__':
    socketio.run(app, debug=True)