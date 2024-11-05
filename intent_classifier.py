import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenizer import Tokenizer
import numpy as np

class IntegratedClassificationSystem:
    def __init__(self, data_path: str, custom_tokenizer, embedding_dim: int = 256, max_sequence_length: int = 50):
        self.data_path = data_path
        self.tokenizer = custom_tokenizer
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length

        self.model = None
        self.label_encoder = LabelEncoder()
        self.intent_features = {}

        # Initialize NLTK components for feature extraction
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Create directories for logs and models
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/integrated_classifier_{self.timestamp}"
        self.model_dir = "models"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        # ... (keep existing initialization code) ...
        self.vocab_size = custom_tokenizer.vocab_size
        # Add padding_idx for the embedding layer
        self.padding_idx = 0

    def extract_intent_features(self, data: List[dict]) -> Dict[str, Dict[str, float]]:
        """Extract and score features for each intent using both BPE and linguistic features."""
        intent_tokens = {}
        intent_tags = {}

        for dialogue in data:
            for turn in dialogue['turns']:
                intent = turn['intent']
                if intent not in intent_tokens:
                    intent_tokens[intent] = []
                    intent_tags[intent] = []

                # Get BPE tokens
                bpe_tokens = self.tokenizer.encode(turn['utterance'])
                intent_tokens[intent].extend(bpe_tokens)

                # Get linguistic tokens
                linguistic_tokens = self.preprocess_text(turn['utterance'])
                intent_tokens[intent].extend(linguistic_tokens)

                # Include tags
                if 'tags' in turn:
                    intent_tags[intent].extend(turn['tags'])

        # Calculate feature scores
        for intent in intent_tokens:
            # Count token frequencies
            token_freq = Counter(intent_tokens[intent])
            tag_freq = Counter(intent_tags[intent])

            total_tokens = len(intent_tokens[intent])

            # Calculate scores considering both token frequency and tag presence
            feature_scores = {}
            for token, freq in token_freq.items():
                # Base score from token frequency
                score = freq / total_tokens

                # Boost score if token appears in tags
                if str(token) in tag_freq:
                    score *= 1.5

                feature_scores[token] = score

            self.intent_features[intent] = feature_scores

        return self.intent_features

    def prepare_input_features(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare both BPE and feature-based inputs for the model."""
        # Get BPE tokens and ensure they're within vocab bounds
        bpe_tokens = self.tokenizer.encode(text)
        # Clip token IDs to be within vocabulary size
        bpe_tokens = [min(token, self.vocab_size - 1) for token in bpe_tokens]
        padded_tokens = pad_sequences([bpe_tokens], maxlen=self.max_sequence_length, padding='post')[0]

        # Calculate feature vector (keep existing feature calculation)
        feature_vec = []
        tokens = self.preprocess_text(text)

        for intent, features in self.intent_features.items():
            score = sum(features.get(token, 0) for token in tokens)
            feature_vec.append(score)

        return padded_tokens, np.array(feature_vec)

    def build_model(self, num_classes: int, feature_dim: int):
        """Build a dual-input model combining BPE tokens and extracted features."""
        # Token input branch
        token_input = tf.keras.Input(shape=(self.max_sequence_length,))
        # Add mask_zero=True and set proper vocab size
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            embeddings_initializer='uniform'
        )(token_input)

        # Rest of the model architecture remains the same
        x1 = LayerNormalization()(embedding)
        x1 = Bidirectional(LSTM(128, return_sequences=True))(x1)
        x1 = Dropout(0.4)(x1)
        x1 = LayerNormalization()(x1)
        x1 = Bidirectional(LSTM(64))(x1)
        x1 = Dropout(0.4)(x1)
        x1 = LayerNormalization()(x1)

        # Feature input branch
        feature_input = tf.keras.Input(shape=(feature_dim,))
        x2 = Dense(64, activation='relu')(feature_input)
        x2 = Dropout(0.3)(x2)

        # Combine branches
        combined = tf.keras.layers.concatenate([x1, x2])

        x = Dense(256, activation='relu')(combined)
        x = Dropout(0.4)(x)
        x = LayerNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=[token_input, feature_input], outputs=outputs)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text with custom tokenizer and additional processing."""
        # Apply tokenizer's preprocessing first
        text = self.tokenizer.preprocess_text(text)

        # Additional preprocessing for feature extraction
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                 if token not in self.stop_words and token.isalnum()]
        return tokens

    def predict(self, text: str, return_probabilities: bool = False):
        """Predict intent for new text using both BPE tokens and features."""
        tokens, features = self.prepare_input_features(text)

        # Expand dimensions for batch size of 1
        tokens = np.expand_dims(tokens, 0)
        features = np.expand_dims(features, 0)

        predictions = self.model.predict([tokens, features])

        if return_probabilities:
            return {
                intent: prob
                for intent, prob in zip(self.label_encoder.classes_, predictions[0])
            }

        predicted_index = np.argmax(predictions[0])
        return self.label_encoder.classes_[predicted_index]

    def main1():
        # Load data
        with open('/content/data.json') as f:
            data = json.load(f)

        # Initialize custom tokenizer with proper vocabulary size
        custom_tokenizer = Tokenizer(vocab_size=300, embedding_dim=256)

        # Make sure to train the tokenizer with all possible tokens
        all_utterances = [turn['utterance'] for dialogue in data for turn in dialogue['turns']]
        custom_tokenizer.train(all_utterances)

        # Initialize and train the classification system
        classifier = IntegratedClassificationSystem(
            data_path='/content/data.json',
            custom_tokenizer=custom_tokenizer,
            embedding_dim=256,
            max_sequence_length=50
        )

        # Rest of the main function remains the same
            # Test the system
        test_utterance = "What's the status of the solar panel array?"
        predicted_intent = classifier.predict(test_utterance, return_probabilities=True)

        print(f"\nTest Utterance: {test_utterance}")
        print("\nPredicted Intents and Probabilities:")
        for intent, prob in sorted(predicted_intent.items(), key=lambda x: x[1], reverse=True):
            print(f"{intent}: {prob:.3f}")
        classifier.extract_intent_features(data)
        history = classifier.train(data, epochs=50)



if __name__ == "__main__":
  main1()