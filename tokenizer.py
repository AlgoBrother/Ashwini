import collections
import sys
import codecs
import json
from typing import Dict, List, Tuple, Set, NamedTuple
import regex as re
from dataclasses import dataclass
import numpy as np
from prettytable import PrettyTable
import time
import psutil
import os

# Function to get current memory usage in MB
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)  # Convert from bytes to MB

# Start tracking memory and time
start_time = time.time()
start_memory = get_memory_usage()

@dataclass
class Token:
    text: str
    count: int
    
class ModelStats(NamedTuple):
    vocab_size: int
    total_parameters: int
    embedding_params: int
    context_window: int
    total_tokens_processed: int
    unique_tokens: int
    token_frequency: Dict[int, int]
    bytes_per_token: float
    compression_ratio: float
    
class TokenizerStats:
    def __init__(self):
        self.total_tokens_processed = 0
        self.unique_tokens_seen = set()
        self.token_frequencies = collections.Counter()
        self.total_bytes = 0
        self.total_characters = 0
        
    def update(self, tokens: List[int], original_text: str):
        self.total_tokens_processed += len(tokens)
        self.unique_tokens_seen.update(tokens)
        self.token_frequencies.update(tokens)
        self.total_bytes += len(original_text.encode('utf-8'))
        self.total_characters += len(original_text)

class Tokenizer:
    def __init__(self, vocab_size: int = 300, embedding_dim: int = 768, context_window: int = 2048):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        self.cache = {}
        self.stats = TokenizerStats()
        
    def train(self, texts: List[str], min_frequency: int = 2):
        """Train tokenizer using BPE algorithm with frequency-based pruning"""
        all_tokens = []
        for text in texts:
            tokens = list(text.encode("utf-8"))
            all_tokens.extend(tokens)
            
        token_freqs = collections.Counter(all_tokens)
        current_tokens = all_tokens
        next_token_id = 256 + len(self.special_tokens)
        
        while len(self.vocab) < self.vocab_size:
            pairs = self._get_pair_frequencies(current_tokens)
            if not pairs:
                break
                
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]

            if len(self.vocab) + 1 > self.vocab_size:
                break
                
            if pairs[best_pair] < min_frequency:
                break
                
            new_token = self._merge_pair(best_pair, next_token_id)
            self.vocab[next_token_id] = new_token
            self.merges[best_pair] = next_token_id
            
            current_tokens = self._apply_merge(current_tokens, best_pair, next_token_id)
            next_token_id += 1
            
    def _get_pair_frequencies(self, tokens: List[int]) -> Dict[Tuple[int, int], int]:
        pairs = collections.Counter()
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += 1
        return pairs
        
    def _merge_pair(self, pair: Tuple[int, int], new_id: int) -> bytes:
        return self.vocab[pair[0]] + self.vocab[pair[1]]
        
    def _apply_merge(self, tokens: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        new_tokens = []
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        if i < len(tokens):
            new_tokens.append(tokens[i])
        return new_tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if text in self.cache:
            return self.cache[text]
            
        tokens = list(text.encode("utf-8"))
        
        while len(tokens) >= 2:
            mergeable = False
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    new_id = self.merges[pair]
                    tokens = tokens[:i] + [new_id] + tokens[i + 2:]
                    mergeable = True
                    break
            if not mergeable:
                break
                
        if add_special_tokens:
            tokens = [self.special_tokens['<BOS>']] + tokens + [self.special_tokens['<EOS>']]
            
        self.cache[text] = tokens
        self.stats.update(tokens, text)  # Update statistics
        return tokens
        
    def decode(self, ids: List[int]) -> str:
        ids = [id for id in ids if id not in self.special_tokens.values()]
        bytes_tokens = b"".join(self.vocab[id] for id in ids)
        return bytes_tokens.decode("utf-8", errors="replace")
    
    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        return text.strip().lower()
    
    def get_model_stats(self) -> ModelStats:
        """Calculate and return comprehensive model statistics"""
        embedding_params = self.vocab_size * self.embedding_dim
        total_parameters = embedding_params
        
        bytes_per_token = (self.stats.total_bytes / self.stats.total_tokens_processed 
                          if self.stats.total_tokens_processed > 0 else 0)
        compression_ratio = (self.stats.total_characters / self.stats.total_tokens_processed 
                           if self.stats.total_tokens_processed > 0 else 0)
        
        return ModelStats(
            vocab_size=self.vocab_size,
            total_parameters=total_parameters,
            embedding_params=embedding_params,
            context_window=self.context_window,
            total_tokens_processed=self.stats.total_tokens_processed,
            unique_tokens=len(self.stats.unique_tokens_seen),
            token_frequency=dict(self.stats.token_frequencies),
            bytes_per_token=bytes_per_token,
            compression_ratio=compression_ratio
        )
    
    def print_model_card(self):
        """Print a formatted model card with all relevant statistics"""
        stats = self.get_model_stats()
        
        print("\n=== Model Card ===")
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.align["Metric"] = "l"
        table.align["Value"] = "r"
        
        table.add_row(["Vocabulary Size", f"{stats.vocab_size:,}"])
        table.add_row(["Total Parameters", f"{stats.total_parameters:,}"])
        table.add_row(["Embedding Parameters", f"{stats.embedding_params:,}"])
        table.add_row(["Context Window", f"{stats.context_window:,}"])
        table.add_row(["Total Tokens Processed", f"{stats.total_tokens_processed:,}"])
        table.add_row(["Unique Tokens Used", f"{stats.unique_tokens:,}"])
        table.add_row(["Bytes per Token", f"{stats.bytes_per_token:.2f}"])
        table.add_row(["Compression Ratio", f"{stats.compression_ratio:.2f}"])
        
        print(table)
        
        print("\nToken Frequency Distribution:")
        freq_table = PrettyTable()
        freq_table.field_names = ["Frequency Range", "Count"]
        freq_table.align["Frequency Range"] = "l"
        freq_table.align["Count"] = "r"
        
        ranges = [(1, 10), (11, 100), (101, 1000), (1001, float('inf'))]
        for start, end in ranges:
            count = sum(1 for freq in stats.token_frequency.values() 
                       if start <= freq <= end)
            freq_table.add_row([f"{start}-{end if end != float('inf') else '∞'}", count])
            
        print(freq_table)
    
    def export_stats(self, filepath: str):
        """Export model statistics to a JSON file"""
        stats = self.get_model_stats()
        stats_dict = {
            "model_config": {
                "vocab_size": stats.vocab_size,
                "embedding_dim": self.embedding_dim,
                "context_window": stats.context_window,
                "total_parameters": stats.total_parameters
            },
            "training_stats": {
                "total_tokens_processed": stats.total_tokens_processed,
                "unique_tokens": stats.unique_tokens,
                "bytes_per_token": stats.bytes_per_token,
                "compression_ratio": stats.compression_ratio
            },
            "token_frequency": {
                str(k): v for k, v in sorted(
                    stats.token_frequency.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:100]  # Top 100 tokens
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats_dict, f, indent=2)

def test_tokenizer(tokenizer):
    with open('C:\\Users\\Ashwin Rajhans\\SIH\\SIH\\data.json') as f:
        data = json.load(f)
    
    # Collect utterances and tags from the data
    utterances = []
    tags = []
    
    for dialogue in data:
        for turn in dialogue['turns']:
            utterances.append(turn['utterance'])
            tags.extend(turn['tags'])
    
    # Test tokenization on the utterances and tags
    print("=== Testing Tokenizer on Utterances ===")
    for utterance in utterances:
        tokens = tokenizer.encode(utterance)
        decoded_text = tokenizer.decode(tokens)
        print(f"Original: {utterance}")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {decoded_text}")
        print("-------------")
        
    print("\n=== Testing Tokenizer on Tags ===")
    for tag in tags:
        tokens = tokenizer.encode(tag)
        decoded_text = tokenizer.decode(tokens)
        print(f"Tag: {tag}")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {decoded_text}")
        print("-------------")
    
    # Print model statistics after testing
    print("\n=== Model Statistics ===")
    tokenizer.print_model_card()
    
    # Export statistics to file
    tokenizer.export_stats("tokenizer_stats.json")

if __name__ == "__main__":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    
    # Initialize tokenizer with desired parameters
    tokenizer = Tokenizer(
        vocab_size=300,
        embedding_dim=768,
        context_window=2048
    )
    
    # Train tokenizer
    with open('C:\\Users\\Ashwin Rajhans\\SIH\\SIH\\data.json') as f:
        data = json.load(f)
    tokenizer.train([turn['utterance'] for dialogue in data for turn in dialogue['turns']])
    
    # Run tests and print statistics
    test_tokenizer(tokenizer)
    
end_time = time.time()
end_memory = get_memory_usage()

# Calculate total time and memory usage
execution_time = end_time - start_time
memory_used = end_memory - start_memory

# Print out the results
print(f"Execution Time: {execution_time:.4f} seconds")
print(f"Memory Used: {memory_used:.4f} MB")
    
