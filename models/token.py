from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from collections import defaultdict

class LogTemplateMiner:
    def __init__(self, state_path):
        """
        Initializes the tokenizer using Drain3 for log messages.
        :param seq_len: Maximum sequence length for tokenized logs.
        :param special_tokens: List of special tokens.
        """
        self.template_miner = TemplateMiner(persistence_handler=FilePersistence(state_path))

    def transform(self, log_message):
        """ Returns a function that encodes log messages into token IDs. """
        return self.get_event_id(log_message)

    def train_template_miner(self, log_messages):
        """ Processes logs with Drain3 and stores event templates & event IDs. """
        for log in log_messages:
            self.add_log_message(log)
            
    def add_log_message(self, log_message):
        """ Adds a single log message to the template miner. """
        self.template_miner.add_log_message(log_message)
            
    def get_event_id(self, log_message):
        """ Retrieves the event ID for a given log message. """
        result = self.template_miner.match(log_message, 'fallback')
        return result.cluster_id

    def get_vocab_size(self):
        """ Returns the vocabulary size of the Drain3 tokenizer. """
        return self.template_miner.drain.clusters_counter + 1

    def get_vocab(self):
        """ Returns the vocabulary of the Drain3 tokenizer. """
        return self.template_miner.drain

    def decode_event_id_sequence(self, event_id):
        """ Converts tokenized event sequence back into a log template. """
        decoded = self.template_miner.drain.id_to_cluster.get(event_id, "[UNK]")
        if isinstance(decoded, str):
            return decoded
        return " ".join(decoded.log_template_tokens)
    
    def load_state(self):
        """ Loads a previously saved state of the tokenizer. """
        self.template_miner.load_state()
        
    def save_state(self):
        """ Saves the current state of the tokenizer. """
        self.template_miner.save_state("For re-use")

class LogTokenizer:
    def __init__(self, tokenizer_length, tokenizer_path=None, vocab_size=30522):
        """
        Initializes the tokenizer. If a path is provided, it loads an existing tokenizer.
        Otherwise, it creates a new tokenizer with incremental training capability.
        """
        self.tokenizer_length = tokenizer_length
        self.vocab_size = vocab_size
        
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.trained = True
        else:
            # Initialize for incremental training
            self.tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            self.word_counts = defaultdict(int)
            self.special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
            self.trained = False

    def train(self, words):
        """Accumulate word counts for training"""
        if isinstance(words, str):
            words = words.split()
            
        for word in words:
            self.word_counts[word] += 1

    def _finalize_training(self):
        """Finalize the tokenizer after accumulating all words"""
        if self.trained:
            return
            
        # Sort by frequency and take top words
        sorted_words = sorted(self.word_counts.items(), 
                            key=lambda x: (-x[1], x[0]))
        
        # Keep most frequent words
        vocab = [word for word, count in sorted_words][:self.vocab_size - len(self.special_tokens)]
        
        # Add special tokens
        vocab = self.special_tokens + vocab
        
        # Create the final vocabulary
        self.tokenizer.model = models.WordLevel(
            vocab={word: i for i, word in enumerate(vocab)},
            unk_token="[UNK]"
        )
        
        # Initialize trainer (needed for save/load compatibility)
        self.tokenizer.trainer = trainers.WordLevelTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens
        )
        
        self.trained = True
        del self.word_counts  # Free memory

    def pads(self, ids):
        """Pads the token IDs to a specified maximum length."""
        if len(ids) < self.tokenizer_length:
            ids += [0] * (self.tokenizer_length - len(ids))
        elif len(ids) > self.tokenizer_length:
            ids = ids[:self.tokenizer_length]
        return ids[:self.tokenizer_length]

    def transform(self, text):
        """Tokenizes and encodes the text into token IDs."""
        if not self.trained:
            self._finalize_training()
        return self.pads(self.tokenizer.encode(text).ids)
        
    def batch_transform(self, texts):
        """Tokenizes and encodes a batch of texts into token IDs."""
        if not self.trained:
            self._finalize_training()
        return [self.transform(text) for text in texts]

    def decode(self, ids):
        """Decodes a sequence of token IDs back into text."""
        return self.tokenizer.decode(ids)

    def get_vocab_size(self):
        """Returns the vocabulary size."""
        if not self.trained:
            self._finalize_training()
        return len(self.tokenizer.get_vocab())

    def get_vocab(self):
        """Returns the vocabulary dictionary."""
        if not self.trained:
            self._finalize_training()
        return self.tokenizer.get_vocab()

    def save(self, path):
        """Saves the tokenizer to a file."""
        if not self.trained:
            self._finalize_training()
        self.tokenizer.save(path)
