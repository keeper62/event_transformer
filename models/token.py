import torch
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from transformers import AutoTokenizer

class LogTokenizer:
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
            self.template_miner.add_log_message(log)
            
    def get_event_id(self, log_message):
        """ Retrieves the event ID for a given log message. """
        result = self.template_miner.match(log_message)
        return result.cluster_id - 1

    def get_vocab_size(self):
        """ Returns the vocabulary size of the Drain3 tokenizer. """
        return self.template_miner.drain.clusters_counter

    def get_vocab(self):
        """ Returns the vocabulary of the Drain3 tokenizer. """
        return self.event_id_to_template

    def decode_event_id_sequence(self, event_seq):
        """ Converts tokenized event sequence back into a log template. """
        return " ".join(self.template_miner.drain.id_to_cluster.get(event_seq, "<UNK>").log_template_tokens)
    
    def load_state(self):
        """ Loads a previously saved state of the tokenizer. """
        self.template_miner.load_state()
        
    def save_state(self):
        """ Saves the current state of the tokenizer. """
        self.template_miner.save_state("For re-use")
