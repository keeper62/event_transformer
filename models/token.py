import torch
from drain3 import TemplateMiner
from transformers import AutoTokenizer

class LogTokenizer:
    def __init__(self):
        """
        Initializes the tokenizer using Drain3 for log messages.
        :param seq_len: Maximum sequence length for tokenized logs.
        :param special_tokens: List of special tokens.
        """
        self.template_miner = TemplateMiner()
        self.event_id_to_template = {}  # Mapping event IDs to templates
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def transform(self):
        """ Returns a function that encodes log messages into token IDs. """
        def _transform(log_message):
            return self.get_event_id(log_message)
        return _transform

    def train_template_miner(self, log_messages):
        """ Processes logs with Drain3 and stores event templates & event IDs. """
        for log in log_messages:
            result = self.template_miner.add_log_message(log)
            self.event_id_to_template[result["cluster_id"] - 1] = result["template_mined"]
            
    def get_event_id(self, log_message):
        """ Retrieves the event ID for a given log message. """
        result = self.template_miner.match(log_message)
        return result.cluster_id - 1

    def get_vocab_size(self):
        """ Returns the vocabulary size of the Drain3 tokenizer. """
        return len(self.event_id_to_template)

    def get_vocab(self):
        """ Returns the vocabulary of the Drain3 tokenizer. """
        return self.event_id_to_template

    def decode_event_id_sequence(self, event_seq):
        """ Converts tokenized event sequence back into a log template. """
        return self.event_id_to_template.get(event_seq, "<UNK>")
