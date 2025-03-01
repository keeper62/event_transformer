import torch
from drain3 import TemplateMiner
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

class LogTokenizer:
    def __init__(self, vocab_size=100, bpe_max_len=512):
        """
        Initializes the tokenizer with log messages and their classifications.
        :param log_messages: List of log message strings.
        :param classifications: Optional list of corresponding classifications.
        :param vocab_size: Size of the vocabulary for BPE.
        :param bpe_max_len: Maximum length of BPE tokenized sequences.
        """
        self.vocab_size = vocab_size
        self.bpe_max_len = bpe_max_len
        self.template_miner = TemplateMiner()
        self.event_id_to_template = {}  # Mapping of event IDs to templates
        
        # Train BPE Tokenizer on Extracted Templates
        self.tokenizer = self.train_bpe_model()
    
    def train_template_miner(self, log_messages):
        """
        Processes logs with Drain3 and stores event templates & event IDs.
        """
        for log in log_messages:
            result = self.template_miner.add_log_message(log)
            self.event_id_to_template[result["cluster_id"]] = result["template_mined"]
    
    def train_bpe_model(self):
        """
        Trains a BPE model on the extracted event templates.
        """
        event_templates = list(set(self.event_id_to_template.values()))
        
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(special_tokens=["<PAD>", "<UNK>"])
        tokenizer.train_from_iterator(event_templates, trainer)

        return tokenizer

    def train_tokenizer(self, x):
        self.train_template_miner(x)
        self.tokenizer = self.train_bpe_model()
        
    def obtain_token_embedding(self, x):
        event_ids = self.get_event_ids(x)
        return torch.tensor(self.encode_event_id_sequences(event_ids), dtype=torch.long)

    def get_event_id(self, log):
        """
        Retrieves the event ID for a given log message.
        """
        result = self.template_miner.add_log_message(log)
        return result["cluster_id"]

    def encode_event_id_sequences(self, event_id_seq):
        """
        Converts event ID sequences into BPE tokenized event templates.
        :param event_id_seq: List of event IDs.
        :return: Padded and tokenized event sequences.
        """
        tokenized_templates = []
        for event_id in event_id_seq:
            template = self.event_id_to_template.get(event_id, "<UNK>")
            tokenized = self.tokenizer.encode(template).ids[:self.bpe_max_len]  # Tokenize & truncate
            tokenized += [0] * (self.bpe_max_len - len(tokenized))  # Pad
            tokenized_templates.append(tokenized)

        return tokenized_templates
    
    def pad_sequence(self, tokenized_templates, max_len=20):
        # Pad sequences for uniform input
        tokenized_event_sequences = [seq + [[0]] * (max_len - len(seq)) for seq in tokenized_templates]
        tokenized_event_sequences = torch.tensor(tokenized_event_sequences, dtype=torch.long)
        return tokenized_event_sequences
    
    def get_event_ids(self, x):
        return [self.get_event_id(log) for log in x]

    def decode_event_id_sequences(self, predicted_event_ids):
        """
        Converts predicted event ID sequences into human-readable templates.
        :param predicted_event_ids: Tensor of predicted event ID sequences.
        :return: List of decoded log templates.
        """
        decoded_templates = []
        for event_seq in predicted_event_ids:
            templates = [self.event_id_to_template.get(event_id.item(), "<UNKNOWN EVENT>") for event_id in event_seq]
            decoded_templates.append(templates)
        return decoded_templates

    def get_vocab_size(self):
        """
        Returns the vocabulary size of the BPE tokenizer.
        """
        return self.tokenizer.get_vocab_size()
