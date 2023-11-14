import esm
import torch
from tqdm import tqdm


class Tokenizer:
    def __init__(self, **kwargs):
        alphabet = kwargs.get('alphabet', 'ESM-1b')
        truncation_seq_length = kwargs.get('truncation_seq_length', None)
        self.alphabet, self.tokenizer = self.init_tokenizer(alphabet, truncation_seq_length)
        self.alphabet_size = len(self.alphabet)
        self.standard_alphabet = list('ACDEFGHIKLMNPQRSTVWY')

    def init_tokenizer(self, alphabet, truncation_seq_length):
        alphabet = esm.data.Alphabet.from_architecture(alphabet)
        batch_converter = alphabet.get_batch_converter(truncation_seq_length)
        return alphabet, batch_converter

    def tokenize(self, sequences):
        if isinstance(sequences, str):
            sequences = [(0, sequences)]
        sequences = [x if isinstance(x, tuple) else (i, x) for i, x in enumerate(sequences)]  # [(index, sequence)]
        batch_ids, batch_seqs, batch_tokens = self.tokenizer(sequences)  # batch_ids: [N], batch_seqs: [N], batch_tokens: [N, L]
        return batch_ids, batch_seqs, batch_tokens

    def decode(self, tokens):
        # tokens: [N, L], a batch of tokenized sequences
        return [''.join(self.alphabet.get_tok(token) for token in seq_token) for seq_token in tqdm(tokens)]

    def one_hot(self, sequences, padding=True):
        if isinstance(sequences, str):
            sequences = [(0, sequences)]

        num_classes = self.alphabet_size
        _, _, batch_tokens = self.tokenize(sequences)
        one_hot_encodings = torch.nn.functional.one_hot(batch_tokens[:, 1:-1], num_classes=num_classes).long()[:, :, 4:24].float()  # [N, L, 20]
        if not padding:
            encodings = []  # [N, seq_len, 20]
            for (_, sequence), one_hot_encoding in zip(sequences, one_hot_encodings):
                one_hot_encoding = one_hot_encoding[:len(sequence)]
                encodings.append(one_hot_encoding)
            one_hot_encodings = encodings
        return one_hot_encodings
