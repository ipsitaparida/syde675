from cProfile import label
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from utils import problem


@problem.tag("hw4-B", start_line=1)
def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a batch of data given a list of N sequences and labels. Sequences are stacked into a single tensor
    of shape (N, max_sequence_length), where max_sequence_length is the maximum length of any sequence in the
    batch. Sequences shorter than this length should be filled up with 0's. Also returns a tensor of shape (N, 1)
    containing the label of each sequence.

    Args:
        batch ([type]): A list of size N, where each element is a tuple containing a sequence LongTensor and a single item
            LongTensor containing the true label of the sequence.

    Returns:
        Tuple[torch.LongTensor, torch.LongTensor]: A tuple containing two tensors (both LongTensor).
            The first tensor has shape (N, max_sequence_length) and contains all sequences.
            Sequences shorter than max_sequence_length are padded with 0s at the end.
            The second tensor has shape (N, 1) and contains all labels.
    """
    sentences, labels = zip(*batch)

    N = len(sentences)
    max_sequence_length = max([len(s) for s in sentences])
    padded_sentences = torch.zeros((N, max_sequence_length), dtype=torch.int32)
    all_labels = torch.zeros((N, 1), dtype=torch.int32)
    for s in range(N):
        n = len(sentences[s])
        padded_sentences[s, 0:n] = sentences[s]
        all_labels[s, 0] = labels[s]

    return (padded_sentences, all_labels)

class RNNBinaryClassificationModel(nn.Module):
    @problem.tag("hw4-B", start_line=10)
    def __init__(self, embedding_matrix: torch.Tensor, rnn_type: str):
        """Create a model with either RNN, LSTM or GRU layer followed by a linear layer.

        Args:
            embedding_matrix (torch.Tensor): Weights for embedding layer.
                Used in starter code, nothing you should worry about.
            rnn_type (str): Either "RNN", "LSTM" or "GRU". Defines what kind of a layer should be used.
        """
        super().__init__()

        vocab_size = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]
        
        hidden_size = 64

        # Construct embedding layer and initialize with given embedding matrix. Do not modify this code.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.embedding.weight.data = embedding_matrix

		# Construct 3 different types of RNN for comparison.
        if rnn_type == 'LSTM':
            self.model = nn.LSTM(embedding_dim, hidden_size, 2, batch_first=True)
        if rnn_type == 'RNN':
            self.model = nn.RNN(embedding_dim, hidden_size, 2, batch_first=True)
        if rnn_type == 'GRU':
            self.model = nn.GRU(embedding_dim, hidden_size, 2, batch_first=True)
        
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)

    @problem.tag("hw4-B")
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Takes in a batch of data of shape (N, max_sequence_length). Returns a tensor of shape (N, 1), where each
        element corresponds to the prediction for the corresponding sequence.

        Args:
            inputs (torch.Tensor): FloatTensor of shape (N, max_sequence_length) containing N sequences to make predictions for.

        Returns:
            torch.Tensor: FloatTensor of predictions for each sequence of shape (N, 1).
        """
        inputs = inputs.type(torch.LongTensor)
        embeds = self.embedding(inputs)
        output, hn = self.model(embeds)
        prediction = self.linear(output[:,-1,:])
        return prediction

    @problem.tag("hw4-B")
    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the binary cross-entropy loss.

        Args:
            logits (torch.Tensor): FloatTensor - Raw predictions from the model of shape (N, 1)
            targets (torch.Tensor): LongTensor - True labels of shape (N, 1)

        Returns:
            torch.Tensor: Binary cross entropy loss between logits and targets as a single item FloatTensor.
        """
        binary = torch.sigmoid(logits)
        targets = targets.type(torch.FloatTensor)  
        loss = nn.functional.binary_cross_entropy(binary, targets)
        return loss

    @problem.tag("hw4-B")
    def accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the accuracy, i.e number of correct predictions / N.

        Args:
            logits (torch.Tensor): FloatTensor - Raw predictions from the model of shape (N, 1)
            targets (torch.Tensor): LongTensor - True labels of shape (N, 1)

        Returns:
            torch.Tensor: Accuracy as a scalar FloatTensor.
        """
        binary = torch.sigmoid(logits)
        prediction = torch.round(binary)
        accuracy = sum(prediction == targets).type(torch.FloatTensor)/len(targets)

        return accuracy


@problem.tag("hw4-B", start_line=5)
def get_parameters() -> Dict:
    """Returns parameters for training a model. Is should have 4 entries, with these specific keys:

    {
        "TRAINING_BATCH_SIZE": TRAINING_BATCH_SIZE,  # type: int
        "VAL_BATCH_SIZE": VAL_BATCH_SIZE,  # type: int
        "NUM_EPOCHS": NUM_EPOCHS,  # type: int
        "LEARNING_RATE": LEARNING_RATE,  # type: float
    }

    Returns:
        Dict: Dictionary, as described above.
            (Feel free to copy dict above, and define TRAINING_BATCH_SIZE and LEARNING_RATE)
    """
    # Batch size for validation, this only affects performance.
    VAL_BATCH_SIZE = 128

    # Training parameters
    NUM_EPOCHS = 16

    TRAINING_BATCH_SIZE = 128

    LEARNING_RATE = 0.0001
    return {
        "TRAINING_BATCH_SIZE": TRAINING_BATCH_SIZE,
        "VAL_BATCH_SIZE": VAL_BATCH_SIZE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
    }
