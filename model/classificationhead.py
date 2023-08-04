import torch.nn as nn
from transformers.activations import get_activation


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, activation_function, num_labels, classifier_dropout, dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        classifier_dropout = (
            classifier_dropout if classifier_dropout is not None else dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        self.activate = activation_function

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation(self.activate)(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
