import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(101)

        # Embeddings
        self.token_embedding = nn.Embedding(2, 2)

        # Block 1
        self.linear_1 = nn.Linear(2, 2)
        self.layernorm_1 = nn.LayerNorm(2)

        # Block 2
        self.linear_2 = nn.Linear(2, 2)
        self.layernorm_2 = nn.LayerNorm(2)

        self.head = nn.Linear(2, 2)

    def forward(self, x):
        hidden_states = self.token_embedding(x)

        # Block 1
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.layernorm_1(hidden_states)

        # Block 2
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.layernorm_2(hidden_states)

        logits = self.head(hidden_states)
        return logits
