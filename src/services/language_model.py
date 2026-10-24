import torch
from torch import nn
from torch.nn import functional as functional

from src.schemas import TransformerConfig


class Head(nn.Module):
    def __init__(self, config: TransformerConfig, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.causal_mask: torch.Tensor
        self.register_buffer(
            "causal_mask",
            torch.ones(config.block_size, config.block_size, dtype=torch.bool).tril(),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        length = inputs.size(1)
        keys, queries = self.key(inputs), self.query(inputs)
        weights = queries @ keys.transpose(-2, -1) * keys.size(-1) ** -0.5
        weights = weights.masked_fill(
            ~self.causal_mask[:length, :length], float("-inf")
        )
        return self.dropout(functional.softmax(weights, dim=-1)) @ self.value(inputs)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(config, config.n_embd // config.n_head) for _ in range(config.n_head)]
        )
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([head(inputs) for head in self.heads], dim=-1)
        return self.dropout(self.projection(combined))


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.norm2 = nn.LayerNorm(config.n_embd)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.norm1(inputs))
        return inputs + self.feed_forward(self.norm2(inputs))


class TransformerLanguageModel(nn.Module):
    def __init__(self, config: TransformerConfig, vocab_size: int) -> None:
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        length = indices.size(1)
        if length > self.block_size:
            raise ValueError("Длина последовательности превышает размер контекста")
        positions = torch.arange(length, device=indices.device)
        inputs = self.token_embedding_table(indices) + self.position_embedding_table(
            positions
        )
        return self.lm_head(self.norm(self.blocks(inputs)))

    @torch.no_grad()
    def generate(
        self, indices: torch.Tensor, max_new_tokens: int, generator: torch.Generator
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self(indices[:, -self.block_size :])[:, -1, :]
            probabilities = functional.softmax(logits, dim=-1)
            token = torch.multinomial(probabilities, 1, generator=generator)
            indices = torch.cat((indices, token), dim=1)
        return indices
