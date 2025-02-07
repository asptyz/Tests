import torch
import torch.nn as nn
import torch.nn.functional as F

# Batch size = 1, Sequence length = 4, Embedding size = 6
X = torch.rand(1, 4, 6)  # (batch_size, seq_len, embedding_dim)

print("Input Embeddings Shape:", X.shape)  # (1, 4, 6)

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads  #for dividing the embedding dimensions for each head.

        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias = False) 
        self.W_k= nn.Linear(embedding_dim, embedding_dim, bias = False)
        self.W_v= nn.Linear(embedding_dim, embedding_dim, bias = False)

        self.W_o = nn.Linear(embedding_dim, embedding_dim, bias = False)

    def forward(self, X):
        batch_size, seq_length, embedding_dim = X.shape

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype = torch.float32))

        attention_weights = F.softmax(scores, dim = -1)

        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1,2).contiguous().view(batch_size, seq_length, embedding_dim)

        output = self.W_o(attention_output)

        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, ffn_hidden_dim):
        super(FeedForwardNetwork, self).__init__()

        self.fc1 = nn.Linear(embedding_dim, ffn_hidden_dim)
        self.fc2 = nn.Linear(ffn_hidden_dim, embedding_dim)

    def forward(self, X):
        return self.fc2(F.relu(self.fc1(X)))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_hidden_dim):
        super(TransformerEncoderBlock, self).__init__()

        self.mha = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = FeedForwardNetwork(embedding_dim, ffn_hidden_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, X):
        attn_output = self.mha(X)
        X = self.norm1(X + attn_output)
        ffn_output = self.ffn(X)
        X = self.norm2( X + ffn_output)

        return X


embedding_dim = 6
num_heads = 2
ffn_hidden_dim = 12
seq_length = 4
batch_size = 1

encoder_block = TransformerEncoderBlock(embedding_dim, num_heads, ffn_hidden_dim)

X = torch.rand(batch_size, seq_length, embedding_dim)

output = encoder_block(X)

print("\nTransformer Encoder output", output.shape)


