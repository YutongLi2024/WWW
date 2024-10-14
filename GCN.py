import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.matmul(adj, x)
        return x

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (N, Batchsize, latent_dim)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2)  # (Batchsize, N, latent_dim)
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x, cond):
        x = x.permute(1, 0, 2)  # (N, Batchsize, latent_dim)
        cond = cond.permute(1, 0, 2)  # (N, Batchsize, latent_dim)
        x, _ = self.attention(x, cond, cond)
        x = x.permute(1, 0, 2)  # (Batchsize, N, latent_dim)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(EncoderBlock, self).__init__()
        self.gcn = GCNLayer(in_features, out_features)
        self.self_attention = AttentionLayer(out_features, num_heads)
        self.cross_attention = CrossAttentionLayer(out_features, num_heads)
        self.downsample = nn.Linear(out_features, out_features)

    def forward(self, x, adj, cond):
        residual = x
        x = self.gcn(x, adj)
        x = self.self_attention(x)
        x = self.cross_attention(x, cond)
        x = self.downsample(x)
        x = x + residual
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(DecoderBlock, self).__init__()
        self.gcn = GCNLayer(in_features, out_features)
        self.self_attention = AttentionLayer(out_features, num_heads)
        self.cross_attention = CrossAttentionLayer(out_features, num_heads)
        self.upsample = nn.Linear(out_features, out_features)

    def forward(self, x, adj, cond):
        # residual = x
        x = self.gcn(x, adj)
        x = self.self_attention(x)
        x = self.cross_attention(x, cond)
        x = self.upsample(x)
        # x = x + residual
        return x

class UNet(nn.Module):
    def __init__(self, in_features, out_features, num_heads, num_layers):
        super(UNet, self).__init__()
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Encoder
        for i in range(num_layers):
            self.encoder_blocks.append(EncoderBlock(in_features, out_features, num_heads))
            in_features = out_features

        # Decoder
        for i in range(num_layers):
            self.decoder_blocks.append(DecoderBlock(in_features, out_features, num_heads))
            in_features = out_features

    def forward(self, hidden, H, s_c):

        H = H[:, :, :H.shape[1]]
        
        H = torch.matmul(H, H.transpose(1, 2))
        H = (H > 0).float()

        encoder_outputs = []

        # Encoder
        for block in self.encoder_blocks:
            hidden = block(hidden, H, s_c)
            encoder_outputs.append(hidden)

        for encoder_output in encoder_outputs:
            hidden = hidden + encoder_output 
            
        # GCN
        hidden = torch.matmul(H, hidden)

        return hidden

