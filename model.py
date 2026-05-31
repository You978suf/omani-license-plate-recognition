import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# valid Omani plate characters for constrained decoding
valid_chars = set("0123456789ABDRSKM WYHLT")


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # stage 1: (B, 3, 64, 200) -> (B, 64, 32, 100)
        self.stage1 = nn.Sequential(
            ConvBNReLU(3, 64),
            nn.MaxPool2d(2, 2)
        )
        # stage 2: (B, 64, 32, 100) -> (B, 128, 16, 50)
        self.stage2 = nn.Sequential(
            ConvBNReLU(64, 128),
            nn.MaxPool2d(2, 2)
        )
        # stage 3: (B, 128, 16, 50) -> (B, 256, 8, 50)
        self.stage3 = nn.Sequential(
            ConvBNReLU(128, 256),
            ConvBNReLU(256, 256),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        # stage 4: (B, 256, 8, 50) -> (B, 256, 4, 50)
        self.stage4 = nn.Sequential(
            ConvBNReLU(256, 256),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        # stage 5: (B, 256, 4, 50) -> (B, 256, 1, 50)
        self.stage5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(4, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 50))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.adaptive_pool(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        T, B, d = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # split into heads: (T, B, d_model) -> (B, heads, T, d_head)
        Q = Q.view(T, B, self.num_heads, self.d_head).permute(1, 2, 0, 3)
        K = K.view(T, B, self.num_heads, self.d_head).permute(1, 2, 0, 3)
        V = V.view(T, B, self.num_heads, self.d_head).permute(1, 2, 0, 3)

        scale = math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, heads, T, d_head)

        # merge heads back: (T, B, d_model)
        out = out.permute(2, 0, 1, 3).contiguous()
        out = out.view(T, B, self.d_model)

        return self.W_out(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x


class CNNTransformer(nn.Module):
    def __init__(self, num_classes, d_model=256, num_heads=4, num_layers=2,
                 ffn_dim=512, dropout=0.1):
        super().__init__()

        self.cnn = CNNBackbone()

        self.input_proj = nn.Linear(256, d_model)

        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=200, dropout=dropout)

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        feat = self.cnn(x)           # (B, 256, 1, 50)
        feat = feat.squeeze(2)       # (B, 256, 50)
        feat = feat.permute(2, 0, 1) # (50, B, 256)
        feat = self.input_proj(feat) # (50, B, d_model)
        feat = self.pos_encoding(feat)

        for layer in self.transformer_layers:
            feat = layer(feat)

        feat = self.norm(feat)
        logits = self.classifier(feat)           # (50, B, num_classes)
        log_probs = F.log_softmax(logits, dim=2)

        return log_probs


def ctc_greedy_decode(log_probs, inv_vocab):
    pred_indices = log_probs.argmax(dim=2).cpu().numpy()  # (T, B)
    T, B = pred_indices.shape
    decoded = []

    for b in range(B):
        chars = []
        prev_idx = -1
        for t in range(T):
            idx = pred_indices[t, b]
            if idx != 0 and idx != prev_idx:
                chars.append(inv_vocab.get(idx, ''))
            prev_idx = idx
        decoded.append(''.join(chars))

    return decoded


def ctc_greedy_decode_constrained(log_probs, inv_vocab):
    pred_indices = log_probs.argmax(dim=2).cpu().numpy()  # (T, B)
    T, B = pred_indices.shape
    decoded = []

    for b in range(B):
        chars = []
        prev_idx = -1
        for t in range(T):
            idx = pred_indices[t, b]
            if idx != 0 and idx != prev_idx:
                ch = inv_vocab.get(idx, '')
                if ch in valid_chars:
                    chars.append(ch)
            prev_idx = idx

        result = ''.join(chars)
        if len(result) > 7:
            result = result[:7]

        decoded.append(result)

    return decoded
