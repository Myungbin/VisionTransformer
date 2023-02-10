import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


class PatchEmbedding(nn.Module):
    def __init__(self, channels=3, patch_size=16, embedding_size=768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * channels, embedding_size)
        )

    def forward(self, x):
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size=768, num_heads=8, drop_out=0):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.keys = nn.Linear(embedding_size, embedding_size)
        self.queries = nn.Linear(embedding_size, embedding_size)
        self.values = nn.Linear(embedding_size, embedding_size)
        self.attention_drop = nn.Dropout(drop_out)
        self.projection = nn.Linear(embedding_size, embedding_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d",
                            h=self.num_heads)  # BATCH, HEADS, SEQUENCE_LEN, EMBEDDING_SIZE
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d",
                         h=self.num_heads)  # BATCH, HEADS, SEQUENCE_LEN, EMBEDDING_SIZE
        values = rearrange(self.values(x), "b n (h d) -> b h n d",
                           h=self.num_heads)  # BATCH, HEADS, SEQUENCE_LEN, EMBEDDING_SIZE

        sum_product = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # (batch, heads, query, key)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            sum_product.mask_fill(~mask, fill_value)

        scaling = self.embedding_size ** (1 / 2)
        attention = F.softmax(sum_product, dim=1) / scaling
        attention = self.attention_drop(attention)

        out = torch.einsum('bhal, bhlv -> bhav ', attention, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        residual = x
        x = self.fn(x, **kwargs)
        x += residual
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, embedding_size, expansion=4, drop_out=0):
        super().__init__(
            nn.Linear(embedding_size, expansion * embedding_size),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(expansion * embedding_size, embedding_size)
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, embedding_size=768, drop_out=0, forward_expansion=4, forward_drop_out=0, **kwargs):
        super().__init__(
            ResidualBlock(nn.Sequential(
                nn.LayerNorm(embedding_size),
                MultiHeadAttention(embedding_size, **kwargs),
                nn.Dropout(drop_out)
            )),
            ResidualBlock(nn.Sequential(
                nn.LayerNorm(embedding_size),
                FeedForwardBlock(
                    embedding_size, expansion=forward_expansion, drop_out=forward_drop_out
                )
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))


class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 depth: int = 12,
                 n_classes: int = 2,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size),
            TransformerEncoder(depth, embedding_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )


summary(ViT(), (3, 256, 256), device='cpu')
