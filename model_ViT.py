import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import math
from einops import rearrange, repeat

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size) 1,768,14,14
        x = x.flatten(2)  # (B, embed_dim, num_patches)  1,768,196
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim) 1,196,768
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1 for the class token

    def forward(self, x):
        return x + self.pos_embed


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = PositionalEncoding(embed_dim, num_patches)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)    # 1
        x = self.patch_embed(x) # 1,196,768
        cls_tokens = self.cls_token.expand(B, -1, -1) # 1,1,768
        x = torch.cat((cls_tokens, x), dim=1) # 1,197,768
        x = self.pos_embed(x)  # Add positional encoding 1,197,768

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x) # 1,197,768
        cls_token_final = x[:, 0]   # 1,768
        out = self.head(cls_token_final) # 1,1000
        return out

'''
model = VisionTransformer(img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, dropout=0.1)
input_tensor = torch.randn(17, 3, 224, 224)  # Batch size 1, 3 color channels, 224x224 image
output = model(input_tensor)
print(output.shape)  # Should print torch.Size([1, 1000])
summary(model, (3, 224, 224))
'''