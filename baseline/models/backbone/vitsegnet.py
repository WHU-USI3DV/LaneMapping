'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
* thanks to https://github.com/lucidrains/vit-pytorch
'''
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from baseline.models.registry import BACKBONE

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

@BACKBONE.register_module
class VitSegNet(nn.Module):
    def __init__(self,
                image_size=144,
                patch_h_size=8,
                patch_w_size=8,
                channels=64,
                dim=512,
                depth=5,
                heads=16,
                output_channels=1024,
                expansion_factor=4,
                dim_head=64,
                dropout=0.,
                emb_dropout=0.,
                is_with_shared_mlp=True,
                is_with_llm=False,
                cfg=None):  # mlp_dim is corresponding to expansion factor
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair((patch_h_size, patch_w_size))

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  #
        patch_dim = channels * patch_height * patch_width
        temp_h = int(image_size / patch_h_size)
        temp_w = int(image_size / patch_w_size)
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # h: patch number in height axis; w: patch number in width axis
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        # Without cls token
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout = nn.Dropout(emb_dropout)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        mlp_dim = int(dim*expansion_factor)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.rearrange = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = temp_h, w=temp_w, p1 = patch_h_size, p2 = patch_w_size)
        
        out_in_channels = int(dim/(patch_h_size*patch_w_size))

        
        if is_with_shared_mlp:
            self.is_with_shared_mlp = True
            self.shared_mlp = nn.Conv2d(in_channels=out_in_channels, out_channels=output_channels, kernel_size=1)
        else:
            self.is_with_shared_mlp = False

        


    def forward(self, img):
        x = self.to_patch_embedding(img)
        _, n, _ = x.shape  # torch.Size([4, 324, 512])  [batch_size, patch_number, patch_channels]

        # Without cls token
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        
        x += self.pos_embedding[:, :n]  # we embedded [patch_number] positions, then concat it to the image features
        
        x = self.dropout(x)
        x = self.transformer(x)   # x.shape = torch.Size([4, 324, 512]) # after transformer
        
        x = self.rearrange(x)     # x.shape = torch.Size([4, 8, 144, 144])  # after rearrange

        if self.is_with_shared_mlp:
            x = self.shared_mlp(x)
            # print(f'4: {x.shape}')  # torch.Size([4, 8, 144, 144])
       
        return x

if __name__ == '__main__':
    v = VitSegNet()

    img = torch.randn(1, 64, 144, 144)

    preds = v(img)

    print(preds.shape)

    # v = ViT(
    #     image_size = 256,
    #     patch_size = 32,
    #     num_classes = 1000,
    #     dim = 1024,
    #     depth = 6,
    #     heads = 16,
    #     mlp_dim = 2048,
    #     dropout = 0.1,
    #     emb_dropout = 0.1
    # )

    # img = torch.randn(1, 3, 256, 256)

    # preds = v(img) # (1, 1000)
    # print(preds.shape)
