from torch import nn
import torch

class PreNorm(nn.Module):
    def __init__(self, dim, net):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.net = net

    def forward(self, x, **kwargs):
        return self.net(self.norm(x), **kwargs)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_per_head=64, qkv_bias=False):
        super().__init__()

        self.num_heads = num_heads
        self.scale = dim_per_head ** -0.5

        inner_dim = dim_per_head * num_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)

        self.attend = nn.Softmax(dim=-1)

        project_out = not (num_heads == 1 and dim_per_head == dim)
        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, l, d = x.shape

        '''i. QKV projection'''
        # (b,l,dim_all_heads x 3)
        qkv = self.to_qkv(x)
        # (3,b,num_heads,l,dim_per_head)
        qkv = qkv.view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # 3 x (1,b,num_heads,l,dim_per_head)
        q, k, v = qkv.chunk(3)
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        '''ii. Attention computation'''
        attn = self.attend(
            torch.matmul(q, k.transpose(-1, -2)) * self.scale
        )

        '''iii. Put attention on Value & reshape'''
        # (b,num_heads,l,dim_per_head)
        z = torch.matmul(attn, v)
        # (b,num_heads,l,dim_per_head)->(b,l,num_heads,dim_per_head)->(b,l,dim_all_heads)
        z = z.transpose(1, 2).reshape(b, l, -1)
        # assert z.size(-1) == q.size(-1) * self.num_heads

        '''iv. Project out'''
        # (b,l,dim_all_heads)->(b,l,dim)
        out = self.out(z)
        # assert out.size(-1) == d

        return out


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, mlp_dim, depth=6, num_heads=8, dim_per_head=64,qkv_bias=False):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, num_heads=num_heads, dim_per_head=dim_per_head,qkv_bias = qkv_bias)),
                PreNorm(dim, FFN(dim, mlp_dim))
            ]))

    def forward(self, x):
        for norm_attn, norm_ffn in self.layers:
            x = x + norm_attn(x)
            x = x + norm_ffn(x)

        return x


class PatchEmbed(nn.Module):
    """ Patch Embedding
    """

    def __init__(self, in_channels, patch_size=5, stride=2, embed_dim=10):
        super().__init__()

        self.in_channels = in_channels

        self.proj = nn.Conv1d(self.in_channels, embed_dim, kernel_size=patch_size, stride=stride)  # with overlapped patches


    def forward(self, x):
        # x = torch.unsqueeze(x, 1)
        x = self.proj(x)  # 32, 3, 151 -> 32, 10, 74
        x = x.transpose(1, 2)  # 32, 10, 74 -> 32, 74, 10
        return x

class WavelengthPositionalEncoding(nn.Module):
    def __init__(self, d_model, wavelength_scale=10):
        super(WavelengthPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.wavelength_scale = wavelength_scale

    def get_wavelength_pe(self, wavelengths):
        """
        Generate wavelength-aware positional encoding.
        wavelengths: (batch_size, seq_len), normalized wavelengths in [0, 1]
        Returns: pe (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = wavelengths.size()
        pe = torch.zeros(batch_size, seq_len, self.d_model, device=wavelengths.device)

        # Create div_term for positional encoding
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        div_term = div_term.view(1, 1, -1)  # (1, 1, d_model//2)

        # Incorporate normalized wavelengths into positional encoding
        wavelengths = wavelengths.unsqueeze(-1) / self.wavelength_scale # (batch_size, seq_len, 1)
        for i in range(self.d_model // 2):
            # Compute sin and cos, and squeeze the last dimension to match pe slice
            pe[:, :, 2*i] = torch.sin(wavelengths * div_term[:, :, i]).squeeze(-1)  # Shape: (batch_size, seq_len)
            pe[:, :, 2*i+1] = torch.cos(wavelengths * div_term[:, :, i]).squeeze(-1)  # Shape: (batch_size, seq_len)
        return pe
        
class SpectralPosEncoder(nn.Module):
    def __init__(self, wavelengths, patch_size, dim, min_wl=400, max_wl=1000):
        super().__init__()
        self.min_wl = min_wl
        self.max_wl = max_wl
        self.dim = dim
        self.wavelengths = wavelengths
        self.patch_size = patch_size
        self.wavelength_patch = None
        self.device = wavelengths.device

    def forward(self):
        # wavelengths: [num_patches]，每个patch的平均波长
        self.wavelength_patch = self.patchify_wavelengths()
        norm_wl = (self.wavelength_patch - self.min_wl) / (self.max_wl - self.min_wl) * 10000

        pos = norm_wl.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.dim)).to(self.device)
        pe = torch.zeros(*pos.shape[:-1], self.dim).to(self.device)
        pe[..., 0::2] = torch.sin(pos * div_term)
        pe[..., 1::2] = torch.cos(pos * div_term)
        return pe

    def patchify_wavelengths(self):
        """
        将波长序列分块并计算每个块的平均波长值。

        参数：
            wavelengths: torch.Tensor, 波长序列，形状为 [batch_size, seq_len]
            patch_size: int, 每个块的大小（点的数量）

        返回：
            patched_wl: torch.Tensor, 每个块的平均波长值，形状为 [batch_size, num_patches]
        """
        seq_len = self.wavelengths.shape[0]
        num_patches = (seq_len + self.patch_size - 1) // self.patch_size  # 上取整，确保覆盖所有点

        padded_wl = torch.Tensor(self.wavelengths)

        # 重塑为 [batch_size, num_patches, patch_size]
        patched_wl = padded_wl.view(num_patches, self.patch_size)

        # 计算每个块的平均波长（忽略填充部分的零值）
        mask = (patched_wl != 0).float()  # 标记非填充部分
        sum_wl = patched_wl.sum(dim=-1)  # 每个块的波长和
        count_wl = mask.sum(dim=-1)  # 每个块的有效点数
        mean_wl = sum_wl / count_wl.clamp(min=1)  # 避免除以零，计算平均值

        return mean_wl