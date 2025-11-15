import torch
from torch import nn
from trans import Transformer,PatchEmbed,SpectralPosEncoder
class FineTuneModel(nn.Module):
    def __init__(self, spectrum_model, hidden_dim,n,embed_dim, dropout_rate=0.1):
        super().__init__()
        # 定义权重映射
        if hidden_dim == 0:
            self.weight_net = nn.Sequential(nn.Linear(embed_dim, 1)) 
             # 定义网络头
            self.predict_net = nn.Sequential(nn.Linear(embed_dim, 1)) 
        else:
            def create_layers():
                layers = []
                prev_size = embed_dim
                for hidden_size in hidden_dim:
                    layers.append(nn.Linear(prev_size, hidden_size))
                    layers.append(nn.ReLU())
                    prev_size = hidden_size
                layers.append(nn.Linear(prev_size, 1))
                return layers
                
            self.weight_net = nn.Sequential(*create_layers())
            self.predict_net = nn.Sequential(*create_layers())
            
        self.softmax = nn.Softmax(dim=1)
       
        # 定义生成每样本独立bias的网络
        # self.bias = nn.Parameter(torch.randn(1))
        # self.bias_net = nn.Linear(embed_dim, 1)  # 新增层，生成样本特定的bias

        # 定义编码器
        self.spectrum_model = spectrum_model

        # 冻结所有参数（先全部锁定）
        for param in self.spectrum_model.parameters():
            param.requires_grad = False

        # 只解冻最后两层
        if n is not None:
            self._unfreeze_last_n_layers(self.spectrum_model, n=n)

        # 只对 self.head 和 self.norm 进行初始化
        self._init_head_and_norm()

    def _unfreeze_last_n_layers(self, encoder, n=2):
        """解冻最后n层"""
        for layer in encoder.encoder.layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
                
        encoder.encoder_norm.weight.requires_grad = True
        encoder.encoder_norm.bias.requires_grad = True

    def _init_head_and_norm(self):
        """只初始化 self.head 中的线性层和 self.norm"""
        # 初始化 self.head 中的 nn.Linear
        for module in self.weight_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.01)  # 小正值避免死神经元
        for module in self.predict_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.01)  # 小正值避免死神经元

        # 初始化 bias_base
        # nn.init.normal_(self.bias, mean=0.0, std=0.01)  # 小方差正态分布bias
        # nn.init.xavier_normal_(self.bias_net.weight)
        # nn.init.constant_(self.bias_net.bias, 0.01)  # 小正值避免死神经元

    def encoder(self, data):
        merged_spectrum = self.spectrum_model.forward_encoder_no_mask([data[:,:,:300], data[:,:,300:]])
        return merged_spectrum
    
    def forward(self, data):
        # encoding
        merged_spectrum = self.encoder(data)
        # weight extraction
        alpha = self.weight_net(merged_spectrum)
        alpha = self.softmax(alpha)
        # prediction map
        pred = self.predict_net(merged_spectrum)
        # 生成每样本独立的bias
        # bias = self.bias_net(merged_spectrum.mean(dim=1))  # 对序列维度取均值，生成 (batch_size, 1) 的bias
        # weighted sum
        out = torch.sum(alpha*pred,axis=1) #+ bias
        return out, [alpha, pred]
        
class Permute(nn.Module):
    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)
        
class MAE(nn.Module):
    def __init__(
            self, in_channel, embed_dim, decoder_dim, patch_size, stride, num_patches, # embed_dim：编码器输入维度 decoder_dim:解码器输入维度
            mask_ratio=[0.5,0.4], encoder_depth=2, decoder_depth=1, mlp_ratio=4, qkv_bias=False,
            num_encoder_heads=4, num_decoder_heads=4, device='cpu'
    ):
        super().__init__()

        self.abs_patch_size, self.pl_patch_size = patch_size
        self.abs_stride, self.pl_stride = stride
        self.abs_num_patches, self.pl_num_patches = num_patches
        self.num_patches = sum(num_patches)
        self.in_channel = in_channel
        self.embed_dim = embed_dim

        # Encoder
        self.encoder = Transformer(
            embed_dim,
            embed_dim * mlp_ratio,
            depth=encoder_depth,
            num_heads=num_encoder_heads,
            dim_per_head=embed_dim // num_encoder_heads,
            qkv_bias=qkv_bias
        )

        self.enc_to_dec = nn.Linear(embed_dim, decoder_dim) if embed_dim != decoder_dim else nn.Identity()

        self.decoder = Transformer(
            decoder_dim,
            decoder_dim * mlp_ratio,
            depth=decoder_depth,
            num_heads=num_decoder_heads,
            dim_per_head=decoder_dim // num_decoder_heads,
            qkv_bias=qkv_bias
        )

        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        # self.pos_norm = nn.LayerNorm(embed_dim)

        # Prediction head 输出的维度数等于1个 patch 的像素值数量
        self.decoder_abs_pred = nn.Linear(decoder_dim, self.abs_patch_size * self.in_channel)
        self.decoder_pl_pred = nn.Linear(decoder_dim, self.pl_patch_size * self.in_channel)

        # patch_embedding
        self.abs_patch_embed = PatchEmbed(self.in_channel, self.abs_patch_size, self.abs_stride, embed_dim)
        self.pl_patch_embed = PatchEmbed(self.in_channel, self.pl_patch_size, self.pl_stride, embed_dim)

        self.abs_pos_patch_embed = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.SiLU(),
            Permute(),
            PatchEmbed(2*embed_dim, self.abs_patch_size, self.abs_stride, embed_dim)
        )
        self.pl_pos_patch_embed = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.SiLU(),
            Permute(),
            PatchEmbed(2*embed_dim, self.pl_patch_size, self.pl_stride, embed_dim)
        )

        # pos embedding
        self.encoder_pos_embed = nn.Embedding(self.num_patches, embed_dim, device=device)
        self.decoder_pos_embed = nn.Embedding(self.num_patches, decoder_dim, device=device)

        self.mask_ratio = mask_ratio
        # mask token 的实质：1个可学习的共享向量
        self.mask_embed = nn.Parameter(torch.randn(1, 1, decoder_dim))
        # initialize
        torch.nn.init.normal_(self.mask_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):  # 新增
            torch.nn.init.xavier_uniform_(m.weight)  # 或xavier_uniform_
        elif isinstance(m, nn.Conv1d):  # 假设PatchEmbed用的是Conv1d
            # 对Conv1d层的权重使用Xavier初始化
            w = m.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def get_positional_encoding(self, sequence, dim, device):
        """
        Generate sinusoidal positional encoding for a given sequence.
        
        Args:
            sequence (torch.Tensor): Input sequence tensor of shape [..., seq_len].
            dim (int): Embedding dimension (must be even).
            device (torch.device): Device to place the output tensor.
        
        Returns:
            torch.Tensor: Positional encoding tensor of shape [..., seq_len, dim].
        """
        assert dim % 2 == 0, "Embedding dimension must be even"
        
        # Calculate frequency terms
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        div_term = div_term.to(device)  # Shape: [dim/2]
        
        # Initialize positional encoding tensor
        pe = torch.zeros(*sequence.shape, dim, device=device)
        
        # Apply sin and cos to even and odd indices
        pe[..., 0::2] = torch.sin(sequence[..., None] * div_term)
        pe[..., 1::2] = torch.cos(sequence[..., None] * div_term)
        
        return pe
    
    # def random_masking(self, x):
    #     """
    #     Perform per-sample random masking by per-sample shuffling.
    #     Per-sample shuffling is done by argsort random noise.
    #     x: [N, L, D], sequence
    #     """
    #     N, L, D = x.shape  # batch, length, dim
    #     len_keep = int(L * (1 - self.mask_ratio))

    #     noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    #     # sort noise for each sample
    #     # 只对 [k:] 的噪声进行排序
    #     k=3
    #     noise_rest = noise[:, k:]  # 形状 [N, L-k]
    #     ids_shuffle_rest = torch.argsort(noise_rest, dim=1)  # 形状 [N, L-k]
    
    #     # 构造完整的排序索引：前 k 个固定为 0,1,...,k-1，其余为 k 之后的排序
    #     ids_fixed = torch.arange(k, device=x.device).unsqueeze(0).repeat(N, 1)  # 形状 [N, k]
    #     ids_shuffle = torch.cat([ids_fixed, ids_shuffle_rest + k], dim=1)  # 形状 [N, L]
    
    #     # 生成恢复原始顺序的索引
    #     ids_restore = torch.argsort(ids_shuffle, dim=1)  # 形状 [N, L]

    #     # keep the first subset
    #     ids_keep = ids_shuffle[:, :len_keep]
    #     x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    #     # generate the binary mask: 0 is keep, 1 is remove
    #     mask = torch.ones([N, L], device=x.device)
    #     mask[:, :len_keep] = 0
    #     # unshuffle to get the binary mask
    #     mask = torch.gather(mask, dim=1, index=ids_restore)

    #     return x_masked, mask, ids_restore
    def random_masking(self, x, N, k, m1, m2):
        """
        Perform per-sample random masking with different ratios for first N tokens and the rest,
        keeping the first k tokens unmasked.
        x: [N_batch, L, D], sequence
        N: number of initial tokens to consider for m1 masking
        k: number of initial tokens to keep unmasked
        m1: masking ratio for tokens from k to N
        m2: masking ratio for tokens after N
        """
        N_batch, L, D = x.shape  # batch, length, dim
        len_keep_1 = k + int((N - k) * (1 - m1))  # Keep all k + (1-m1) of N-k tokens
        len_keep_2 = int((L - N) * (1 - m2))  # Number of tokens to keep after N
        len_keep = len_keep_1 + len_keep_2  # Total tokens to keep
    
        noise = torch.rand(N_batch, L, device=x.device)  # noise in [0, 1]
    
        # Split noise: first k (unmasked), k to N (m1 masking), rest (m2 masking)
        noise_mid = noise[:, k:N]  # Shape [N_batch, N-k]
        noise_rest = noise[:, N:]  # Shape [N_batch, L-N]
    
        # Sort noise for maskable segments
        ids_shuffle_mid = torch.argsort(noise_mid, dim=1) + k  # Shape [N_batch, N-k], offset by k
        ids_shuffle_rest = torch.argsort(noise_rest, dim=1) + N  # Shape [N_batch, L-N], offset by N
    
        # Keep indices: first k, top (N-k)*(1-m1) from k to N, top (L-N)*(1-m2) from rest
        ids_keep_first = torch.arange(k, device=x.device).unsqueeze(0).repeat(N_batch, 1)  # Shape [N_batch, k]
        ids_keep_mid = ids_shuffle_mid[:, :int((N - k) * (1 - m1))]  # Shape [N_batch, (N-k)*(1-m1)]
        ids_keep_rest = ids_shuffle_rest[:, :len_keep_2]  # Shape [N_batch, len_keep_2]
    
        # Combine kept indices
        ids_keep = torch.cat([ids_keep_first, ids_keep_mid, ids_keep_rest], dim=1)  # Shape [N_batch, len_keep]
    
        # Shuffle ids_keep to maintain randomness
        shuffle_indices = torch.randperm(len_keep, device=x.device)
        ids_keep = ids_keep[:, shuffle_indices]
    
        # Gather masked tokens
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    
        # Generate full shuffle indices for restoration
        ids_shuffle_first = torch.arange(k, device=x.device).unsqueeze(0).repeat(N_batch, 1)  # Shape [N_batch, k]
        ids_shuffle = torch.cat([ids_shuffle_first, ids_shuffle_mid, ids_shuffle_rest], dim=1)  # Shape [N_batch, L]
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # Shape [N_batch, L]
    
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N_batch, L], device=x.device)
        mask[:, :k] = 0  # First k tokens kept
        mask[:, k:k + int((N - k) * (1 - m1))] = 0  # Kept tokens from k to N
        mask[:, N:N + len_keep_2] = 0  # Kept tokens after N
        mask = torch.gather(mask, dim=1, index=ids_restore)
    
        return x_masked, mask, ids_restore

    def forward_encoder(self, spectrum):
        Abs_wave, pl_wave = spectrum
        device = Abs_wave.device

        # pos_embedding
        abs_pos_emb = self.get_positional_encoding(Abs_wave[:,1,:], self.embed_dim, device=device)
        pl_pos_emb = self.get_positional_encoding(pl_wave[:,1,:], self.embed_dim, device=device)

        abs_patch_pos_emb = self.abs_pos_patch_embed(abs_pos_emb) # [batch, embed, patch]
        pl_patch_pos_emb = self.pl_pos_patch_embed(pl_pos_emb)
        
        # embed patches
        Abs_token = self.abs_patch_embed(Abs_wave[:,0:1,:])
        pl_token = self.pl_patch_embed(pl_wave[:,0:1,:])

        # add pos embedding
        Abs_token += abs_patch_pos_emb # [batch, patch, embed]
        pl_token += pl_patch_pos_emb

        spectrum_token = torch.cat((Abs_token, pl_token),axis=1)
        # print(spectrum_token.shape)
        # print(spectrum_token.shape)
        # add pos embed
        # abs_patch_wavelength = self.get_patch_wavelengths(self.wavelength_x[0],self.wavelength_x[-1],self.patch_size,self.num_patches)
        x = spectrum_token + self.encoder_pos_embed((torch.arange(self.num_patches, device=device)))
        # x = self.pos_norm(x)
        # masking: length -> length * mask_ratio
        masked_x, mask, ids_restore = self.random_masking(x, self.abs_num_patches, 0, self.mask_ratio[0], self.mask_ratio[1])

        # apply Transformer blocks
        embedding_x = self.encoder(masked_x)

        embedding_norm_x = self.encoder_norm(embedding_x)

        return embedding_norm_x, mask, ids_restore

    def forward_encoder_no_mask(self, spectrum):
        # embed patches
        Abs_wave, pl_wave = spectrum
        device = Abs_wave.device

        # pos_embedding
        abs_pos_emb = self.get_positional_encoding(Abs_wave[:,1,:], self.embed_dim, device=device)
        pl_pos_emb = self.get_positional_encoding(pl_wave[:,1,:], self.embed_dim, device=device)

        abs_patch_pos_emb = self.abs_pos_patch_embed(abs_pos_emb) # [batch, patch, embed]
        pl_patch_pos_emb = self.pl_pos_patch_embed(pl_pos_emb)
        
        # embed patches
        Abs_token = self.abs_patch_embed(Abs_wave[:,0:1,:])
        pl_token = self.pl_patch_embed(pl_wave[:,0:1,:])

        # add pos embedding
        Abs_token += abs_patch_pos_emb # [batch, patch, embed]
        pl_token += pl_patch_pos_emb

        spectrum_token = torch.cat((Abs_token, pl_token), axis=1)

        # add pos embed
        x = spectrum_token + self.encoder_pos_embed((torch.arange(self.num_patches, device=device)))
        # x = self.pos_norm(x)
        # apply Transformer blocks
        embedding_x = self.encoder(x)
        embedding_norm_x = self.encoder_norm(embedding_x)

        return embedding_norm_x

    def forward_decoder(self, x, ids_restore):

        N, L, D = x.shape
        # embed tokens
        x = self.enc_to_dec(x)
        decode_dim = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_embed.repeat(N, ids_restore.shape[1] - L, 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, decode_dim))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed((torch.arange(self.num_patches, device=x.device)))

        # apply Transformer blocks
        x = self.decoder(x)
        x = self.decoder_norm(x)

        # predictor projection
        abs_pred = self.decoder_abs_pred(x[:,:self.abs_num_patches,:])
        pl_pred = self.decoder_pl_pred(x[:, self.abs_num_patches:, :])

        return abs_pred, pl_pred

    def forward(self,  spectrum):
        encode_x, mask, ids_restore = self.forward_encoder(spectrum)
        abs_pred, pl_pred = self.forward_decoder(encode_x, ids_restore)

        return abs_pred, pl_pred

    def patchify(self, spectrum,patch_size):
        """
        spectrum: (batch_size, channel, length)
        x: (batch_size, num_patches, patch_size)
        """
        batch_size = spectrum.shape[0]
        w = spectrum.shape[-1] // patch_size
        # patchify spectrum
        x = spectrum.reshape(batch_size, spectrum.shape[1], w, patch_size)
        x = torch.einsum('nclp->nlpc', x)
        x = x.reshape(batch_size, w, -1)

        return x

    def unpatchify(self, x, channel):
        """
        x: (batch_size, num_patches, patch_size)
        spectrum: (batch_size, channel, length)
        """
        patch_size = x.shape[2]
        num_patches = x.shape[1]
        batch_size = x.shape[0]

        # patchify spectrum
        spectrum = x.reshape(batch_size, num_patches, patch_size//channel, channel)
        spectrum = torch.einsum('nlpc->nclp', spectrum)
        spectrum = spectrum.reshape(batch_size, channel, -1)
        return spectrum

    def loss_cal(self, pred, target, mask, norm_spec_loss=False):
        abs_pred, pl_pred = pred
        Abs_target, pl_target = target

        def norm(target):
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
            return target

        if norm_spec_loss:
            Abs_target = norm(Abs_target)
            pl_target = norm(pl_target)

        Abs_loss = ((abs_pred - Abs_target) ** 2).mean(dim=-1)
        pl_loss = ((pl_pred - pl_target) ** 2).mean(dim=-1)
        loss = torch.cat((Abs_loss,pl_loss),axis=1)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

    def forward_loss(self, spectrum, norm_spec_loss=False):
        """
        spectrum: [batch_size, channel, length]
        pred: [batch_size, num_patches, patch_size]
        mask: [batch_size, num_patches], 0 is keep, 1 is remove,
        """
        encode_x, mask, ids_restore = self.forward_encoder(spectrum)
        abs_pred_tokens, pl_pred_tokens = self.forward_decoder(encode_x, ids_restore)
 
        Abs_wave, pl_wave = spectrum
        Abs = Abs_wave[:,0:1,:]
        pl = pl_wave[:,0:1,:]
        Abs_target = self.patchify(Abs,self.abs_patch_size)
        pl_target = self.patchify(pl, self.pl_patch_size)

        # print(abs_pred_tokens.shape, Abs_target.shape)
        # cal loss
        loss = self.loss_cal([abs_pred_tokens, pl_pred_tokens], [Abs_target, pl_target], mask, norm_spec_loss)

        abs_pred = self.unpatchify(abs_pred_tokens * mask.unsqueeze(-1)[:,:self.abs_num_patches],self.in_channel)
        pl_pred = self.unpatchify(pl_pred_tokens * mask.unsqueeze(-1)[:,self.abs_num_patches:],self.in_channel)

        def norm(target):
            if norm_spec_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6) ** .5
            return target

        # abs_true = self.unpatchify(norm(Abs_target) * mask.unsqueeze(-1)[:,:self.abs_num_patches],self.in_channel)
        # pl_true = self.unpatchify(norm(pl_target) * mask.unsqueeze(-1)[:,self.abs_num_patches:],self.in_channel)
        abs_true = self.unpatchify(norm(Abs_target),self.in_channel)
        pl_true = self.unpatchify(norm(pl_target),self.in_channel)

        return loss, [abs_true, pl_true], [abs_pred, pl_pred], mask

#     def reconstruction(self, spectrum, pred, mask, norm_spec_loss=False):
#         if norm_spec_loss:
    
#     def reconstruction(self, spectrum, pred, mask, norm_spec_loss=False):
#         if norm_spec_loss:
           
            


# model = MAE(10,10)
#
# spectrum = torch.ones(2,151)
#
# encode_x, mask, ids_restore = model.forward_encoder(spectrum)
#
# print(encode_x.shape)
#
# pred = model.forward_decoder(encode_x, ids_restore)
# print(pred.shape)
#
# loss = model.forward_loss(spectrum, pred, mask)
# print(loss)