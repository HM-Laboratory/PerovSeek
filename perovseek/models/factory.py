"""Construct the encoder architecture used by the distributed weights."""

from .mae import MAE


def create_encoder(device="cpu"):
    return MAE(
        in_channel=1,
        embed_dim=36,
        decoder_dim=18,
        patch_size=[15, 11],
        stride=[15, 11],
        num_patches=[20, 12],
        mask_ratio=[0.6, 0.4],
        encoder_depth=24,
        decoder_depth=1,
        mlp_ratio=4,
        qkv_bias=True,
        num_encoder_heads=6,
        num_decoder_heads=6,
        device=device,
    ).to(device)
