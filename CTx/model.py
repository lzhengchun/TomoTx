import torch, sys, os
import numpy as np
from transformer import Block

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import init_weights, pos_embd_gen

class CTx(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        mdl_cfg = params['model']
        self.enc_emb_dim = mdl_cfg['enc_emb_dim']
        self.dec_emb_dim = mdl_cfg['dec_emb_dim']
        self.out_dim     = mdl_cfg['out_dim']
        self.out_seqlen  = mdl_cfg['out_seqlen']

        self.dec_proj    = torch.nn.Conv2d(1, self.dec_emb_dim, kernel_size=(1, self.enc_emb_dim), bias=True)
        self.mask_token  = torch.nn.Parameter(torch.randn(1, 1, self.dec_emb_dim), requires_grad=True)
        self.dec_pos_emb = torch.nn.Parameter(torch.zeros(1, self.out_seqlen+1, self.dec_emb_dim), requires_grad=False)

        self.dec_blocks  = torch.nn.ModuleList([
            Block(self.dec_emb_dim, mdl_cfg['dec_nhead'], mlp_ratio=4, qkv_bias=True, qk_scale=None, \
                  norm_layer=torch.nn.LayerNorm) for i in range(mdl_cfg['dec_nlayer'])])
        self.dec_norm = torch.nn.LayerNorm(self.dec_emb_dim)
        
        self.dec_pred = torch.nn.Linear(self.dec_emb_dim, self.out_dim, bias=True) 
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        dec_pos_emb = pos_embd_gen(self.out_seqlen, self.dec_emb_dim, cls_token=True)
        self.dec_pos_emb.data.copy_(torch.from_numpy(dec_pos_emb).float().unsqueeze(0))

        # initialize proj like nn.Linear (instead of nn.Conv2d)
        w = self.dec_proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights)

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * 1)
        imgs: (N, 3, H, W)
        """
        p = int(x.shape[2]**.5)
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.dec_proj(x[:,None]).flatten(2).transpose(1, 2)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x  = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        if x.shape[1] - 1 >= self.out_seqlen:
            x = x[:,:self.out_seqlen+1] + self.dec_pos_emb
        else:
            n, l, d = x.shape
            x = torch.cat([x, self.mask_token.repeat(n, self.out_seqlen-l+1, 1)], axis=1) + self.dec_pos_emb

        # apply Transformer blocks
        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)

        # predictor projection
        x = self.dec_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        recon = self.unpatchify(x)

        return recon

