import torch 
import numpy as np
from transformer import Block

class MWTx(torch.nn.Module):
    def __init__(self, seqlen, in_dim, params):
        super().__init__()
        self.seqlen = seqlen
        self.in_dim = in_dim

        mdl_cfg = params['model']
        self.enc_emb_dim = mdl_cfg['enc_emb_dim']
        self.dec_emb_dim = mdl_cfg['dec_emb_dim']
        self.mask_ratio = mdl_cfg['mask_ratio']
        
        # encoder specifics
        self.cls_token   = torch.nn.Parameter(torch.zeros(1, 1, self.enc_emb_dim), requires_grad=True)
        self.enc_pos_emb = torch.nn.Parameter(torch.zeros(1, self.seqlen+1, self.enc_emb_dim), requires_grad=False)
        self.enc_proj    = torch.nn.Conv2d(1, self.enc_emb_dim, kernel_size=(1, self.in_dim), bias=True)
        
        self.enc_blocks  = torch.nn.ModuleList([
            Block(self.enc_emb_dim, mdl_cfg['enc_nhead'], mlp_ratio=4, qkv_bias=True, \
                  norm_layer=torch.nn.LayerNorm) for i in range(mdl_cfg['enc_nlayer'])])
        self.enc_norm = torch.nn.LayerNorm(self.enc_emb_dim)
        
        # decoder specifics
        self.dec_proj    = torch.nn.Conv2d(1, self.dec_emb_dim, kernel_size=(1, self.enc_emb_dim), bias=True)
        self.mask_token  = torch.nn.Parameter(torch.randn(1, 1, self.dec_emb_dim), requires_grad=True)
        self.dec_pos_emb = torch.nn.Parameter(torch.zeros(1, self.seqlen+1, self.dec_emb_dim), requires_grad=False)

        self.dec_blocks  = torch.nn.ModuleList([
            Block(self.dec_emb_dim, mdl_cfg['dec_nhead'], mlp_ratio=4, qkv_bias=True, \
                  norm_layer=torch.nn.LayerNorm) for i in range(mdl_cfg['dec_nlayer'])])
        self.dec_norm = torch.nn.LayerNorm(self.dec_emb_dim)
        
        self.dec_pred = torch.nn.Linear(self.dec_emb_dim, self.in_dim, bias=True) 
        self.initialize_weights()
        
    def pos_embd_gen(self, seqlen, emb_dim, cls_token):
        pos_emb = np.zeros((seqlen, emb_dim), dtype=np.float32)
        for _pos in range(seqlen):
            for _c in range(emb_dim):
                pos_emb[_pos, _c] = _pos / np.power(10000, 2 * (_c // 2) / emb_dim) 
        
        pos_emb[:, 0::2] = np.sin(pos_emb[:, 0::2])  # dim 2i
        pos_emb[:, 1::2] = np.cos(pos_emb[:, 1::2])  # dim 2i+1

        if cls_token:
            pos_emb = np.concatenate([np.zeros([1, emb_dim]), pos_emb], axis=0)
        return pos_emb
    
    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        enc_pos_emb = self.pos_embd_gen(self.seqlen, self.enc_emb_dim, cls_token=True)
        self.enc_pos_emb.data.copy_(torch.from_numpy(enc_pos_emb).float().unsqueeze(0))

        dec_pos_emb = self.pos_embd_gen(self.seqlen, self.dec_emb_dim, cls_token=True)
        self.dec_pos_emb.data.copy_(torch.from_numpy(dec_pos_emb).float().unsqueeze(0))

        # initialize proj like nn.Linear (instead of nn.Conv2d)
        w = self.enc_proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        w = self.dec_proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token,  std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x):
        N, L, D = x.shape  # batch, length, dim (a.k.a. BNC)
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def uniform_masking(self, x):
        N, L, D = x.shape  # batch, length, dim

        ids_keep = torch.linspace(0, L-1, L-round(self.mask_ratio*L), device=x.device).round().long()
        x_masked = torch.gather(x, dim=1, index=ids_keep.repeat(N, 1).unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones(L, device=x.device)
        mask.index_fill_(0, ids_keep, 0)
        masked_ids  = torch.masked_select(torch.arange(0, L, device=x.device), mask==1)
        ids_restore = torch.argsort(torch.cat([ids_keep, masked_ids]))
        return x_masked, mask.repeat(N, 1), ids_restore.repeat(N, 1)

    def limitted_view_mask(self, x):
        N, L, D = x.shape  # batch, length, dim
        if self.mask_ratio >= 0:
            ids_keep = torch.arange(round(self.mask_ratio*L), L,   device=x.device)
        else:
            ids_keep = torch.arange(0, L+round(self.mask_ratio*L), device=x.device)
            
        x_masked = torch.gather(x, dim=1, index=ids_keep.repeat(N, 1).unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones(L, device=x.device)
        mask.index_fill_(0, ids_keep, 0)
        masked_ids  = torch.masked_select(torch.arange(0, L, device=x.device), mask==1)
        ids_restore = torch.argsort(torch.cat([ids_keep, masked_ids]))
        return x_masked, mask.repeat(N, 1), ids_restore.repeat(N, 1)

    def missing_wedge_mask(self, x):
        N, L, D = x.shape  # batch, length, dim
        mw = round(self.mask_ratio*L)
        
        mseqs = torch.randint(low=0, high=L-mw, size=(N, ), device=x.device)
        
        ids_keep = torch.cat([torch.cat([torch.arange(0, mseqs[n], device=x.device), \
                                        torch.arange(mseqs[n]+mw, L, device=x.device)])[None] for n in range(N)], axis=0)
        masked_ids = torch.cat([torch.arange(mseqs[n], mseqs[n]+mw, device=x.device)[None] for n in range(N)], axis=0)

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones((N, L), device=x.device)
        
        mask[torch.arange(mask.size(0)).unsqueeze(1), ids_keep] = 0

        ids_restore = torch.argsort(torch.cat([ids_keep, masked_ids], axis=1))
        return x_masked, mask, ids_restore

    def forward_enc(self, x):
        _emb = self.enc_proj(x).flatten(2).transpose(1, 2)

        # add pos embed w/o cls token
        _tmp = _emb + self.enc_pos_emb[:, 1:, :]

        # masking: length -> length * mask_ratio
        # _tmp, mask, ids_restore = self.random_masking(_tmp)
        # _tmp, mask, ids_restore = self.uniform_masking(_tmp)
        _tmp, mask, ids_restore = self.missing_wedge_mask(_tmp)

        # append cls token
        cls_token  = self.cls_token + self.enc_pos_emb[:, :1, :]
        cls_tokens = cls_token.expand(_tmp.shape[0], -1, -1)
        _tmp = torch.cat((cls_tokens, _tmp), dim=1)

        for blk in self.enc_blocks:
            _tmp = blk(_tmp)
        _tmp = self.enc_norm(_tmp)
        
        return _tmp, mask, ids_restore

    def forward_dec(self, x, ids_restore):
        # embed tokens
        x = self.dec_proj(x[:,None]).flatten(2).transpose(1, 2)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x  = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.dec_pos_emb

        # apply Transformer blocks
        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)

        # predictor projection
        x = self.dec_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def forward_loss(self, target, pred, mask):
        """
        imgs: [N, H, W]
        pred: [N, L, W]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        loss = (pred - target) ** 2
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed angles
        loss = loss.mean()
        return loss

    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is not None:
            self.mask_ratio = mask_ratio
        latent, mask, ids_restore = self.forward_enc(imgs)
        pred = self.forward_dec(latent, ids_restore)  # [N, L, D]

        loss = self.forward_loss(imgs[:,0], pred, mask)
        return loss, pred, mask
