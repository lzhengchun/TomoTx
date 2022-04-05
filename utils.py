import torch 
import numpy as np

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, torch.nn.Linear) and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)

def pos_embd_gen(seqlen, emb_dim, cls_token):
    pos_emb = np.zeros((seqlen, emb_dim), dtype=np.float32)
    for _pos in range(seqlen):
        for _c in range(emb_dim):
            pos_emb[_pos, _c] = _pos / np.power(10000, 2 * (_c // 2) / emb_dim) 
    
    pos_emb[:, 0::2] = np.sin(pos_emb[:, 0::2])  # dim 2i
    pos_emb[:, 1::2] = np.cos(pos_emb[:, 1::2])  # dim 2i+1

    if cls_token:
        pos_emb = np.concatenate([np.zeros([1, emb_dim]), pos_emb], axis=0)
    return pos_emb

def random_masking(x, mask_ratio):
    N, L, D = x.shape  # batch, length, dim (a.k.a. BNC)
    len_keep = int(L * (1 - mask_ratio))
    
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

# this is to mimic sparse view
def uniform_masking(x, mask_ratio):
    N, L, D = x.shape  # batch, length, dim

    ids_keep = torch.linspace(0, L-1, L-round(mask_ratio*L), device=x.device).round().long()
    x_masked = torch.gather(x, dim=1, index=ids_keep.repeat(N, 1).unsqueeze(-1).repeat(1, 1, D))
    
    mask = torch.ones(L, device=x.device)
    mask.index_fill_(0, ids_keep, 0)
    masked_ids  = torch.masked_select(torch.arange(0, L, device=x.device), mask==1)
    ids_restore = torch.argsort(torch.cat([ids_keep, masked_ids]))
    return x_masked, mask.repeat(N, 1), ids_restore.repeat(N, 1)

# mimic missing wedge, limitted-view
def missing_wedge_mask(x, mask_ratio):
    N, L, D = x.shape  # batch, length, dim
    mw = round(mask_ratio*L)
    
    mseqs = torch.randint(low=0, high=L-mw, size=(N, ), device=x.device)
    
    ids_keep = torch.cat([torch.cat([torch.arange(0, mseqs[n], device=x.device), \
                                    torch.arange(mseqs[n]+mw, L, device=x.device)])[None] for n in range(N)], axis=0)
    masked_ids = torch.cat([torch.arange(mseqs[n], mseqs[n]+mw, device=x.device)[None] for n in range(N)], axis=0)

    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones((N, L), device=x.device)
    
    mask[torch.arange(mask.size(0)).unsqueeze(1), ids_keep] = 0

    ids_restore = torch.argsort(torch.cat([ids_keep, masked_ids], axis=1))
    return x_masked, mask, ids_restore
