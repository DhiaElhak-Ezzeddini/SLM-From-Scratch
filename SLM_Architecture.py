import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

class LayerNorm(nn.Module):
    def __init__(self,ndim,bias) : 
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self,x):
        return F.layer_norm(x,self.weight.shape,self.weight,self.bias,1e-5)
class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_head == 0 
        self.c_attn = nn.Linear(config.n_embed,3*config.n_embed,bias=config.bias) ## for the 3 matrices ; K Q V
        self.c_proj = nn.Linear(config.n_embed,config.n_embed,bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.flash = hasattr(F,"scaled_dot_product_attention")
        if not self.flash : 
            self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))
    
    def forward(self,x) : 
        B,T,C = x.size() 
        q,k,v = self.c_attn(x).split(self.n_embed,dim=2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        
        if self.flash : 
            y = F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        else : 
            att = (q@k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T]==0,float("-inf"))
            att = F.softmax(att,dim=-1)
            att = self.attn_dropout(att)
            y = att @ v 
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.residual_dropout(self.c_proj(y))
        
        return y         

class MLP(nn.Module):
    def __init__(self,config) : 
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed,4*config.n_embed,bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj  = nn.Linear(4*config.n_embed,config.n_embed,bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self,x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x)))) 
class Transformer_Block(nn.Module):
    def __init__(self,config) : 
        super().__init__()
        self.ln1 = LayerNorm(config.n_embed,config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embed,config.bias)
        self.mlp = MLP(config)
    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        
        return x 
    
class GPTModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Transformer_Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embed,config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embed,config.vocab_size,bias=False)
        self.transformer.wte.weight = self.lm_head.weight # weight tying
        self.apply(self._init_weights)
        for pn,p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p,mean=0.0,std=0.02/math.sqrt(2*config.n_layer))
    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None : 
                nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)
    def forward(self,idx,targets=None) : 
        b,t = idx.size()
        device = idx.device
        assert t <= self.config.block_size
        pos = torch.arange(0,t,dtype=torch.long,device=device)
        tok_embed = self.transformer.wte(idx)
        pos_embed = self.transformer.wpe (pos)
        x = self.transformer.drop(tok_embed + pos_embed)
        for block in self.transformer.h : 
            x = block(x)
        x = self.transformer.ln_f(x)
        logits  = self.lm_head(x)
        if targets is not  None : 
            
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1)
        else : 
            loss = None
        return logits , loss

    @torch.no_grad()
    def generate(self,idx,max_new_tokens,temperature=1.0,top_k=None):
        for _ in range(max_new_tokens) : 
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:,-self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:] / temperature
            if top_k is not None : 
                v,_ = torch.topk(logits,k=min(top_k,logits.size(-1)))
                logits[logits < v[:,[-1]]] = -float('Inf')
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx

@dataclass
class GPTConfig:
    block_size : int
    vocab_size : int
    n_layer : int
    n_head  : int
    n_embed : int
    dropout : float =0.0
    bias : bool = True
             