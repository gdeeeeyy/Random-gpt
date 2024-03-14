#conda activate gpt-env

import torch 
import torch.nn as nn
from torch.nn import functional as F

batch_size=54
block_size=256
max_iters=5000
eval_interval=500
learning_rate=3e-4
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
n_embd=384
n_head=6
n_layer=6
dropout=0.2

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text=f.read()

chars=sorted(list(set(text)))
vocab_size=len(chars)
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
encode=lambda s:[stoi[c] for c in s]
decode=lambda l: ''.join([itos[i] for i in l])

data=torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) #the 1000 chars will look like this to chatgpt

data=torch.tensor(encode(text), dtype=torch.long)

#train-test split
n=int(0.9*len(data)) #90% train, rest test
train_data=data[:n]
val_data=data[n:]

def get_batch(split):
    #generating a smol batch of data of inp x and target y
    data=train_data if split=='train' else val_data
    ix=torch.randint(len(data)-block_size, (batch_size,)) #random offsets into training set - 4 nos
    x=torch.stack([data[i:i+block_size] for i in ix]) #first block size chars from i
    y=torch.stack([data[i+1:i+block_size+1] for i in ix]) # offset by 1 of x
    x,y=x.to(device), y.to(device)
    return x,y

@torch.no_grad() #to avoid back propagation
def estimate_loss():
    out={}
    model.eval()
    for split in ['train', 'val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits, loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out

class Head(nn.Module):
    #creates one head of self attention
    def __init__(self, head_size):
        super().__init__()
        self.key=nn.Linear(n_embd, head_size, bias=False)
        self.query=nn.Linear(n_embd, head_size, bias=False)
        self.value=nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        #computing attention scores(affinities)
        wei=q@k.transpose(-2,-1)*C**-0.5 #(B,T,C)@(B,C,T)->(B,T,T)
        wei=wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) #(B,T,T)
        wei=F.softmax(wei, dim=-1)#(B,T,T)
        wei=self.dropout(wei)
        #performing weighted aggregation of values
        v=self.value(x) #(B,T,C)
        out=wei@v #(B,T,T)@(B,T,C)->(B,T,C)
        return out

#defining multi headed attention, they run in parallel
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj=nn.Linear(n_embd, n_embd)
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        out=torch.cat([h(x) for h in self.heads], dim=-1) #(B,T,C)
        out=self.proj(out)
        return out
    
#a simple linear layer followed by a non linear fn
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

#Creating a transformer block that facilitates communication and computation
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size=n_embd//n_head
        self.sa=MultiHeadAttention(n_head, head_size)
        self.ffwd=FeedForward(n_embd)
        #layer norms
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)
    
    def forward(self, x):
        #forking off and doing computations
        x=x+self.sa(self.ln1(x))
        x=x+self.ffwd(self.ln2(x))
        return x


class BigrameLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #each token reads off the logits for the next token from the vocab_size table
        self.token_embedding_table=nn.Embedding(vocab_size, n_embd) #size is vocab_size * vocab_size
        self.position_embedding_table=nn.Embedding(block_size, n_embd)
        # self.sa_heads=MultiHeadAttention(4, n_embd//4) #4 heads of 8 dim self attention
        # #the final layer is linear, maps from the embedding size to the vocab size
        # self.ffwd=FeedForward(n_embd)
        # self.blocks=nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd),
        # ) #combining comms many times
        self.blocks=nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embd)#final layer norm
        self.lm_head=nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T=idx.shape
        #idx and targets are both (B, T) tensor of integers
        tok_emb=self.token_embedding_table(idx) #(Batch,Time,Channel)
        pos_emb=self.position_embedding_table(torch.arange(T, device=device))#(T,C)
        x=tok_emb+pos_emb #(B,T,C)
        x=self.blocks(x) #(B,T,C)
        logits=self.lm_head(x)#(B,T,vocab_size)
        if targets is None:
            loss=None
        else:
            B, T, C=logits.shape
            logits=logits.view(B*T, C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is the (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond=idx[:, -block_size:] #crop idx to the last block_size tokens
            logits, loss=self(idx_cond) #getting predictions
            logits=logits[:, -1, :] #becomes (B, C) [focuses only on the last time step therefore -1]
            probs=F.softmax(logits, dim=-1) #(B, C) [apply softmax to get probs]
            idx_next=torch.multinomial(probs, num_samples=1) #(B, 1) [sample from the dist]
            idx=torch.cat((idx, idx_next), dim=1) #(B, T+1) [append sampled index to running sequence]
        return idx

model=BigrameLanguageModel()
m=model.to(device)

#creating an optimizer
optimizer=torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):
    #every once in a while eval the loss on train and val sets
    if(iter%eval_interval==0):
        losses=estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    #sample a batch of data
    xb,yb=get_batch('train')
    #evaluate loss
    logits, loss=m(xb, yb) 
    optimizer.zero_grad(set_to_none=True) #zeroing out grads from prev steps
    loss.backward() #getting grads for all params
    optimizer.step() #using grads to update all params

#generate from the model
context=torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
