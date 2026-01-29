import torch, numpy as np, zlib, base64, argparse

# Exporte un UTTTPVNet (comme dans uttt_ppo_self_play.py) vers un starter CodinGame <100k chars.
# - Quantization int8 + scale float16 par tenseur
# - Packaging en un seul blob base85 (ASCII)
# - Génère un fichier Answer.py autonome (numpy only)

def load_state_dict(pth):
    sd = torch.load(pth, map_location='cpu')
    if not isinstance(sd, dict):
        raise TypeError("Bad .pth: expected a state_dict dict/OrderedDict")
    return sd

def pack(sd, n_blocks=4):
    arr = []
    arr += [sd['stem.0.weight'], sd['stem.1.weight'], sd['stem.1.bias']]
    for b in range(n_blocks):
        arr += [sd[f'blocks.{b}.conv1.weight'], sd[f'blocks.{b}.gn1.weight'], sd[f'blocks.{b}.gn1.bias']]
        arr += [sd[f'blocks.{b}.conv2.weight'], sd[f'blocks.{b}.gn2.weight'], sd[f'blocks.{b}.gn2.bias']]
    arr += [sd['pi_head.0.weight'], sd['pi_head.1.weight'], sd['pi_head.1.bias']]
    arr += [sd['pi_head.3.weight'], sd['pi_head.3.bias']]
    arr = [a.detach().cpu().numpy().astype(np.float32) for a in arr]

    scales = []
    qbytes = []
    for a in arr:
        m = float(np.max(np.abs(a)))
        s = m / 127.0 if m > 0 else 1.0
        scales.append(s)
        q = np.round(a / s).astype(np.int8)
        qbytes.append(q.tobytes())

    blob = np.array(scales, dtype=np.float16).tobytes() + b''.join(qbytes)
    comp = zlib.compress(blob, 9)
    return base64.b85encode(comp).decode('ascii')

TEMPLATE = r'''import sys,zlib,base64,numpy as np
from numpy.lib.stride_tricks import as_strided
A={A!r};c=32;ic=7;nb=4;gg=8
def L():
 r=zlib.decompress(base64.b85decode(A))
 s=np.frombuffer(r,np.float16,32).astype(np.float32);q=np.frombuffer(r,np.int8,offset=64)
 w=[];o=0;i=0
 def take(n,sh):
  nonlocal o,i
  a=q[o:o+n].astype(np.float32)*s[i];i+=1;o+=n
  w.append(a.reshape(sh))
 take(c*ic*9,(c,ic,3,3));take(c,(c,));take(c,(c,))
 for _ in range(nb):
  take(c*c*9,(c,c,3,3));take(c,(c,));take(c,(c,))
  take(c*c*9,(c,c,3,3));take(c,(c,));take(c,(c,))
 take(16*c,(16,c,1,1));take(16,(16,));take(16,(16,))
 take(16,(1,16,1,1));take(1,(1,))
 return w
W=L()
R=lambda x:np.maximum(x,0.)
def GN(x, w, b, g, e=1e-5):
 C,H,Wd=x.shape
 xg=x.reshape(g,C//g,H,Wd)
 m=xg.mean(axis=(1,2,3), keepdims=True)
 v=xg.var(axis=(1,2,3), keepdims=True)
 xhat=(xg - m) / np.sqrt(v + e)
 xhat=xhat.reshape(C, H, Wd)
 return xhat * w[:, None, None] + b[:, None, None]
def C3(x,w,p=1):
 if p:x=np.pad(x,((0,0),(p,p),(p,p)))
 s0,s1,s2=x.strides;H=x.shape[1]-2;W=x.shape[2]-2
 P=as_strided(x,(x.shape[0],H,W,3,3),(s0,s1,s2,s1,s2))
 return np.tensordot(w,P,([1,2,3],[0,3,4]))
def C1(x,w,b=None):
 y=w.reshape(w.shape[0],w.shape[1])@x.reshape(w.shape[1],-1);y=y.reshape(w.shape[0],x.shape[1],x.shape[2])
 return y if b is None else y+b[:,None,None]
def F(o):
 w=W;i=0
 x=C3(o,w[i]);i+=1;x=GN(x,w[i],w[i+1],gg);i+=2;x=R(x)
 for _ in range(nb):
  r=x
  x=C3(x,w[i]);i+=1;x=GN(x,w[i],w[i+1],gg);i+=2;x=R(x)
  x=C3(x,w[i]);i+=1;x=GN(x,w[i],w[i+1],gg);i+=2;x=R(x+r)
 x=C1(x,w[i]);i+=1;x=GN(x,w[i],w[i+1],1);i+=2;x=R(x)
 x=C1(x,w[i],w[i+1]);return x.reshape(-1)
def update_micro(board,micro):
 for i in range(9):
  r,c=divmod(i,3);s=board[r*3:r*3+3,c*3:c*3+3]
  v=0
  for p in (1,-1):
   if any((s[j,:]==p).all() for j in range(3)) or any((s[:,j]==p).all() for j in range(3)) or (s[0,0]==p and s[1,1]==p and s[2,2]==p) or (s[0,2]==p and s[1,1]==p and s[2,0]==p):v=p;break
  if not v and not (s==0).any():v=2
  micro[i]=v
def legal_mask(board,micro,next_board):
 m=np.zeros(81,bool)
 if next_board==-1 or micro[next_board]!=0:
  for i in range(9):
   if micro[i]==0:
    r,c=divmod(i,3)
    for rr in range(3):
     for cc in range(3):
      if board[r*3+rr,c*3+cc]==0:m[(r*3+rr)*9+(c*3+cc)]=1
 else:
  r,c=divmod(next_board,3)
  for rr in range(3):
   for cc in range(3):
    if board[r*3+rr,c*3+cc]==0:m[(r*3+rr)*9+(c*3+cc)]=1
 return m
def make_obs(board,micro,next_board,me,mask):
 me_p=(board==me).astype(np.float32);op_p=(board==-me).astype(np.float32)
 lp=np.zeros((9,9),np.float32);idx=np.flatnonzero(mask);lp[idx//9,idx%9]=1
 mm=np.zeros((9,9),np.float32);mo=np.zeros((9,9),np.float32);md=np.zeros((9,9),np.float32)
 for i in range(9):
  r,c=divmod(i,3);r*=3;c*=3
  if micro[i]==me:mm[r:r+3,c:c+3]=1
  elif micro[i]==-me:mo[r:r+3,c:c+3]=1
  elif micro[i]==2:md[r:r+3,c:c+3]=1
 npb=np.ones((9,9),np.float32)
 if next_board!=-1:
  npb.fill(0);r,c=divmod(next_board,3);npb[r*3:r*3+3,c*3:c*3+3]=1
 return np.stack([me_p,op_p,lp,mm,mo,md,npb],0)
board=np.zeros((9,9),np.int8);micro=np.zeros(9,np.int8);next_board=-1;me=1
def _tf9(a, k):
 # a: (C,9,9) or (9,9)
 if k >= 4:
  a = np.flip(a, axis=-1)  # mirror left-right
  k -= 4
 if k: a = np.rot90(a, k, axes=(-2,-1))
 return a

def _perm_invperm_81():
 perm = np.zeros((8,81), np.int16)
 invp = np.zeros((8,81), np.int16)
 for k in range(8):
  for r in range(9):
   for c in range(9):
    rr,cc = r,c
    kk=k
    if kk >= 4:
     cc = 8-cc
     kk -= 4
    if kk == 1: rr,cc = cc, 8-rr
    elif kk == 2: rr,cc = 8-rr, 8-cc
    elif kk == 3: rr,cc = 8-cc, rr
    i = r*9+c
    j = rr*9+cc
    perm[k,i] = j        # original i -> transformed j
    invp[k,j] = i        # transformed j -> original i
 return perm, invp

PERM, INVP = _perm_invperm_81()

def _softmax(x):
 x = x - np.max(x)
 e = np.exp(x)
 return e / np.sum(e)

def logits_ensemble(obs, mask):
 # obs: (7,9,9), mask: (81,) bool in ORIGINAL indexing
 acc = np.zeros(81, np.float32)
 for k in range(8):
  obs_k = _tf9(obs, k)                  # (7,9,9)
  mask_k = mask[INVP[k]]                # legality in transformed indexing
  lg = F(obs_k).astype(np.float32)      # (81,)
  lg[~mask_k] = -1e9
  p = _softmax(lg)                      # (81,) in transformed indexing
  acc += p[PERM[k]]                     # back to original indexing
 p = acc / 8.0
 # enforce original legality + renorm (safety)
 p *= mask.astype(np.float32)
 s = p.sum()
 if s > 0: p /= s
 return p
while 1:
 ox,oy=map(int,input().split())
 if ox!=-1:board[ox,oy]=-me;next_board=(ox%3)*3+(oy%3)
 n=int(input());moves=[tuple(map(int,input().split())) for _ in range(n)]
 update_micro(board,micro)
 mask=legal_mask(board,micro,next_board)
 obs=make_obs(board,micro,next_board,me,mask)
 probs = logits_ensemble(obs, mask)
 a = int(np.argmax(probs))
 print(a//9, a%9)
 board[a//9,a%9]=me;next_board=(a//9%3)*3+(a%9%3)
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_pth")
    ap.add_argument("-o", "--out", default="Answer.py")
    args = ap.parse_args()

    sd = load_state_dict(args.model_pth)
    A = pack(sd)
    code = TEMPLATE.format(A=repr(A))
    # TEMPLATE uses A={A!r} but we can't keep !r with format, so we inserted repr(A) already.
    # Fix double repr (repr('...')) -> '...'
    code = code.replace("A=" + repr(repr(A)), "A=" + repr(A))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(code)
    print("wrote", args.out, "chars", len(code))

if __name__ == "__main__":
    main()
