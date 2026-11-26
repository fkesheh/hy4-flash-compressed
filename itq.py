"""Fit a packed expert projection with iterative quantization and sign refinement."""
import argparse
from pathlib import Path
import numpy as np
import compress
from verify import reference

def fit_itq(w, budget, sweeps, seed):
 m,n=w.shape; k=max(k for k in range(1,min(m,n)+1) if compress.packed_size(m,n,k,1)<=budget)
 u,s,vt=np.linalg.svd(w,full_matrices=False); u=u[:,:k]; v=vt[:k].T
 rng=np.random.default_rng(seed)
 def itq(x):
  q,_=np.linalg.qr(rng.standard_normal((k,k)).astype(np.float32))
  b=np.where(x@q>=0,1.,-1.).astype(np.float32)
  for _ in range(25):
   p,_,qt=np.linalg.svd(x.T@b,full_matrices=False); q=p@qt; b=np.where(x@q>=0,1.,-1.).astype(np.float32)
  return b
 a=itq(u); b=itq(v).T
 row=np.ones(m,np.float32); col=np.ones(n,np.float32); latent=s[:k].astype(np.float32)
 for step in range(5+sweeps):
  if step>=5:
   compress.refine_signs(w,a*row[:,None]*latent,col,b)
   compress.refine_signs(w.T,b.T*col[:,None]*latent,row,a.T)
  x=row[:,None]*a; y=b*col
  gram=(x.T@x)*(y@y.T); rhs=np.sum((w@y.T)*x,axis=0)
  latent=np.linalg.lstsq(gram,rhs,rcond=None)[0]
  base=(a*latent)@b
  scaled=base*col; row=np.sum(w*scaled,axis=1)/np.maximum((scaled**2).sum(axis=1),1e-30)
  scaled=base*row[:,None]; col=np.sum(w*scaled,axis=0)/np.maximum((scaled**2).sum(axis=0),1e-30)
 parts=[compress.HEADER.pack(b'HYB1',m,n,k,1),np.packbits(a>0).tobytes(),np.packbits(b>0).tobytes()]
 for scale in (row,latent,col): parts.append(scale.astype('<f2').tobytes())
 return b''.join(parts),k


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projection", choices=["gate-up", "down"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sweeps", type=int, required=True)
    parser.add_argument("--references", type=Path)
    args = parser.parse_args()
    weights = reference(args.projection, args.references).astype(np.float32)
    budget = 105762 if args.projection == "gate-up" else 52870
    data, _ = fit_itq(weights, budget, args.sweeps, args.seed)
    args.output.write_bytes(data)
