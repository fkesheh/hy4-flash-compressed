"""Low-rank sign factors plus learned product-quantized residual."""
import argparse,json,math,sys,struct
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT.parent.parent))
import compress
sys.path.insert(0,str(ROOT.parent));from score import score
LIMIT=158632; PVQ_HEADER=struct.Struct('<4sIIBB')

def fit_itq(w,budget,sweeps,seed):
 m,n=w.shape; k=max(k for k in range(1,min(m,n)+1) if compress.packed_size(m,n,k,1)<=budget)
 u,s,vt=np.linalg.svd(w,full_matrices=False);u=u[:,:k];v=vt[:k].T;rng=np.random.default_rng(seed)
 def rotate(x):
  q,_=np.linalg.qr(rng.standard_normal((k,k)).astype(np.float32));b=np.where(x@q>=0,1.,-1.).astype(np.float32)
  for _ in range(25):
   p,_,qt=np.linalg.svd(x.T@b,full_matrices=False);q=p@qt;b=np.where(x@q>=0,1.,-1.).astype(np.float32)
  return b
 a=rotate(u);b=rotate(v).T;row=np.ones(m,np.float32);col=np.ones(n,np.float32);latent=s[:k].astype(np.float32)
 for step in range(5+sweeps):
  if step>=5:
   compress.refine_signs(w,a*row[:,None]*latent,col,b);compress.refine_signs(w.T,b.T*col[:,None]*latent,row,a.T)
  x=row[:,None]*a;y=b*col;gram=(x.T@x)*(y@y.T);rhs=np.sum((w@y.T)*x,axis=0);latent=np.linalg.lstsq(gram,rhs,rcond=None)[0]
  base=(a*latent)@b;scaled=base*col;row=np.sum(w*scaled,axis=1)/np.maximum((scaled**2).sum(axis=1),1e-30);scaled=base*row[:,None];col=np.sum(w*scaled,axis=0)/np.maximum((scaled**2).sum(axis=0),1e-30)
 parts=[compress.HEADER.pack(b'HYB1',m,n,k,1),np.packbits(a>0).tobytes(),np.packbits(b>0).tobytes()]
 for scale in (row,latent,col):parts.append(scale.astype('<f2').tobytes())
 return b''.join(parts),k

def pq_fit(x,width,bits,seed,iters):
 flat=np.asarray(x,np.float32).ravel(); v=flat.reshape(-1,width); k=1<<bits; rng=np.random.default_rng(seed)
 smp=v[rng.choice(len(v),min(8192,len(v)),replace=False)]; centers=[smp[rng.integers(len(smp))]]; nearest=((smp-centers[0])**2).sum(axis=1)
 for _ in range(1,k):
  centers.append(smp[np.argmax(nearest)]); nearest=np.minimum(nearest,((smp-centers[-1])**2).sum(axis=1))
 c=np.stack(centers).astype(np.float32)
 for _ in range(iters):
  sums=np.zeros_like(c,np.float64);counts=np.zeros(k,np.int64)
  for st in range(0,len(v),32768):
   z=v[st:st+32768]; ix=((z[:,None,:]-c[None,:,:])**2).sum(axis=2).argmin(axis=1)
   for j in range(k):
    mask=ix==j; counts[j]+=mask.sum()
    if mask.any():sums[j]+=z[mask].sum(axis=0,dtype=np.float64)
  cn=np.stack([sums[j]/counts[j] if counts[j] else c[j] for j in range(k)]).astype(np.float32)
  if np.allclose(cn,c,rtol=0,atol=1e-8):c=cn;break
  c=cn
 cf=c.astype('<f2').astype(np.float32); ix=np.empty(len(v),np.uint8)
 for st in range(0,len(v),32768):
  z=v[st:st+32768];ix[st:st+len(z)]=((z[:,None,:]-cf[None,:,:])**2).sum(axis=2).argmin(axis=1)
 packed=np.packbits(((ix[:,None]>>np.arange(bits-1,-1,-1))&1).astype(np.uint8).ravel(),bitorder='little').tobytes()
 header=PVQ_HEADER.pack(b'PVQ1',len(flat),width,bits,0)
 return header+cf.astype('<f2').tobytes()+packed

def pq_unpack(data):
 magic,n,w,bits,_=PVQ_HEADER.unpack_from(data);k=1<<bits
 c=np.frombuffer(data,dtype='<f2',count=k*w,offset=PVQ_HEADER.size).astype(np.float32).reshape(k,w)
 off=PVQ_HEADER.size+k*w*2; raw=np.frombuffer(data,dtype=np.uint8,offset=off); bx=np.unpackbits(raw,bitorder='little')[:(n//w)*bits].reshape(-1,bits)
 ix=(bx<<np.arange(bits-1,-1,-1)).sum(axis=1);return c[ix].reshape(-1)[:n]

def decode_artifact(path):
 blob=Path(path).read_bytes();nbase=struct.unpack_from('<I',blob)[0];base=blob[4:4+nbase];resid=blob[4+nbase:];left,right=compress.unpack(base);return left@right+pq_unpack(resid).reshape(left.shape[0],right.shape[1])

def main():
 p=argparse.ArgumentParser();p.add_argument('--rg',type=int,default=100);p.add_argument('--rd',type=int,default=100);p.add_argument('--sweeps',type=int,default=20);p.add_argument('--width',type=int,default=48);p.add_argument('--bits',type=int,default=2);p.add_argument('--seed',type=int,default=7);p.add_argument('--iters',type=int,default=12);p.add_argument('--base-init',choices=['svd','itq'],default='itq');a=p.parse_args()
 files=[('gate_up','experiments/reference/hy4-gate_up_proj.npy',a.rg),('down','experiments/reference/hy4-down_proj.npy',a.rd)]; paths=[];outputs=[];lowranks=[]
 try:
  for name,src,rank in files:
   w=np.load(src).astype(np.float32); budget=compress.packed_size(*w.shape,rank,1); data,actual,_=compress.fit(w,8*budget/w.size,True,a.sweeps) if a.base_init=='svd' else (*fit_itq(w,budget,a.sweeps,a.seed),None); left,right=compress.unpack(data); approx=left@right;res=w-approx; q=pq_fit(res,a.width,a.bits,a.seed,a.iters); rec=approx+pq_unpack(q).reshape(w.shape); lowranks.append(data);outputs.append(rec)
   path=ROOT/f'residual-{name}-{a.base_init}-r{rank}-w{a.width}-b{a.bits}-s{a.seed}.bin';path.write_bytes(struct.pack('<I',len(data))+data+q);paths.append(path)
  val=score(*outputs);size=sum(x.stat().st_size for x in paths)
  rec={'config':vars(a),'hypothesis':f'Combine {a.base_init}-initialized low-rank sign factors with learned product-VQ of their dense residual; spend bytes across base and residual.','bytes':size,'within_budget':size<=LIMIT,'score':val,'command':f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/hy4-compress-env/bin/python experiments/product-vq/residual.py --rg {a.rg} --rd {a.rd} --sweeps {a.sweeps} --width {a.width} --bits {a.bits} --seed {a.seed} --iters {a.iters}','artifacts':[str(x) for x in paths]}
 except Exception as e:
  rec={'config':vars(a),'hypothesis':f'Combine {a.base_init}-initialized low-rank sign factors with learned product-VQ of their dense residual; spend bytes across base and residual.','bytes':sum(x.stat().st_size for x in paths),'failure':repr(e),'command':f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/hy4-compress-env/bin/python experiments/product-vq/residual.py --rg {a.rg} --rd {a.rd} --sweeps {a.sweeps} --width {a.width} --bits {a.bits} --seed {a.seed} --iters {a.iters}','artifacts':[str(x) for x in paths]};raise
 finally:
  with (ROOT/'results.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
 print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
