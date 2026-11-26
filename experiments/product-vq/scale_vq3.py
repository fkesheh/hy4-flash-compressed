"""SVQ3: learned 8-entry scalar codebooks and packed 3-bit scale indices."""
import numpy as np
import compress
MAGIC=b'SVQ3';K=8;BITS=3

def encode_hyb1(data):
 magic,m,n,k,per=compress.HEADER.unpack_from(data)
 if magic!=b'HYB1' or per!=1:raise ValueError('SVQ3 expects HYB1 per-row scales')
 off=compress.HEADER.size;na=(m*k+7)//8;nb=(k*n+7)//8;signs=data[off:off+na+nb];off+=na+nb
 scales=np.frombuffer(data,dtype='<f2',count=m+k+n,offset=off).astype(np.float32);parts=[compress.HEADER.pack(MAGIC,m,n,k,per),signs];i=0
 for count in (m,k,n):
  x=scales[i:i+count];c=np.quantile(x,np.linspace(0,1,K)).astype(np.float32)
  for _ in range(100):
   ix=np.abs(x[:,None]-c[None,:]).argmin(axis=1);s=np.bincount(ix,weights=x,minlength=K);n0=np.bincount(ix,minlength=K);cn=np.sort(np.where(n0,s/np.maximum(n0,1),c))
   if np.allclose(c,cn,rtol=0,atol=1e-9):c=cn;break
   c=cn
  c=c.astype('<f2');ix=np.abs(x[:,None]-c.astype(np.float32)[None,:]).argmin(axis=1).astype(np.uint8);bits=((ix[:,None]>>np.arange(BITS-1,-1,-1))&1).astype(np.uint8).ravel();parts.extend((c.tobytes(),np.packbits(bits,bitorder='little').tobytes()));i+=count
 return b''.join(parts)

def decode_factors(data):
 magic,m,n,k,per=compress.HEADER.unpack_from(data)
 if magic!=MAGIC or per!=1:raise ValueError('invalid SVQ3 payload')
 off=compress.HEADER.size;na=(m*k+7)//8;nb=(k*n+7)//8;signs=data[off:off+na+nb]
 if len(signs)!=na+nb:raise ValueError('truncated SVQ3 signs')
 off+=na+nb;scales=[]
 for count in (m,k,n):
  end=off+2*K
  if end>len(data):raise ValueError('truncated SVQ3 codebook')
  c=np.frombuffer(data,dtype='<f2',count=K,offset=off).astype(np.float32);off=end;size=(count*BITS+7)//8
  raw=np.unpackbits(np.frombuffer(data,dtype=np.uint8,count=size,offset=off),bitorder='little')[:count*BITS].reshape(count,BITS);ix=(raw<<np.arange(BITS-1,-1,-1)).sum(axis=1);off+=size;scales.extend(c[ix])
 if off!=len(data):raise ValueError('trailing SVQ3 bytes')
 hyb=compress.HEADER.pack(b'HYB1',m,n,k,per)+signs+np.asarray(scales,dtype='<f2').tobytes();return compress.unpack(hyb)
