"""ITQ base plus SVQ3 3-bit scale codebooks with measured pair budget."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent.parent));import compress
sys.path.insert(0,str(ROOT));from residual import fit_itq;from scale_vq3 import encode_hyb1,decode_factors
sys.path.insert(0,str(ROOT.parent));from score import score
LIMIT=158632

def main():
 p=argparse.ArgumentParser();p.add_argument('--rg',type=int,default=183);p.add_argument('--rd',type=int,default=130);p.add_argument('--sweeps',type=int,default=240);p.add_argument('--gate-seed',type=int,default=53);p.add_argument('--down-seed',type=int,default=101);a=p.parse_args();paths=[];out=[]
 try:
  for name,rank,seed in [('gate_up',a.rg,a.gate_seed),('down',a.rd,a.down_seed)]:
   w=np.load(f'experiments/reference/hy4-{name}_proj.npy').astype(np.float32);budget=compress.packed_size(*w.shape,rank,1);raw,actual=fit_itq(w,budget,a.sweeps,seed);blob=encode_hyb1(raw);path=ROOT/f'{name}-r{rank}-s{a.sweeps}-seed{seed}-scale3b.svq';path.write_bytes(blob);paths.append(path);l,r=decode_factors(blob);out.append(l@r)
  val=score(*out);nbytes=sum(x.stat().st_size for x in paths);rec={'config':vars(a)|{'scale_codebook_size':8,'scale_index_bits':3},'hypothesis':'Trade one scale-index bit and half the scale centroids for two more ITQ ranks, testing if the lower scale precision is offset by stronger factors.','bytes':nbytes,'within_budget':nbytes<=LIMIT,'score':val,'command':f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/hy4-compress-env/bin/python experiments/product-vq/scale_alloc3.py --rg {a.rg} --rd {a.rd} --sweeps {a.sweeps} --gate-seed {a.gate_seed} --down-seed {a.down_seed}','artifacts':[str(x) for x in paths]}
 except Exception as e:
  rec={'config':vars(a)|{'scale_codebook_size':8,'scale_index_bits':3},'hypothesis':'Trade one scale-index bit and half the scale centroids for two more ITQ ranks.','bytes':sum(x.stat().st_size for x in paths),'failure':repr(e),'command':f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/hy4-compress-env/bin/python experiments/product-vq/scale_alloc3.py --rg {a.rg} --rd {a.rd} --sweeps {a.sweeps} --gate-seed {a.gate_seed} --down-seed {a.down_seed}','artifacts':[str(x) for x in paths]};raise
 finally:
  with (ROOT/'results.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
 print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
