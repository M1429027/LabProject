from __future__ import annotations
import argparse, io, json, zipfile, math
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np, torch
from torch.utils.data import DataLoader
from learning.karate_selfcal.evaluation.compare_camera_geometry import build_comparison, camera_center_from_extrinsics, relative_centers
from learning.karate_selfcal.stage_a.dataset import KarateStageADataset, collate_stage_a
from learning.karate_selfcal.stage_a.model import build_model
from learning.karate_selfcal.stage_a.train import move_batch, resolve_device

def parse_args():
 p=argparse.ArgumentParser();
 p.add_argument('--checkpoint',required=True); p.add_argument('--manifest',required=True); p.add_argument('--rough-extrinsics-json',required=True); p.add_argument('--reference-extrinsics-json',required=True); p.add_argument('--gt-zip',required=True); p.add_argument('--sequence',required=True); p.add_argument('--split',default='test'); p.add_argument('--anchor-view',default='karate004_cam02'); p.add_argument('--view-ids',nargs='+',default=['karate004_cam02','karate004_cam03','karate004_cam07','karate004_cam13']); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--device',default='cuda'); p.add_argument('--output-dir',required=True); p.add_argument('--aggregation',choices=['mean','median'],default='mean'); return p.parse_args()

def load_json(p): return json.loads(Path(p).read_text())
def save_json(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def ident(pid):
 s=str(pid).lower();
 if s.endswith('01'): return 0
 if s.endswith('02'): return 1
 ds=''.join(c for c in s if c.isdigit()); return max(int(ds)-1,0) if ds else 0

def load_gt(zip_path, sequence):
 out={}
 with zipfile.ZipFile(zip_path) as z:
  pref=f'{sequence}/processed_data/poses3d/'
  for n in z.namelist():
   if n.startswith(pref) and n.endswith('.npy'):
    fid=int(Path(n).stem); payload=np.load(io.BytesIO(z.read(n)),allow_pickle=True).item(); people={}
    for pid,j in payload.items(): people[0 if str(pid).endswith('01') else 1]=np.asarray(j,dtype=float)[:17,:3]
    out[fid]=people
 return out

def pelvis(x): return 0.5*(x[11]+x[12])
def umeyama(src,dst):
 src=np.asarray(src,float); dst=np.asarray(dst,float); ms=src.mean(0); md=dst.mean(0); xs=src-ms; xd=dst-md; cov=xs.T@xd/max(len(src),1); u,s,vt=np.linalg.svd(cov); r=vt.T@u.T
 if np.linalg.det(r)<0: vt[-1]*=-1; r=vt.T@u.T
 sc=float(s.sum()/max(float((xs**2).sum()/max(len(src),1)),1e-8)); t=md-sc*(r@ms); return sc*(src@r.T)+t

def add(bucket,p,g,joints):
 p=p[joints]; g=g[joints]; ok=np.isfinite(p).all(1)&np.isfinite(g).all(1)
 if ok.sum()<3: return False
 p=p[ok]; g=g[ok]; bucket['raw']+=np.linalg.norm(p-g,axis=1).tolist();
 if ok.sum()>=4: bucket['similarity']+=np.linalg.norm(umeyama(p,g)-g,axis=1).tolist()
 return True

def summarize(items,gt,joints):
 b={'raw':[],'similarity':[]}; n=0
 for (fid,pid),pred in sorted(items.items()):
  gf=gt.get(fid+1)
  if gf is not None and pid in gf and add(b,pred,gf[pid],joints): n+=1
 return {'num_person_frames':n,'raw_mpjpe_m':float(np.mean(b['raw'])) if b['raw'] else None,'similarity_aligned_mpjpe_m':float(np.mean(b['similarity'])) if b['similarity'] else None}

def skew(v):
 z=torch.zeros_like(v[...,0]); x,y,zc=v[...,0],v[...,1],v[...,2]
 return torch.stack([torch.stack([z,-zc,y],-1),torch.stack([zc,z,-x],-1),torch.stack([-y,x,z],-1)],-2)

def rotvec_to_matrix(rv):
 theta=torch.linalg.norm(rv,dim=-1,keepdim=True).clamp_min(1e-8); axis=rv/theta; K=skew(axis); eye=torch.eye(3,device=rv.device,dtype=rv.dtype).expand(rv.shape[:-1]+(3,3)); th=theta.unsqueeze(-1); return eye+torch.sin(th)*K+(1-torch.cos(th))*(K@K)

def triangulate(batch, center_delta, rot_delta=None, scale_delta=None):
 rt=batch['ray_tokens']; mask=batch['joint_view_mask'].bool(); origin_norm=rt[...,0:3]; direction=torch.nn.functional.normalize(rt[...,3:6],dim=-1); conf=rt[...,8].clamp_min(0); scale=batch['ray_origin_scale'].reshape(-1,1,1,1); center=batch['ray_origin_center'].reshape(-1,1,1,3); origin=origin_norm*scale+center
 if scale_delta is not None: origin=center+(origin-center)*torch.exp(scale_delta.reshape(1,1,-1,1))
 origin=origin+center_delta.reshape(1,1,-1,3)
 if rot_delta is not None:
  R=rotvec_to_matrix(rot_delta).reshape(1,1,-1,3,3); direction=(R@direction.unsqueeze(-1)).squeeze(-1); direction=torch.nn.functional.normalize(direction,dim=-1)
 w=(conf*mask.float()).clamp_min(0); eye=torch.eye(3,device=rt.device,dtype=rt.dtype).reshape(1,1,1,3,3); P=eye-direction.unsqueeze(-1)*direction.unsqueeze(-2); WP=P*w.unsqueeze(-1).unsqueeze(-1); lhs=WP.sum(2); rhs=(WP@origin.unsqueeze(-1)).sum(2).squeeze(-1); valid=w.gt(0).sum(2)>=2; pts=torch.linalg.solve(lhs+1e-4*eye.reshape(1,1,3,3),rhs.unsqueeze(-1)).squeeze(-1); return torch.where(valid.unsqueeze(-1),pts,batch['triangulated_3d'])

def camera_payload(rough, view_ids, center_delta, rot_delta, scale_delta, aggregation):
 payload=json.loads(json.dumps(rough)); centers=np.vstack([camera_center_from_extrinsics(payload['extrinsics_by_view'][v]) for v in view_ids]); rig_center=centers.mean(0)
 def axang_to_R(v):
  th=float(np.linalg.norm(v));
  if th<1e-9: return np.eye(3)
  a=v/th; x,y,z=a; K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]],float); return np.eye(3)+math.sin(th)*K+(1-math.cos(th))*(K@K)
 for i,v in enumerate(view_ids):
  meta=payload['extrinsics_by_view'][v]; R=np.asarray(meta['rotation'],float).reshape(3,3); C=camera_center_from_extrinsics(meta); C2=rig_center+(C-rig_center)*math.exp(float(scale_delta[i]))+center_delta[i]; Rc=axang_to_R(rot_delta[i]); R2=R@Rc.T; t2=-R2@C2; meta['rotation']=R2.tolist(); meta['translation']=t2.tolist(); meta['camera_center_world']=C2.tolist()
  if 'projection_matrix' in meta:
   P=np.asarray(meta['projection_matrix'],float)
   if P.shape==(3,4): P[:,:3]=R2; P[:,3]=t2; meta['projection_matrix']=P.tolist()
 payload['stage']='stage_a_static_full_extrinsic_correction'; payload['aggregation']=aggregation; return payload

def main():
 args=parse_args(); out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); ck=torch.load(args.checkpoint,map_location='cpu'); model=build_model(ck['config']).to(resolve_device(args.device)); model.load_state_dict(ck['model_state'],strict=False); model.eval(); ds=KarateStageADataset(args.manifest,split=args.split); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate_stage_a); dev=resolve_device(args.device)
 cd=defaultdict(list); rd=defaultdict(list); sd=defaultdict(list)
 with torch.no_grad():
  for batch in loader:
   bd=move_batch(batch,dev); o=model(bd['ray_tokens'],view_mask=bd['view_mask'],joint_view_mask=bd['joint_view_mask']); vm=bd['view_mask'].cpu().numpy().astype(bool); c=(o['pred_camera_origin_delta']*bd['ray_origin_scale'].reshape(-1,1,1)).cpu().numpy(); r=o.get('pred_camera_rotation_delta',torch.zeros_like(o['pred_camera_origin_delta'])).cpu().numpy(); s=o.get('pred_camera_scale_delta',torch.zeros(o['pred_camera_origin_delta'].shape[:2],device=dev)).cpu().numpy()
   for bi in range(c.shape[0]):
    for vi in range(c.shape[1]):
     if vm[bi,vi]: cd[vi].append(c[bi,vi]); rd[vi].append(r[bi,vi]); sd[vi].append(s[bi,vi])
 agg=np.median if args.aggregation=='median' else np.mean; center=np.asarray([agg(np.asarray(cd[i]),axis=0) for i in range(len(args.view_ids))],np.float32); rot=np.asarray([agg(np.asarray(rd[i]),axis=0) for i in range(len(args.view_ids))],np.float32); scl=np.asarray([agg(np.asarray(sd[i]),axis=0) for i in range(len(args.view_ids))],np.float32)
 center_t=torch.tensor(center,device=dev); rot_t=torch.tensor(rot,device=dev); scl_t=torch.tensor(scl,device=dev)
 methods={'rough_anchor':{},'center_only':{},'full_center_rotation_scale':{}}
 with torch.no_grad():
  for batch in loader:
   bd=move_batch(batch,dev); rough=bd['triangulated_3d'].cpu().numpy(); center_pts=triangulate(bd,center_t).cpu().numpy(); full_pts=triangulate(bd,center_t,rot_t,scl_t).cpu().numpy()
   for i in range(rough.shape[0]):
    key=(int(batch['frame_id'][i]), ident(batch['person_id'][i])); methods['rough_anchor'][key]=rough[i]; methods['center_only'][key]=center_pts[i]; methods['full_center_rotation_scale'][key]=full_pts[i]
 gt=load_gt(args.gt_zip,args.sequence); metrics={m:{'all_joints_0_16':summarize(v,gt,list(range(17))),'body_joints_5_16':summarize(v,gt,list(range(5,17)))} for m,v in methods.items()}
 rough=load_json(args.rough_extrinsics_json); ref=load_json(args.reference_extrinsics_json); corr=camera_payload(rough,args.view_ids,center,rot,scl,args.aggregation); save_json(out/'static_full_corrected_extrinsics.json',corr); geom={'rough_vs_oracle':build_comparison(relative_centers(rough,args.anchor_view),relative_centers(ref,args.anchor_view),args.anchor_view),'full_corrected_vs_oracle':build_comparison(relative_centers(corr,args.anchor_view),relative_centers(ref,args.anchor_view),args.anchor_view)}
 report={'stage':'static_full_extrinsic_refinement_diagnostic','aggregation':args.aggregation,'static_delta':{v:{'center_delta_m':center[i].tolist(),'rotation_delta_axis_angle_rad':rot[i].tolist(),'scale_delta_log':float(scl[i])} for i,v in enumerate(args.view_ids)},'camera_geometry':geom,'gt_metrics':metrics,'paths':{'corrected_extrinsics':str(out/'static_full_corrected_extrinsics.json')}}; save_json(out/'static_full_extrinsic_refinement_diagnostic.json',report); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
