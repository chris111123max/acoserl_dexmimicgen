"""Frozen, strictly verified public backbones; only spatial adapters learn.

ResFiT's QAgent factory is extended without overriding act/update/losses.
Input is RGB float [0,1] AFTER the unchanged RandomShiftsAug.
"""
import hashlib,json,os,sys,subprocess
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
ROOT=Path(__file__).resolve().parents[2]
WEIGHTS={
 'dinov2_vits14':('dinov2_vits14_pretrain.pth','b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'),
 'resnet18_imagenet':('resnet18-f37072fd.pth','f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec')}
WEIGHTS['squeezenet1_1_imagenet']=('squeezenet1_1-b8a52dc0.pth','b8a52dc049b60e4b6ab68ad0df457362afab8b6304b2febdc1650a5dab4d7e7b')
WEIGHTS['resnet18_spatial84']=WEIGHTS['resnet18_imagenet']
DINO_REV='7764ea0f912e53c92e82eb78a2a1631e92725fc8'

def digest(module):
 h=hashlib.sha256()
 for k,v in sorted(module.state_dict().items()):
  h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

class SpatialPretrainedEncoder(nn.Module):
 num_patch=81;patch_repr_dim=128;repr_dim=81*128
 def __init__(self,name):
  super().__init__();self.name=name
  filename,sha=WEIGHTS[name];path=ROOT/'cache/pretrained_visual'/filename
  assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,'pretrained checkpoint hash mismatch'
  state=torch.load(path,map_location='cpu',weights_only=True)
  if name=='dinov2_vits14':
   assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'third_party/dinov2',text=True).strip()==DINO_REV,'unverified DINO source revision'
   os.environ['XFORMERS_DISABLED']='1'
   sys.path.insert(0,str(ROOT/'third_party/dinov2'))
   from dinov2.hub.backbones import dinov2_vits14
   self.backbone=dinov2_vits14(pretrained=False)
   self.backbone.load_state_dict(state,strict=True)
   width=384
  elif name=='squeezenet1_1_imagenet':
   from torchvision.models import squeezenet1_1
   self.backbone=squeezenet1_1(weights=None)
   self.backbone.load_state_dict(state,strict=True)
   width=512
  else:
   from torchvision.models import resnet18
   self.backbone=resnet18(weights=None)
   self.backbone.load_state_dict(state,strict=True)
   width=128 if name=='resnet18_spatial84' else 256
  # Verify EVERY public tensor before discarding any classifier representation.
  loaded=self.backbone.state_dict();extra=set(loaded)-set(state)
  assert set(state)<=set(loaded)
  assert all(k.endswith('num_batches_tracked') and int(loaded[k])==0 for k in extra),extra
  assert all(torch.equal(loaded[k],v) for k,v in state.items())
  self.load_evidence={'file':str(path),'sha256':sha,'strict':True,'keys':len(state),'all_tensors_exact':True,'legacy_batchnorm_zero_counters':sorted(extra)}
  self.load_evidence['state_sha256']=digest(self.backbone)
  self.backbone.requires_grad_(False);self.backbone.eval()
  self.adapter=nn.Linear(width,128)
  self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
  self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
 def train(self,mode=True):
  super().train(mode);self.backbone.eval();return self
 def spatial(self,obs):
  if self.name=='resnet18_spatial84':
   # Fully-convolutional pretrained prefix at the ORIGINAL camera resolution.
   # This intentionally differs from the 224-pixel classification preset.
   x=(obs-self.mean)/self.std;b=self.backbone
   x=b.layer2(b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(x))))))
   return F.adaptive_avg_pool2d(x,(9,9)).flatten(2).transpose(1,2)
  if self.name=='dinov2_vits14':
   # 126 is divisible by the official patch size 14: native 9x9 spatial tokens.
   # Positional embeddings use the official multiresolution interpolation.
   x=F.interpolate(obs,size=(126,126),mode='bicubic',align_corners=False,antialias=True)
   x=(x-self.mean)/self.std
   return self.backbone.forward_features(x)['x_norm_patchtokens']
  # Official ImageNet resize-256 / center-crop-224 preprocessing.
  # For this fixed 84->256 UPSAMPLE, disabling AA differs by <=1.2e-7
  # on CPU and avoids unsupported NPU AA CPU round trips. No downsample here.
  x=F.interpolate(obs,size=(256,256),mode='bilinear',align_corners=False,antialias=False)[:,:,16:240,16:240]
  x=(x-self.mean)/self.std;b=self.backbone
  if self.name=='squeezenet1_1_imagenet':x=b.features(x)
  else:x=b.maxpool(b.relu(b.bn1(b.conv1(x))));x=b.layer3(b.layer2(b.layer1(x)))
  return F.adaptive_avg_pool2d(x,(9,9)).flatten(2).transpose(1,2)
 def forward(self,obs,flatten=True):
  if obs.dtype==torch.uint8:obs=obs.float()/255.
  with torch.no_grad():features=self.spatial(obs)
  result=self.adapter(features)
  assert result.shape[1:]==(81,128)
  return result.flatten(1) if flatten else result

class PretrainedQAgent(QAgent):
 def __init__(self,*args,visual_name,**kwargs):
  self.visual_name=visual_name
  super().__init__(*args,**kwargs)
 def _build_encoders(self,obs_shape):
  assert obs_shape==(3,84,84)
  return nn.ModuleList([SpatialPretrainedEncoder(self.visual_name).to(self.cfg.device) for _ in self.rl_cameras])

def make_agent(obs_shape,prop_shape,action_dim,cameras,cfg,visual_name='minivit'):
 if visual_name=='minivit':return QAgent(obs_shape,prop_shape,action_dim,cameras,cfg)
 assert not cfg.freeze_encoder,'freeze only backbone, leave spatial adapter trainable'
 cfg.enc_type=visual_name
 return PretrainedQAgent(obs_shape,prop_shape,action_dim,cameras,cfg,visual_name=visual_name)

def manifest(agent):
 enc=agent.encoders
 hashes=[digest(e.backbone) for e in enc if hasattr(e,'backbone')]
 assert hashes==[e.load_evidence['state_sha256'] for e in enc if hasattr(e,'backbone')],'pretrained weights overwritten after load'
 return {'name':getattr(agent,'visual_name','minivit'),'feature_shape_per_camera':[81,128],
  'parameters_total':sum(p.numel() for p in agent.parameters()),
  'parameters_trainable':sum(p.numel() for p in agent.parameters() if p.requires_grad),
  'encoder_trainable':sum(p.numel() for p in enc.parameters() if p.requires_grad),
  'encoder_frozen':sum(p.numel() for p in enc.parameters() if not p.requires_grad),
  'load_evidence':[getattr(e,'load_evidence',None) for e in enc],
  'backbone_hashes':hashes}
