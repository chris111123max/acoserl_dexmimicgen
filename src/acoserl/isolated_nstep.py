"""Collector-side n-step assembly; upstream ReplayBuffer and QAgent stay unchanged.

Keep n pending rows (one-step commit delay) so a before-step Safe_Stop can close
its preceding real transition before any window ending there has been emitted.
"""
from collections import deque
import torch

class IsolatedNStep:
 def __init__(self,n=3,gamma=.99):
  assert n>=1 and 0<gamma<1
  self.n=n;self.gamma=gamma;self.queues={};self.episodes={};self.frames={};self.closed=set()
 def _emit(self,q):
  rows=list(q)[:self.n];out=rows[0].clone();last=rows[-1]
  assert all(int(r['env_id'])==int(out['env_id']) and int(r['episode'])==int(out['episode']) for r in rows)
  reward=sum(self.gamma**i*r['next','reward'] for i,r in enumerate(rows))
  out['next']=last['next'].clone();out['next','reward']=reward
  out['next','original_reward']=rows[0]['next','reward'].clone()
  out['gamma']=torch.tensor(self.gamma**len(rows),dtype=torch.float32)
  out['steps_to_next_obs']=torch.tensor(len(rows),dtype=torch.long)
  out['nonterminal']=last['next','bootstrap_allowed'].clone() if ('next','bootstrap_allowed') in last.keys(True) else ~last['next','terminated'].clone()
  q.popleft();return out
 def append(self,row):
  assert not row.batch_size
  assert row.device in (None,torch.device('cpu'))
  env=int(row['env_id']);episode=int(row['episode']);frame=int(row['frame'])
  q=self.queues.setdefault(env,deque())
  if env in self.closed:
   assert episode!=self.episodes[env],'closed episode reused'
   self.closed.remove(env)
  if q:assert self.episodes[env]==episode and self.frames[env]+1==frame,'episode/frame discontinuity'
  elif env in self.episodes:
   assert episode!=self.episodes[env] or frame==self.frames[env]+1,'frame discontinuity'
  self.episodes[env]=episode;self.frames[env]=frame;q.append(row.clone())
  done=bool(row['next','done']);assert done==bool(row['next','terminated'] or row['next','truncated'])
  result=[]
  if done:
   self.closed.add(env)
   while q:result.append(self._emit(q))
  elif len(q)>self.n:result.append(self._emit(q))
  return result
 def close(self,env,bootstrap=True,terminated=False,boundary_kind=6):
  """Close only existing complete rows; no action or observation is fabricated."""
  q=self.queues.get(env,deque());result=[]
  if q:
   self.closed.add(env)
   if not bool(q[-1]['next','done']):
    q[-1]['next','done'].fill_(True)
    q[-1]['next','terminated'].fill_(terminated)
    q[-1]['next','truncated'].fill_(not terminated)
    if ('next','bootstrap_allowed') in q[-1].keys(True):
     q[-1]['next','bootstrap_allowed'].fill_(bootstrap)
     q[-1]['next','boundary_kind'].fill_(boundary_kind)
   while q:result.append(self._emit(q))
  return result
 def truncate(self,env):
  return self.close(env,bootstrap=True)
 def pending(self):return sum(len(q) for q in self.queues.values())
