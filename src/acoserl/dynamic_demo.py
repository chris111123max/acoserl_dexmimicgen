"""Logical Demo pool = immutable original demo + actual intervention experience.

Official mode admits complete legal task interventions without requiring every
individual control to monotonically reduce TCP error. Safe-handback mode waits
for an observed successful handback. Both store full executed 14D transitions;
expert_action_mask is provenance metadata, NOT a masked RLPD loss or BC target.
"""
from pathlib import Path
import json
import torch
from acoserl.ascend_data import replay
from acoserl.isolated_nstep import IsolatedNStep
from acoserl.transition_semantics import Boundary,TASK_INTERVENTION_SOURCES,SAFETY_PROJECTION_SOURCES,source_family

class DynamicDemo:
    admission_protocol = 'task_interventions_reviewed_v3'
    def __init__(self,capacity=5000,original_fraction=.5,n=3,gamma=.99,
                 admission_mode='safe_handback',original_quota=None,
                 positive_quota=0,sampling_seed=0):
        assert capacity>0 and admission_mode in ('official','safe_handback')
        self.buffer=replay(capacity,None)
        self.original_fraction=original_fraction
        self.original_quota=round(128*original_fraction) if original_quota is None else original_quota
        assert self.original_quota in (64,96,128) and positive_quota in (0,2,4)
        self.positive_quota=positive_quota;self.admission_mode=admission_mode
        self.n,self.gamma=n,gamma;self.pending={};self.stream=IsolatedNStep(n,gamma)
        self.stream_last={};self.last_seen={};self.generator=torch.Generator().manual_seed(sampling_seed)
        self.stats=dict(accepted_segments=0,accepted_rows=0,rejected_segments=0,rejected_rows=0,
            rejected_partial_steps=0,duplicates=0,sampled_batches=0,sampled_original=0,
            sampled_dynamic=0,sampled_positive=0,stage_rows={},observed_complete_source_rows={},
            safety_projection_rows=0,safety_projection_excluded_rows=0,
            task_intervention_candidate_rows=0,accepted_action_sources={},quality_rejected_rows=0,
            quality_wait_rows=0,accepted_error_reduction_m=0.,safe_handbacks=0,
            dynamic_guide_rows=0,dynamic_recovery_rows=0,expert_action_dimensions=0)
        self.last_sample={};self.max_segment_steps=350;self._positive_cache=None

    def _close_official(self,env,bootstrap=True,kind=7):
        for out in self.stream.close(env,bootstrap=bootstrap,boundary_kind=kind):self.buffer.add(out)
        self.stream.episodes.pop(env,None);self.stream.frames.pop(env,None)
        self.stream.closed.discard(env);self.stream_last.pop(env,None)

    def reject(self,env,partial=False,bootstrap=True,kind=None):
        if self.admission_mode=='official':
            reason=int(Boundary.PARTIAL_PHYSICS if partial else Boundary.DEMO_SEGMENT_END) if kind is None else int(kind)
            self._close_official(env,bool(bootstrap and not partial),reason)
        rows=self.pending.pop(env,[])
        if rows:self.stats['rejected_segments']+=1;self.stats['rejected_rows']+=len(rows)
        self.stats['rejected_partial_steps']+=int(partial)

    def close_episode(self,env,bootstrap=False,kind=4):
        if self.admission_mode=='official':self._close_official(env,bootstrap,kind)
        else:self.reject(env)

    def _accepted(self,row):
        self.stats['accepted_rows']+=1
        self.stats['expert_action_dimensions']+=int(row['expert_action_mask'].sum())
        name=source_family(row['action_source'])
        table=self.stats['accepted_action_sources'];table[name]=table.get(name,0)+1
        self.stats['dynamic_guide_rows' if name.startswith('Guide_') else 'dynamic_recovery_rows']+=1
        if 'quality_error_before' in row.keys():
            self.stats['accepted_error_reduction_m']+=float(row['quality_error_before']-row['quality_error_after'])
        stages=self.stats['stage_rows'];key=str(int(row['stage_code']));stages[key]=stages.get(key,0)+1

    def _commit_pending(self,env):
        rows=self.pending.pop(env,[])
        if not rows:return
        assembler=None;previous=None
        for item in rows:
            identity=(int(item['episode']),int(item['frame']),int(item['takeover_id']))
            if previous is None or identity!=(previous[0],previous[1]+1,previous[2]):
                if assembler is not None:
                    for out in assembler.close(env,bootstrap=True,boundary_kind=7):self.buffer.add(out)
                assembler=IsolatedNStep(self.n,self.gamma)
            accepted=item.clone();accepted['source'].fill_(2);self._accepted(accepted)
            for out in assembler.append(accepted):self.buffer.add(out)
            previous=identity
        for out in assembler.close(env,bootstrap=True,boundary_kind=7):self.buffer.add(out)
        self.stats['accepted_segments']+=1

    def observe(self,row,handback=False,safe=True):
        env,ep,frame,source,takeover=[int(row[k]) for k in ('env_id','episode','frame','action_source','takeover_id')]
        if not bool(row['complete_physics']):self.reject(env,partial=True);return
        identity=(ep,frame);previous=self.last_seen.get(env)
        if previous==identity:self.stats['duplicates']+=1;return
        if previous is not None and previous>identity:raise ValueError('reordered intervention')
        if previous is not None and (ep!=previous[0] or frame!=previous[1]+1):self.reject(env)
        self.last_seen[env]=identity
        name=source_family(source);observed=self.stats['observed_complete_source_rows']
        observed[name]=observed.get(name,0)+1
        rows=self.pending.get(env,[])
        same=bool(rows and int(rows[-1]['episode'])==ep and int(rows[-1]['takeover_id'])==takeover)
        confirmed='quality_handoff_confirmed' in row.keys() and bool(row['quality_handoff_confirmed'])
        if handback and safe and confirmed:
            self.stats['safe_handbacks']+=1
            if source not in TASK_INTERVENTION_SOURCES and same:self._commit_pending(env)
        if not safe:
            # A real physical/supervisor failure closes pending intervention
            # tails without bootstrapping through that failure. The actual
            # failed complete row is retained by Online, never invented here.
            kind=int(row['next','boundary_kind']) if ('next','boundary_kind') in row.keys(True) else int(Boundary.PHYSICAL_FAILURE)
            self.reject(env,bootstrap=False,kind=kind)
            self.stats['rejected_rows']+=1
            return
        if source in SAFETY_PROJECTION_SOURCES:
            self.stats['safety_projection_rows']+=1;self.stats['safety_projection_excluded_rows']+=1
            self.reject(env);return
        if source not in TASK_INTERVENTION_SOURCES:self.reject(env);return
        self.stats['task_intervention_candidate_rows']+=1
        valid=('quality_validated' in row.keys() and bool(row['quality_validated']) and
               'expert_action_mask' in row.keys() and 0<int(row['expert_action_mask'].sum())<=7)
        if valid:
            before,after=float(row['quality_error_before']),float(row['quality_error_after'])
            valid=bool(torch.isfinite(torch.tensor([before,after])).all() and before>=0 and after>=0)
        waiting=('quality_handoff_wait' in row.keys() and bool(row['quality_handoff_wait'])
                 and not bool(row['next','done']))
        if not valid:
            self.stats['quality_rejected_rows']+=1
            if waiting and (same or not rows):
                self.stats['quality_wait_rows']+=1
                if self.admission_mode=='official':self._close_official(env)
                return
            self.reject(env);return
        if self.admission_mode=='official':
            previous=self.stream_last.get(env)
            if previous and previous!=(ep,frame-1,takeover):self._close_official(env)
            accepted=row.clone();accepted['source'].fill_(2);self._accepted(accepted)
            for out in self.stream.append(accepted):self.buffer.add(out)
            self.stream_last[env]=(ep,frame,takeover)
            if handback or bool(row['next','done']):
                self._close_official(env);self.stats['accepted_segments']+=1
            return
        rows=self.pending.setdefault(env,[])
        if rows and (int(rows[-1]['episode'])!=ep or int(rows[-1]['takeover_id'])!=takeover):
            self.reject(env);rows=self.pending.setdefault(env,[])
        rows.append(row.clone())
        if len(rows)>self.max_segment_steps or (bool(row['next','done']) and not handback):
            self.reject(env);return
        if handback and confirmed:self._commit_pending(env)

    def sample_demo(self,original,count):
        assert count==128 and len(original)==224
        original_count=128 if not len(self.buffer) else self.original_quota
        dynamic_count=count-original_count
        if self._positive_cache is None or self._positive_cache[0]!=id(original):
            positives=torch.nonzero(original[:]['next','reward']>0).flatten()
            if not len(positives):raise ValueError('original demo has no positive reward')
            self._positive_cache=(id(original),positives)
        positives=self._positive_cache[1];parts=[];forced=self.positive_quota
        if forced:
            ids=positives[torch.randint(len(positives),(forced,),generator=self.generator)]
            parts.append(original[ids].clone())
        if original_count-forced:
            ids=torch.randint(len(original),(original_count-forced,),generator=self.generator)
            parts.append(original[ids].clone())
        if dynamic_count:
            ids=torch.randint(len(self.buffer),(dynamic_count,),generator=self.generator)
            parts.append(self.buffer[ids].clone())
        batch=torch.cat(parts,0);batch=batch[torch.randperm(count,generator=self.generator)]
        self.last_sample=dict(original=original_count,dynamic=dynamic_count,
            positive=int((batch['next','reward']>0).sum()),forced_positive=forced,
            positive_oversampling=bool(forced),repeated_samples_are_not_new_experience=True)
        for k in ('original','dynamic','positive'):self.stats['sampled_'+k]+=self.last_sample[k]
        self.stats['sampled_batches']+=1
        return batch

    def save(self,directory):
        directory=Path(directory);directory.mkdir(exist_ok=False)
        if len(self.buffer):self.buffer.dumps(directory/'buffer')
        torch.save(dict(admission_protocol=self.admission_protocol,pending=self.pending,stats=self.stats,
            original_fraction=self.original_fraction,capacity=self.buffer._storage.max_size,n=self.n,gamma=self.gamma,
            admission_mode=self.admission_mode,original_quota=self.original_quota,positive_quota=self.positive_quota,
            generator=self.generator.get_state(),stream=self.stream,stream_last=self.stream_last,
            last_seen=self.last_seen,last_sample=self.last_sample),directory/'state.pt')
        (directory/'metadata.json').write_text(json.dumps(dict(rows=len(self.buffer),**self.stats),indent=2))

    def load(self,directory,discard_pending=True):
        directory=Path(directory);state=torch.load(directory/'state.pt',weights_only=False,map_location='cpu')
        if state.get('admission_protocol')!=self.admission_protocol:
            raise ValueError('incompatible intervention protocol; do not import old replay silently')
        for key in ('original_fraction','n','gamma','admission_mode','original_quota','positive_quota'):
            assert state[key]==getattr(self,key),key
        assert state['capacity']==self.buffer._storage.max_size
        for key in ('stats','pending','stream','stream_last','last_seen','last_sample'):setattr(self,key,state[key])
        self.generator.set_state(state['generator']);self._positive_cache=None
        if (directory/'buffer').exists():self.buffer.loads(directory/'buffer')
        if discard_pending:
            for env in list(self.stream.queues):self._close_official(env,True,6)
            for env in list(self.pending):self.reject(env)
