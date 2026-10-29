from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];cfg=json.loads((ROOT/'configs/bimanual_autoserl/recovery_points.json').read_text());trace=np.load(ROOT/cfg['reference_path']);fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for ai,a in enumerate(['A','B']):
 d=json.loads((ROOT/f'outputs/autoserl_two_point/final_verified_{a}_s4.json').read_text());r=d['records'];pos=np.asarray([x['snapshot']['poses'][ai]['position'] for x in r]);ax=axes[0,ai];component=ai
 ax.plot(pos[:,component],pos[:,2],label='Actual control path')
 for key,marker in [('p0','s'),('p1','*')]:
  p=trace['positions'][cfg['arms'][a][key],ai];ax.scatter(p[component],p[2],marker=marker,s=70,label=key.upper())
 ax.set_xlabel(['world x (m)','world y (m)'][ai]);ax.set_ylabel('world z (m)');ax.set_title(a+' recovery: actual path');ax.legend()
 rr=[x for x in r if x['decision']['mode']=='Recover_'+a];axes[1,0].plot(np.arange(len(rr))*.05,[1000*x['after']['passive_error']['position_m'] for x in rr],label='Passive during Recover_'+a)
 if a=='B':
  for j,name in enumerate(['A','B']):axes[1,1].plot(np.arange(len(r))*.05,[x['after']['indices'][j] for x in r],label=name)
axes[1,0].axhline(2,color='red',linestyle='--',label='Unchanged 2mm limit');axes[1,0].set_title('Measured passive EEF drift');axes[1,0].set_xlabel('Recovery physical time (s)');axes[1,0].set_ylabel('Drift (mm)');axes[1,0].legend()
axes[1,1].set_title('Independent endpoints, not task completion');axes[1,1].set_xlabel('Physical control time since test start (s)');axes[1,1].set_ylabel('Reference state index');axes[1,1].legend()
for ax in axes.flat:ax.grid(alpha=.2)
fig.savefig(ROOT/'docs/autoserl_two_point/recovery_and_windows.png',dpi=160)
