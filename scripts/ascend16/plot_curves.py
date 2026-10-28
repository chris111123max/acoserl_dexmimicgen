"""Export measured learning/throughput curves; missing evaluations stay missing."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--tag');a=p.parse_args();tag=a.tag or (root/'logs/ascend16/formal.tag').read_text().strip();out=root/'outputs/ascend16'/tag
rows=[]
f=out/'learning_curve.jsonl'
if f.exists():
 with f.open() as h:
  for line in h:
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:pass
assert rows,'No real collection records yet'
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for key,label in [('train/critic_loss','Critic'),('train/actor_loss_total','Actor')]:
 r=[x for x in rows if key in x];axes[0,0].plot([x['actual_interactions'] for x in r],[x[key] for x in r],label=label)
axes[0,0].set_title('Last update loss per collection round');axes[0,0].legend()
for key,label in [('actual_steps_s','Actual interactions/s'),('updates_s','Critic updates/s')]:axes[0,1].plot([x['actual_interactions'] for x in rows],[x[key] for x in rows],label=label)
axes[0,1].legend();axes[0,1].set_title('Cumulative effective throughput')
for name in sorted({k for r in rows for k in r.get('stage_counts',{})}):axes[1,0].plot([x['actual_interactions'] for x in rows],[x.get('stage_counts',{}).get(name,0) for x in rows],label=name)
axes[1,0].set_title('Cumulative frames with task event (not episode success)')
if any(r.get('stage_counts') for r in rows):axes[1,0].legend(fontsize=7)
evaluations=[json.loads(f.read_text()) for f in sorted(out.glob('evaluation_*.json'))];ev=[e for e in evaluations if e['complete']]
if ev:axes[1,1].plot([e['training_step'] for e in ev],[e['success_rate'] for e in ev],marker='o')
else:axes[1,1].text(.5,.5,'No completed intervention-free evaluation yet',ha='center',va='center',transform=axes[1,1].transAxes,fontsize=9)
axes[1,1].set_ylim(-.02,1.02);axes[1,1].set_title('Intervention-free full task success rate')
for ax in axes.flat:ax.set_xlabel('Actual training interactions');ax.grid(alpha=.2)
fig.suptitle(tag);fig.savefig(out/'learning_curves.png',dpi=160);print(out/'learning_curves.png')
