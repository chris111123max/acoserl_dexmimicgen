import sys,json,numpy as np,mujoco
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/assembly_feasibility'))
from common import setup,save
E,D,R=setup();records=[]
try:
 for i in range(71):
  if i>0:R.step(D['actions'][i-1])
  if i not in [25,30,35,38,40,42,44,45,48,50,52,54,55,56,57,58,60,62,64,65,68,70]:continue
  row={'index':i,'arms':{}};m=E.sim.model;raw=m._model
  for ai,(a,o) in enumerate([('A','piece_1'),('B','piece_2')]):
   gp=E.robots[ai].gripper['right'].contact_geoms;op=getattr(E,o).contact_geoms;pairs=[]
   for gn in gp:
    g=m.geom_name2id(gn)
    for on in op:
     q=m.geom_name2id(on)
     if not (int(raw.geom_contype[g])&int(raw.geom_conaffinity[q]) or int(raw.geom_contype[q])&int(raw.geom_conaffinity[g])):continue
     gap=float(mujoco.mj_geomDistance(raw,E.sim.data._data,g,q,1.,None));pairs.append((gap,gn,on))
   rawcontacts=[]
   for j in range(E.sim.data.ncon):
    c=E.sim.data.contact[j];g1=m.geom_id2name(int(c.geom1));g2=m.geom_id2name(int(c.geom2))
    if (g1 in gp and g2 in op) or (g2 in gp and g1 in op):rawcontacts.append({'distance':float(c.dist),'geom1':g1,'geom2':g2})
   row['arms'][a]={'minimum_collision_enabled_distance':min(pairs) if pairs else None,'raw_contacts':rawcontacts}
  records.append(row)
 save(ROOT/'outputs/autoserl_two_point/proximity.json',records)
 for r in records:print(r['index'],[(a,x['minimum_collision_enabled_distance'],len(x['raw_contacts'])) for a,x in r['arms'].items()],flush=True)
finally:E.close()
