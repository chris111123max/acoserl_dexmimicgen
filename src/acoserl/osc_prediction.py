"""One-control-period OSC nominal response, not a contact dynamics certificate."""
import numpy as np

def predict(env,ai,action):
 ctl=env.robots[ai].part_controllers['right']
 assert ctl.input_type=='delta' and ctl.input_ref_frame=='base'
 site=env.sim.model.site_id2name(env.robots[ai].eef_site_id['right'])
 dt=1./float(env.control_freq);assert np.allclose(ctl.kp,ctl.kp[0]),'nominal screen requires isotropic Panda OSC gains';omega=np.sqrt(np.asarray(ctl.kp));response=1-(1+omega*dt)*np.exp(-omega*dt)
 v=np.asarray(env.sim.data.get_site_xvelp(site));w=np.asarray(env.sim.data.get_site_xvelr(site))
 scaled=np.asarray(action[7*ai:7*ai+6])*np.asarray(ctl.output_max);origin=np.asarray(ctl.origin_ori)
 inertia=v*dt*np.exp(-omega[:3]*dt)
 dp=response[:3]*(origin@scaled[:3])+inertia;dw=response[3:]*(origin@scaled[3:])+w*dt*np.exp(-omega[3:]*dt)
 return dict(displacement=dp,command_displacement=origin@scaled[:3],angular_displacement=dw,response=response,linear_inertia=inertia,control_dt_s=dt,physics_dt_s=float(env.sim.model._model.opt.timestep),estimate='fixed_OSC_one_period_nominal_not_contact_certificate')

def command_for_displacement(env,ai,action,desired):
 prediction=predict(env,ai,action);ctl=env.robots[ai].part_controllers['right'];result=np.asarray(action).copy()
 result[7*ai:7*ai+3]=np.asarray(ctl.origin_ori).T@((desired-prediction['linear_inertia'])/prediction['response'][:3])/np.asarray(ctl.output_max)[:3]
 return result
