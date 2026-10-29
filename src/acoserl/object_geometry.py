"""Object pose error, factored from P0 without importing historical trial artifacts."""
import numpy as np

def object_error(a,b):
    qa=np.asarray(a['quaternion_wxyz']);qb=np.asarray(b['quaternion_wxyz'])
    return {'position_m':float(np.linalg.norm(np.asarray(a['position'])-b['position'])),'rotation_rad':float(2*np.arccos(np.clip(abs(np.dot(qa,qb)),0,1)))}
