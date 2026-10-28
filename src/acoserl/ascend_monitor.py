"""Driver telemetry is queried without importing torch_npu in another process."""
import subprocess,threading,time,json,re
from pathlib import Path
class Telemetry:
 def __init__(self,path,interval=5):
  self.path=Path(path);self.interval=interval;self.stop=threading.Event();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
 def loop(self):
  while not self.stop.is_set():
   row={'time':time.time()}
   try:
    raw=subprocess.check_output(['npu-smi','info'],stderr=subprocess.STDOUT,text=True,timeout=10);row['raw']=raw
    for line in raw.splitlines():
     if '0000:' in line:
      cells=[x.strip() for x in line.split('|')]
      values=re.fullmatch(r'\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s*',cells[3])
      if values:row.update(aicore_percent=int(values[1]),hbm_used_mb=int(values[4]),hbm_total_mb=int(values[5]))
      else:raise ValueError('Unrecognized driver metrics: '+line)
   except Exception as e:row['error']=repr(e)
   with self.path.open('a') as f:f.write(json.dumps(row)+'\n')
   self.stop.wait(self.interval)
 def close(self):self.stop.set();self.thread.join(timeout=12)
