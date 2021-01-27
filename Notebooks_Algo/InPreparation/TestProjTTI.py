import sys; sys.path.insert(0,'../..')
import time
import numpy as np

from agd.Eikonal.HFM_CUDA import ProjectionTTI
from agd.Metrics.Seismic import Hooke

norm = Hooke.mica[0].rotate_by(0.8,(1,2,3))

start = time.time()
out = ProjectionTTI.ProjectionTTI(norm.hooke) #,samples=np.array([[0.1,0.2,0.3]]).T)
print(out)
print("Elapsed : ",time.time()-start)