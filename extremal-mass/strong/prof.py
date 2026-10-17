import time, numpy as np
from ode_poly import coeffs
from horizon_series import residual_series
from shoot import rhs
t=time.time()
for i in range(200): rhs(1.5, np.array([0.3,2.,1.]), 0.01)
print('rhs per call ms', (time.time()-t)/200*1e3)
t=time.time()
F=np.zeros(12,complex); H=np.zeros(13,complex); F[0]=4; H[0]=2
for i in range(5): residual_series(1.0, F, H, 0.01, 16)
print('residual_series K=16 per call s', (time.time()-t)/5)
