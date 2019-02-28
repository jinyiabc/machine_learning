
import numpy as np

x = np.array([0, 1, 2, 3])
y = np.array([1, 2, 3, 4])

# least square method
A = np.vstack([x, np.ones(len(x))]).T
coeff = np.linalg.lstsq(A, y, rcond=None)[0]
res = np.linalg.lstsq(A, y, rcond=None)[1]

# polyfit method
coeff1 = np.polyfit(x, y, 1, full=True)[0]
res1 = np.polyfit(x, y, 1, full=True)[1]