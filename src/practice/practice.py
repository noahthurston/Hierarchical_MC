import numpy as np

a = np.zeros(3)
a.fill(1)


y = np.array([])
for mod in range(13):
    curr_mod = np.zeros(5000)
    curr_mod.fill(mod)
    y = np.append(y, curr_mod)

