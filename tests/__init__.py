import trchime as tce
import numpy as np

a = tce.random.randn(3, 4)

print(tce.var(a, axis = 1, keepdims = True))


