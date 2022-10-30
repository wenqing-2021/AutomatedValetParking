from scipy import integrate
import numpy as np

x = np.linspace(0, 1, 100)
y = []
for i in x:
    y.append(np.sqrt(1+i))

length = integrate.simpson(y=y, x=x)

print(length)
