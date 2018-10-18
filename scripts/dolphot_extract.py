import numpy as np
import matplotlib.pyplot as plt

data_dir = "..//data//"

data = np.genfromtxt(data_dir + "attempt1")
data = data[data[:,10]==1]

mags = data[:,11]
x = data[:,2]
y = data[:,3]

print(x)
print(y)

plt.scatter(x,y)
plt.show()