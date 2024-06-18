import matplotlib.pyplot as plt
from step02 import common

xs = common.load_sample_height_data()
print(xs.shape)

plt.hist(xs, bins="auto", density=True)
plt.xlabel("Height(cm)")
plt.ylabel("Probability Density")
plt.show()
