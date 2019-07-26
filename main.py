#plot test
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(0, 3.14*2, 50)
y1 = np.sin(x1)
y2 = np.cos(x1)
plt.plot(x1, y1, marker='o', label='sin')
plt.plot(x1, y2, marker='x', label='cos')
plt.title('Sin and Cos curve')
plt.xlabel('rad')
plt.legend(loc='upper right')
plt.show()
