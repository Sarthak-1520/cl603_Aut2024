
# %% [markdown]
# ### QUESTION #1

# %%
import numpy as np 
import matplotlib.pyplot as plt

def f(x):
    return -0.1 * x / ((1 + 0.05 * x) * (1 + 0.1 * x))

x_range = np.linspace(0, 30, 100)
y_range = f(x_range)

plt.plot(x_range, y_range)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
