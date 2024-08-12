
# %% [markdown]
# ### QUESTION 2 

# %%
import numpy as np
import matplotlib.pyplot as plt

def q(x):
    return np.cos(x)**2 + 0.1 * x

x_range = np.linspace(-5, 5, 100)
y_range = q(x_range)

plt.plot(x_range, y_range)
plt.xlabel('x')
plt.ylabel('q(x)')
plt.show()
