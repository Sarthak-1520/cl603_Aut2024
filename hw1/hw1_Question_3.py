
# %% [markdown]
# ### QUESTION #3 

# %%
import numpy as np
import matplotlib.pyplot as plt

def g(x_1 : float , x_2 : float) -> float:
    return (x_1**2 + x_2 - 11)**2 + (x_1 + x_2**2-7)**2

x_1_range = np.linspace(-6, 6, 100)
x_2_range = np.linspace(-6, 6, 100)

X_1, X_2 = np.meshgrid(x_1_range, x_2_range)
Y = g(X_1, X_2)

plt.contour(X_1, X_2, Y, alpha=0.5)

plt.imshow(Y, cmap='hot', extent=[-6, 6, -6, 6], alpha=0.5)
plt.colorbar(label='g(x_1, x_2)')

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()
