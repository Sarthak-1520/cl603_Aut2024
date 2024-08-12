
# %% [markdown]
# ### QUESTION #4

# %%
import numpy as np
from typing import Callable, List

def g(x_1 : float , x_2 : float) -> float:
    return (x_1**2 + x_2 - 11)**2 + (x_1 + x_2**2-7)**2


def gradient(g: Callable[[float, float], float], x: List[float], h: float = 1e-5) -> np.ndarray:
    """
    Compute the gradient of the function g at a given point x using the central difference formula.
    
    Parameters:
    g (Callable[[float, float], float]): The function for which the gradient is computed.
    x (List[float]): The point at which to compute the gradient.
    h (float, optional): The step size for central difference. Default is 1e-5.
    
    Returns:
    np.ndarray: The gradient vector at the point x.
    """
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_forward = np.copy(x)
        x_backward = np.copy(x)
        x_forward[i] += h
        x_backward[i] -= h
        grad[i] = (g(*x_forward) - g(*x_backward)) / (2 * h)
    return grad


def hessian(g: Callable[[float, float], float], x: List[float], h: float = 1e-5) -> np.ndarray:
    """
    Compute the Hessian matrix of the function g at a given point x using the central difference formula.
    
    Parameters:
    g (Callable[[float, float], float]): The function for which the Hessian is computed.
    x (List[float]): The point at which to compute the Hessian.
    h (float, optional): The step size for central difference. Default is 1e-5.
    
    Returns:
    np.ndarray: The Hessian matrix at the point x.
    """
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_ij_pp = np.copy(x)
            x_ij_pm = np.copy(x)
            x_ij_mp = np.copy(x)
            x_ij_mm = np.copy(x)
            
            x_ij_pp[i] += h
            x_ij_pp[j] += h
            
            x_ij_pm[i] += h
            x_ij_pm[j] -= h
            
            x_ij_mp[i] -= h
            x_ij_mp[j] += h
            
            x_ij_mm[i] -= h
            x_ij_mm[j] -= h
            
            hess[i, j] = (g(*x_ij_pp) - g(*x_ij_pm) - g(*x_ij_mp) + g(*x_ij_mm)) / (4 * h**2)
    
    return hess

# Test the functions at the point (1, 2)
x_test: List[float] = [1.0, 2.0]

grad_g: np.ndarray = gradient(g, x_test)
hess_g: np.ndarray = hessian(g, x_test)

# Round the output to 2 decimal places
grad_g = np.round(grad_g, 2)
hess_g = np.round(hess_g, 2)

# Assert the shapes are correct
assert grad_g.shape == (2,)
assert hess_g.shape == (2, 2)

# Assert the values are correct
assert np.array_equal(grad_g, np.array([-36., -32.]))
assert np.array_equal(hess_g, np.array([[-22., 12.], [12., 26.]]))

print("Gradient at (1, 2):", grad_g)
print("Hessian at (1, 2):\n", hess_g)

