
# %% [markdown]
# ### QUESTION # 5

# %%
import numpy as np

def gaussian_elimination_positive_definite(A: np.ndarray) -> bool:
    """
    Determine if a given symmetric matrix A is positive definite using Gaussian elimination.
    
    Parameters:
    A (np.ndarray): The symmetric matrix to check.
    
    Returns:
    bool: True if the matrix is positive definite, False otherwise.
    """
    n = len(A)
    for i in range(n):
        # Pivot element should be positive for positive definiteness
        if A[i, i] <= 0:
            return False
        
        for j in range(i+1, n):
            # Multiplier for the row operation
            multiplier = A[j, i] / A[i, i]
            # Eliminate the element below the pivot
            A[j, i:] = A[j, i:] - multiplier * A[i, i:]
            
    # After elimination, check if all diagonal elements are positive
    return np.all(np.diag(A) > 0)

# Hessian matrix from the previous question
hessian_matrix = np.array([[-22., 12.], [12., 26.]])

assert hessian_matrix.any() == hessian_matrix.T.any()

# Check if the Hessian matrix is positive definite
is_pd = gaussian_elimination_positive_definite(hessian_matrix)

print("Is the Hessian matrix positive definite?", is_pd)

