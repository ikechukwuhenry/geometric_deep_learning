import numpy as np

# Define the unitary matrix
U = (1/np.sqrt(2)) * np.array([
    [1, 1j],
    [1j, 1]
])

# Compute the conjugate transpose (Hermitian) of U
U_dagger = np.conjugate(U.T)

# Verify that U * U† = I
check = np.allclose(U_dagger @ U, np.eye(2))

print("Is U unitary?:", check)

# Lets a create a reusable function to check unitarity
def is_unitary(U, tol=1e-10):
    U_dagger = np.conjugate(U.T)
    identity = np.eye(U.shape[0])
    return np.allclose(U_dagger @ U, identity, atol=tol)


# Example usage of the function
H = (1/np.sqrt(2)) * np.array([
    [1, 1],
    [1, -1]
])
print("Is H unitary?:", is_unitary(H))


# 45 degree rotation matrix is also unitary
rot_45 = (np.sqrt(2)/2) * np.array([
    [1, -1],
    [1, 1]
])
print("Is rot_45 unitary?:", is_unitary(rot_45))

# pauli matrices
sigma_x = np.array([[0, 1],
                    [1, 0]])

sigma_y = np.array([[0, -1j],
                    [1j, 0]])   

sigma_z = np.array([[1, 0],
                    [0, -1]])

print("Is sigma_x unitary?:", is_unitary(sigma_x))
print("Is sigma_y unitary?:", is_unitary(sigma_y))
print("Is sigma_z unitary?:", is_unitary(sigma_z))

# 4 x 4 discrete Fourier transform matrix
F_4 = (1/2) * np.array([
    [1, 1, 1, 1],
    [1, 1j, -1, -1j],
    [1, -1, 1, -1],
    [1, -1j, -1, 1j]
])

print("Is F_4 unitary?:", is_unitary(F_4))

