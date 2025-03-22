# Module: derivative_v0
# Author: Yi Deng <dengyi@stu.pku.edu.cn>

import numpy as np

def cross_derivative(A):
    """
    Function to generate B and C from A for the cross derivative relation.
    
    Parameters:
    A (3D complex numpy array): A tensor.
    
    Returns:
    B (3D complex numpy array): B tensor.
    C (3D complex numpy array): C tensor.
    """
    # initialize B and C
    B = np.zeros(A.shape, dtype=complex)
    C = np.zeros(A.shape, dtype=complex)
    
    # create l_indices and m_indices with center symmetry and shift left by l//2, m//2
    l_indices = np.arange(A.shape[1]) - A.shape[1] // 2
    m_indices = np.arange(A.shape[2]) - A.shape[2] // 2
    l_indices, m_indices = np.meshgrid(l_indices, m_indices, indexing='ij')
    
    # when k = 0
    B[0, :, :] = 1j * l_indices * A[0, :, :] / 2
    C[0, :, :] = 1j * m_indices * A[0, :, :] / 2
    
    # when k > 0
    for k in range(1, A.shape[0]):
        B[k, :, :] = (1j * l_indices * A[k, :, :] + (k - 1) * B[k - 1, :, :]) / (k + 2)
        C[k, :, :] = (1j * m_indices * A[k, :, :] + (k - 1) * C[k - 1, :, :]) / (k + 2)
    
    return B, C

# calculate Az, B_theta, C_v, B_v
def derivative(A, B, C):
    """
    Function to calculate the derivative tensors.
    
    Parameters:
    A (3D complex numpy array): A tensor.
    B (3D complex numpy array): B tensor.
    C (3D complex numpy array): C tensor.
    
    Returns:
    Az (3D complex numpy array): Az tensor.
    B_theta (3D complex numpy array): B_theta tensor.
    C_v (3D complex numpy array): C_v tensor.
    B_v (3D complex numpy array): B_v tensor.
    """
    # initialize Az, B_theta, C_v, B_v
    Az = np.zeros(A.shape, dtype=complex)
    B_theta = np.zeros(A.shape, dtype=complex)
    C_v = np.zeros(A.shape, dtype=complex)
    B_v = np.zeros(A.shape, dtype=complex)
    
    # calculate Az
    Az[:-1, :, :] = np.arange(1, A.shape[0])[:, None, None] * A[1:, :, :]
    
    # calculate B_theta, C_v, B_v
    l_indices = np.arange(A.shape[1]) - A.shape[1] // 2
    m_indices = np.arange(A.shape[2]) - A.shape[2] // 2
    l_indices, m_indices = np.meshgrid(l_indices, m_indices, indexing='ij')
    
    B_theta = 1j * l_indices * B
    C_v = 1j * m_indices * C
    B_v = 1j * m_indices * B
    
    return Az, B_theta, C_v, B_v

def get_tensor(A):
    """
    Function to calculate all derivative tensors from A.
    
    Parameters:
    A (3D complex numpy array): A tensor.
    
    Returns:
    B (3D complex numpy array): B tensor.
    C (3D complex numpy array): C tensor.
    Az (3D complex numpy array): Az tensor.
    B_theta (3D complex numpy array): B_theta tensor.
    C_v (3D complex numpy array): C_v tensor.
    B_v (3D complex numpy array): B_v tensor.
    """
    B, C = cross_derivative(A)
    Az, B_theta, C_v, B_v = derivative(A, B, C)
    return B, C, Az, B_theta, C_v, B_v
