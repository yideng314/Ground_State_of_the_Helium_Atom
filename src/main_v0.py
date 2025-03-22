# Module: main_v0
# Author: Yi Deng <dengyi@stu.pku.edu.cn>

import numpy as np
import time
import matplotlib.pyplot as plt
from src.operations import add, conv, trim, save_to_json, load_from_json
from src.coeff_v0 import coeff_v0
from src.derivative_v0 import get_tensor

def plot_tensor_slices(A):
    """
    Function to plot each slice of the tensor A.
    
    Parameters:
    A (3D complex numpy array): A tensor.
    """
    num_slices = A.shape[0]
    for i in range(num_slices):
        plt.figure(figsize=(6, 6))
        slice_data = A[i]
        plt.imshow(np.log10(np.abs(slice_data)), cmap='viridis', interpolation='none')
        plt.colorbar(label='log10(abs)')
        plt.title(f'Slice {i}')
        plt.xlabel('m')
        plt.ylabel('l')
        plt.show()

def pad_tensor(A, new_shape):
    """
    Function to pad a 3D tensor with zeros.
    
    Parameters:
    A (3D complex numpy array): A tensor.
    new_shape (tuple): New shape for the tensor.
    
    Returns:
    A_padded (3D complex numpy array): Padded tensor.
    """
    pad_width = (
        (0, new_shape[0] - A.shape[0]),  # Pad the first dimension from the end
        ((new_shape[1] - A.shape[1]) // 2, (new_shape[1] - A.shape[1]) // 2),  # Pad the second dimension on both sides
        ((new_shape[2] - A.shape[2]) // 2, (new_shape[2] - A.shape[2]) // 2)   # Pad the third dimension on both sides
    )
    A_padded = np.pad(A, pad_width, mode='constant', constant_values=0)
    return A_padded

def residual(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, d_f13, d_bound, A, d):
    """
    Function to calculate the residual tensor.
    
    Parameters:
    f1, f2, ..., f12 (3D complex numpy array): Riccati coefficients defined by c constant.
    d_f13 (3D complex numpy array): d*f13 tensor.
    d_bound (3D complex numpy array): A tensor with the boundary condition.
    A (3D complex numpy array): A tensor.
    d (float): (-E)**(1/2).
    
    Returns:
    res (1D complex numpy array): Residual tensor flattened.
    """
    B, C, Az, B_theta, C_v, B_v = get_tensor(A)
    f13 = d_f13 / d
    bound = d_bound / d
    
    res = conv(f1, Az)
    res = add(res, conv(f2, conv(A, A)))
    res = add(res, conv(f3, A))
    res = add(res, conv(f4, B_theta))
    res = add(res, conv(f5, conv(B, B)))
    res = add(res, conv(f6, B))
    res = add(res, conv(f7, C_v))
    res = add(res, conv(f8, conv(C, C)))
    res = add(res, conv(f9, C))
    res = add(res, conv(f10, B_v))
    res = add(res, conv(f11, conv(B, C)))
    res = add(res, f12)
    res = add(res, f13)
    
    trim_res = trim(res, res.shape).flatten()*d**2
    bound_res = add(np.expand_dims(A[0, :, :], axis=0), -bound).flatten()
    
    total_res = np.concatenate((trim_res, bound_res))
    return total_res

def jacobian(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, d_f13, d_bound, A, d, h=1e-6):
    """
    Function to calculate the Jacobian matrix using numerical differentiation.
    
    Parameters:
    f1, f2, ..., f12 (3D complex numpy array): Riccati coefficients defined by c constant.
    d_f13 (3D complex numpy array): d*f13 tensor.
    d_bound (3D complex numpy array): A tensor with the boundary condition.
    A (3D complex numpy array): A tensor.
    d (float): (-E)**(1/2).
    h (float): Step size for numerical differentiation.
    
    Returns:
    J (2D complex numpy array): Jacobian matrix.
    """
    x0 = A.flatten()
    x = np.append(x0, d)
    res0 = residual(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, d_f13, d_bound, A, d)
    J = np.zeros((res0.size, x.size), dtype=complex)
    for i in range(x.size):
        x[i] += h
        A_new = x[:-1].reshape(A.shape)
        d_new = x[-1]
        res1 = residual(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, d_f13, d_bound, A_new, d_new)
        J[:, i] = (res1 - res0) / h
        x[i] -= h
    return J

def newton_iteration(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, d_f13, d_bound, A, d, tol=1e-8, max_iter=10):
    """
    Function to perform Newton's iteration to update A and d.
    
    Parameters:
    f1, f2, ..., f12 (3D complex numpy array): Riccati coefficients defined by c constant.
    d_f13 (3D complex numpy array): d*f13 tensor.
    d_bound (3D complex numpy array): A tensor with the boundary condition.
    A (3D complex numpy array): A tensor.
    d (float): (-E)**(1/2).
    tol (float): Tolerance for convergence.
    max_iter (int): Maximum number of iterations.
    
    Returns:
    A (3D complex numpy array): Updated A tensor.
    d (float): Updated d.
    """
    for _ in range(max_iter):
        res = residual(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, d_f13, d_bound, A, d)
        J = jacobian(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, d_f13, d_bound, A, d)

        J_pinv = np.linalg.pinv(J)  # Calculate the Moore-Penrose pseudoinverse
        delta = -np.dot(J_pinv, res)

        A += delta[:-1].reshape(A.shape)
        d += delta[-1]
        
        # print('norm of res', f'{np.linalg.norm(res):.10f}')
        print('newton info: norm of delta = ', f'{np.linalg.norm(delta):.2E}',' norm of res = ',f'{np.linalg.norm(res):.2E}')
        with open(log_file, 'a') as f:
            f.write(f'newton info: norm of delta = {np.linalg.norm(delta):.2E}, norm of res = {np.linalg.norm(res):.2E}\n')
        if np.linalg.norm(delta) < tol:
            break
    return A, d

def expand_and_iter(c, A_size_list, initial_d=1.7):
    """
    Function to expand A tensor using Newton's iteration based on A_size_list.
    
    Parameters:
    c (int): Constant in transformation.
    A_size_list (list of tuples): List of shapes for expanding A tensor, ending with 'END'.
    initial_d (float): Initial d value.
    
    Returns:
    A (3D complex numpy array): Final expanded A tensor.
    d (float): Final d value.
    """
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, d_f13, d_bound = coeff_v0(c)
    A = np.full(A_size_list[0], 0, dtype=complex)
    initial_A = load_from_json(f'results/dc_helium_c_4_shape_(3, 3, 3).json')
    A = add(A, initial_A)
    A = trim(A, A_size_list[0])
    
    d = initial_d
    
    for i in range(len(A_size_list)):
        # save the current A to json
        filename = f'results/helium_c_{c}_shape_{A_size_list[i]}.json'
        start_time = time.time()
        A, d = newton_iteration(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, d_f13, d_bound, A, d)
        end_time = time.time()
        E = -d**2
        print(f"Iteration {i}: A.shape {A_size_list[i]}, E {E}, Time taken {end_time - start_time:.2f} seconds")
        with open(log_file, 'a') as f:
            f.write(f"Iteration {i}: A.shape {A_size_list[i]}, E {E}, Time taken {end_time - start_time:.2f} seconds\n")
        
        if A_size_list[i+1] == 'END':
            save_to_json(A, filename)
            with open(log_file, 'a') as f:
                f.write('LOOP ENDS\n')
            print('LOOP ENDS')
            break
        save_to_json(A, filename)
        # Pad A tensor to the next shape
        A = pad_tensor(A, A_size_list[i + 1])
    
    return A, d

if __name__ == "__main__":
    c = 4
    l, m = 9, 5
    log_file = log_file = f'results/helium_c_{c}_l_{l}_m_{m}.txt'
    A_size_list = [(i,l,m) for i in range(2,11)] + ['END']
    A, d = expand_and_iter(c, A_size_list,initial_d=1.70403)
    # filename = 'results/helium_c_1_shape_(10, 3, 3).json'
    # A = load_from_json(filename)
    # print(A)
    # plot_tensor_slices(A)