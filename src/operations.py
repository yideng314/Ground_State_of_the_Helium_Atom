# Module: operations
# Author: Yi Deng <dengyi@stu.pku.edu.cn>

import numpy as np
import json

def conv(tensor1, tensor2):
    """
    Function to convolve two 3D tensors.
    
    Parameters:
    tensor1 (3D complex numpy array): First tensor to convolve.
    tensor2 (3D complex numpy array): Second tensor to convolve.
    
    Returns:
    result (3D complex numpy array): Convolution of tensor.
    """
    result_shape = np.array(tensor1.shape) + np.array(tensor2.shape) - 1
    result = np.zeros(result_shape, dtype=complex)
    for i in range(tensor1.shape[0]):
        for j in range(tensor1.shape[1]):
            for k in range(tensor1.shape[2]):
                result[i:i+tensor2.shape[0], j:j+tensor2.shape[1], k:k+tensor2.shape[2]] += tensor1[i,j,k]*tensor2
    return result

def add(tensor1, tensor2):
    """
    Function to add two 3D tensors, aligning them at the center with zero padding.
    
    Parameters:
    tensor1 (3D complex numpy array): First tensor to add.
    tensor2 (3D complex numpy array): Second tensor to add.
    
    Returns:
    result (3D complex numpy array): Sum of the tensors.
    """
    result_shape = np.maximum(tensor1.shape, tensor2.shape)
    result = np.zeros(result_shape, dtype=complex)
    # padding the tensors
    pad1 = np.array(result_shape - tensor1.shape)//2
    pad2 = np.array(result_shape - tensor2.shape)//2
    # the first dimension should be zero padded
    result[:tensor1.shape[0], pad1[1]:pad1[1]+tensor1.shape[1], pad1[2]:pad1[2]+tensor1.shape[2]] += tensor1
    result[:tensor2.shape[0], pad2[1]:pad2[1]+tensor2.shape[1], pad2[2]:pad2[2]+tensor2.shape[2]] += tensor2
    return result

def trim(tensor, shape):
    """
    Function to trim a 3D tensor to the specified shape, aligning it at the center.
    
    Parameters:
    tensor (3D complex numpy array): Tensor to be trimmed.
    shape (tuple): Desired 3D shape.
    
    Returns:
    trimmed_tensor (3D complex numpy array): Trimmed tensor.
    """
    # calculate the starting and ending indices for trimming
    start_indices = np.array((0, (tensor.shape[1] - shape[1]) // 2, (tensor.shape[2] - shape[2]) // 2))
    end_indices = start_indices + np.array(shape)
    # trim the tensor
    trimmed_tensor = tensor[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]]
    return trimmed_tensor

def save_to_json(tensor, filename):
    """
    Function to save a 3D tensor to a json file.
    
    Parameters:
    tensor (3D numpy.ndarray): The tensor to be saved.
    filename (str): The name of the json file.
    """
    array_list = tensor.tolist()
    array_str = [[[str(element) for element in row] for row in array] for array in array_list]
    with open(filename, 'w') as f:
        json.dump(array_str, f)

def load_from_json(filename):
    """
    Function to load a 3D tensor from a json file.
    
    Parameters:
    filename (str): The name of the json file.
    
    Returns:
    tensor (3D numpy.ndarray): The loaded tensor.
    """
    with open(filename, 'r') as f:
        array_str = json.load(f)
    array_list = [[[complex(element) for element in row] for row in array] for array in array_str]
    tensor = np.array(array_list, dtype=complex)
    return tensor
