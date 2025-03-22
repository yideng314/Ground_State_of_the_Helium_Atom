# Module: coeff_v0
# Author: Yi Deng <dengyi@stu.pku.edu.cn>

import numpy as np
from src.operations import conv, add

def coeff_v0(c):
    """
    Function to get the coefficient of the tensor.
    
    Parameters:
    c (int): Integer to get the coefficient tensor.
    
    Returns:
    coeff_tensor (3D complex numpy array): Coefficient tensor.
    """
    dict = {}
    dict['1'] = np.array([[[1]]])
    dict['cos_theta'] = np.array([[[1/2],[0],[1/2]]])
    dict['sin_theta'] = np.array([[[1j/2],[0],[-1j/2]]])
    dict['cos_v'] = np.array([[[1/2,0,1/2]]])
    dict['sin_v'] = np.array([[[1j/2,0,-1j/2]]])
    
    dict['cos_2theta'] = add(2*conv(dict['cos_theta'],dict['cos_theta']),-dict['1'])
    dict['sin_2theta'] = 2*conv(dict['sin_theta'],dict['cos_theta'])
    dict['sin_v^2'] = conv(dict['sin_v'],dict['sin_v'])

    dict['M1'] = add(dict['cos_theta'],-dict['sin_theta'])
    dict['M2'] = add(dict['cos_theta'],dict['sin_theta'])
    dict['M1^2'] = conv(dict['M1'],dict['M1'])
    
    dict['1-cos_v'] = add(dict['1'],-dict['cos_v'])
    dict['1+cos_v'] = add(dict['1'],dict['cos_v'])
    dict['M3'] = add(conv(dict['1-cos_v'],dict['cos_theta']),conv(dict['1+cos_v'],dict['sin_theta']))/2
    
    dict['K'] = conv(dict['sin_v'],conv(dict['M1'],dict['M1']))
    dict['K'] = conv(dict['K'],dict['M2'])
    dict['K'] = conv(dict['K'],dict['M3'])
    
    dict['tan_2theta*K'] = conv(dict['sin_2theta'],dict['sin_v'])
    dict['tan_2theta*K'] = conv(dict['tan_2theta*K'],dict['M1'])
    dict['tan_2theta*K'] = conv(dict['tan_2theta*K'],dict['M3'])
    
    dict['F1*K'] = -4*dict['sin_v']
    dict['F2*K'] = -4*dict['cos_v']
    dict['F2*K'] = add(dict['F2*K'],-2*dict['cos_2theta']*dict['sin_v^2'])
    dict['F3*K'] = 2*conv(dict['sin_v^2'],dict['M1^2'])
    dict['d*F4*K'] = add(dict['cos_2theta'],-8*conv(dict['cos_theta'],dict['M3']))
    dict['d*F4*K'] = conv(dict['d*F4*K'],dict['sin_v'])
    dict['d*F4*K'] = conv(dict['d*F4*K'],dict['M1'])
    # d*F5 = (4*M1+3*M2+M1*cos_v)/4
    dict['d*F5'] = add(4*dict['M1'],3*dict['M2'])
    dict['d*F5'] = add(dict['d*F5'],conv(dict['M1'],dict['cos_v']))
    dict['d*F5'] = dict['d*F5']/4
    dict['-c^2*d*F5'] = -c**2*dict['d*F5']
    
    dict['z'] = np.array([[[0]],[[1]]])
    dict['(z-1)'] = add(dict['z'],-dict['1'])
    dict['(z-1)^2'] = conv(dict['(z-1)'],dict['(z-1)'])
    dict['(z-1)^3'] = conv(dict['(z-1)^2'],dict['(z-1)'])
    dict['z^2'] = conv(dict['z'],dict['z'])
    
    # f1 = (z-1)^3*z*K
    dict['(z-1)^3*z'] = conv(dict['(z-1)^3'],dict['z'])
    dict['f1'] = conv(dict['(z-1)^3*z'],dict['K'])
    # f2 = -z^2*K
    dict['f2'] = -conv(dict['z^2'],dict['K'])
    # f3 = -10*(z-1)^2*K
    dict['f3'] = -10*conv(dict['(z-1)^2'],dict['K'])
    # f4 = -4*(z-1)^2*K
    dict['f4'] = -4*conv(dict['(z-1)^2'],dict['K'])
    # f5 = -4*z^2*K
    dict['f5'] = -4*conv(dict['z^2'],dict['K'])
    # f6 = 16*(z-1)^2*tan_2theta*K
    dict['f6'] = 16*conv(dict['(z-1)^2'],dict['tan_2theta*K'])
    # f7 = 2*(z-1)^2*F1*K
    dict['f7'] = 2*conv(dict['(z-1)^2'],dict['F1*K'])
    # f8 = 2*z^2*F1*K
    dict['f8'] = 2*conv(dict['z^2'],dict['F1*K'])
    # f9 = 2*(z-1)^2*F2*K
    dict['f9'] = 2*conv(dict['(z-1)^2'],dict['F2*K'])
    # f10 = 2*(z-1)^2*F3*K
    dict['f10'] = 2*conv(dict['(z-1)^2'],dict['F3*K'])
    # f11 = 2*z^2*F3*K
    dict['f11'] = 2*conv(dict['z^2'],dict['F3*K'])
    # f12 = c^4*z^2*K
    dict['f12'] = c**4*conv(dict['z^2'],dict['K'])
    # d*f13 = 2*c^2*(z-1)^2*d*F4*K
    dict['d*f13'] = 2*c**2*conv(dict['(z-1)^2'],dict['d*F4*K'])

    f1 = dict['f1']
    f2 = dict['f2']
    f3 = dict['f3']
    f4 = dict['f4']
    f5 = dict['f5']
    f6 = dict['f6']
    f7 = dict['f7']
    f8 = dict['f8']
    f9 = dict['f9']
    f10 = dict['f10']
    f11 = dict['f11']
    f12 = dict['f12']
    d_f13 = dict['d*f13']
    
    return f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, d_f13, dict['-c^2*d*F5']
