import numpy as np
import scipy

from scipy import linalg, matrix, special
# from spherical_harmonics.np_spherical_harmonics import complex_sh_, real_sh_, complex_to_real_sh
# from sympy.physics.quantum.spin import Rotation
from scipy.spatial.transform.rotation import Rotation

def real_to_complex_sh(l):
    C = np.zeros(shape=(2*l+1, 2*l+1), dtype=np.complex64)
    c = 1./np.sqrt(2.)
    for m in range(1, l+1):
        C[l + m, l + m] = -1j * c
        C[l + m, l - m] = c
    for m in range(-l, 0):
        C[l + m, l + m] = ((-1)**m)*c
        C[l + m, l - m] = 1j*((-1) ** m)*c

    C[l, l] = 1.
    C = np.flip(C, 0)
    C = np.flip(C, 1)


    return np.asmatrix(C)

def complex_to_real_sh(l):
    return (real_to_complex_sh(l).conjugate()).T

def z_rot(a):
    c = np.cos(a)
    s = np.sin(a)
    return np.matrix([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

def y_rot(a):
    c = np.cos(a)
    s = np.sin(a)
    return np.matrix([[c, 0., -s], [0., 1., 0.], [s, 0., c]])

def euler_rot_zyz(a, b ,c):
    return np.matmul(np.matmul(z_rot(a), y_rot(b)), z_rot(c))


def complex_wigner_2_(a, b, c):
    ea = np.exp(1j*a)
    eb = np.exp(1j*b)
    ec = np.exp(1j*c)

    e_a = np.exp(-1j*a)
    e_b = np.exp(-1j*b)
    e_c = np.exp(-1j*c)

    e2a = np.exp(1j*2.*a)
    e2b = np.exp(1j*2.*b)
    e2c = np.exp(1j*2.*c)

    e_2a = np.exp(-1j*2.*a)
    e_2b = np.exp(-1j*2.*b)
    e_2c = np.exp(-1j*2.*c)

    sa = np.imag(ea)
    ca = np.real(ea)

    # sb = np.imag(eb)
    # cb = np.real(eb)
    sb = np.sin(b)
    cb = np.cos(b)

    sc = np.imag(ec)
    cc = np.real(ec)

    # c2b = np.real(e2b)
    # s2b = np.imag(e2b)
    c2b = np.cos(2.*b)
    s2b = np.sin(2.*b)
    
    d22 = ((1+cb)*(1.+cb))/4.
    d21 = -sb*(1.+cb)/2.
    d20 = np.sqrt(3./8.)*sb*sb
    d2_1 = -sb*(1.-cb)/2.
    d2_2 = (1.-cb)*(1.-cb)/4.
    d11 = (2.*cb*cb+cb-1.)/2.
    d10 = -np.sqrt(3./8.)*s2b
    d1_1 = (-2.*cb*cb+cb+1.)/2.
    d00 = (3.*cb*cb-1.)/2.

    d = np.asmatrix([[d22, -d21, d20, -d2_1, d2_2],
                     [d21, d11, -d10, d1_1, -d2_1],
                     [d20, d10, d00, -d10, d20],
                     [d2_1, d1_1, d10, d11, -d21],
                     [d2_2, d2_1, d20, d21, d22]], dtype=np.complex64)

    d = d.T




    Ea = np.asmatrix([[e_2a, 0., 0., 0., 0.],
                      [0., e_a, 0., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., 0., ea, 0.],
                      [0., 0., 0., 0., e2a]], dtype=np.complex64)

    Ec = np.asmatrix([[e_2c, 0., 0., 0., 0.],
                      [0., e_c, 0., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., 0., ec, 0.],
                      [0., 0., 0., 0., e2c]], dtype=np.complex64)


    """
    Ea = np.asmatrix([[e2a, 0., 0., 0., 0.],
                      [0., ea, 0., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., 0., e_a, 0.],
                      [0., 0., 0., 0., e_2a]], dtype=np.complex64)

    Ec = np.asmatrix([[e2c, 0., 0., 0., 0.],
                      [0., ec, 0., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., 0., e_c, 0.],
                      [0., 0., 0., 0., e_2c]], dtype=np.complex64)
    """

    return np.matmul(np.matmul(Ea, d), Ec)

def complex_wigner_1_(a, b, c):
    cb = np.cos(b)
    sb = np.sin(b)

    ea = np.exp(1j * a)
    ec = np.exp(1j * c)

    e_a = np.exp(-1j * a)
    e_c = np.exp(-1j * c)

    d11 = (1.+cb)/2.
    d10 = -sb/(np.sqrt(2.))
    d1_1 = (1.-cb)/2.
    d00 = cb

    d = np.asmatrix([[d11, -d10, d1_1],
                     [d10, d00, -d10],
                     [d1_1, d10, d11]], dtype=np.complex64)

    d = d.T

    Ea = np.asmatrix([[e_a, 0., 0.],
                     [0., 1., 0.],
                     [0., 0., ea]], dtype=np.complex64)

    Ec = np.asmatrix([[e_c, 0., 0.],
                     [0., 1., 0.],
                     [0., 0., ec]], dtype=np.complex64)

    return np.matmul(np.matmul(Ea, d), Ec)





def complex_wigner_(l, a, b, c):
    assert (l == 0 or l == 1 or l == 2)
    if l == 0:
        return np.asmatrix([[1.]], dtype=np.complex64)
    if l == 1:
        return complex_wigner_1_(a, b, c)
    if l == 2:
        return complex_wigner_2_(a, b, c)

def wigner_d_matrix_coeffs(l, j, k, b):
    p = np.math.factorial(l+j)*np.math.factorial(l-j)*np.math.factorial(l+k)*np.math.factorial(l-k)
    p = np.sqrt(p)

    # l + k - s >= 0
    # s >= 0
    # j - k + s >= 0
    # l - j - s >= 0

    # l + k >= s
    # s >= 0
    # s >= k - j
    # l - j >= s

    s1 = np.max([0, k-j])
    s2 = np.min([l+k, l-j])
    s_ = np.sin(b/2.)
    c_ = np.cos(b/2.)
    d = 0.
    for s in range(s1, s2+1):
        q = np.math.factorial(l+k-s)*np.math.factorial(s)*np.math.factorial(j-k+s)*np.math.factorial(l-j-s)
        x = (1.*p)/(1.*q)
        x *= (-1)**(j-k+s)
        x *= (c_**(2*l+k-j-2*s))*(s_**(j-k+2*s))
        d += x
    return d

def wigner_d_matrix(l, b, dtype=np.float32):
    d = np.zeros(shape=(2*l+1, 2*l+1), dtype=dtype)
    """
    for m in range((2*l+1)*(2*l+1)):
        k = m % (2*l+1)
        j = np.int((m - k) / (2*l+1))
        d[j, k] = wigner_d_matrix_coeffs(l, j-l, k-l, b)
    """
    for j in range(2*l+1):
        for k in range(2*l+1):
            d[j, k] = wigner_d_matrix_coeffs(l, j-l, k-l, b)
    return np.asmatrix(d)

def diag_exp(l, a):
    e = np.zeros(shape=(2*l+1, 2*l+1), dtype=np.complex64)

    for m in range(l+1):
        e[m + l, m + l] = np.exp(m * 1j * a)
        e[m, m] = np.exp((m - l) * 1j * a)


    return np.asmatrix(e)


"""
def complex_D_wigner(l, a, b, c):
    D = diag_exp(l, a)*wigner_d_matrix(l, b, dtype=np.complex64)*diag_exp(l, c)
    return np.conjugate(D)
"""

def complex_D_wigner(l, a, b, c):

    d = wigner_d_matrix(l, b, dtype=np.complex64)
    ea = diag_exp(l, a)
    ec = diag_exp(l, c)
    # D = np.matmul(np.matmul(ea, d), ec)
    D = d

    for p in range(2*l+1):
        for q in range(2*l+1):
            D[q, p] *= np.exp(-(p-l)*1j*a)*np.exp(-(q-l)*1j*c)
    # np.conjugate(D)
    # print(D)
    # D = np.flip(D, axis=0)
    # D = np.flip(D, axis=1)
    # D = np.conjugate(D)
    return D

def real_D_wigner_(l, a, b, c):
    C = complex_to_real_sh(l)
    D = complex_D_wigner(l, a, b, c)
    # return np.conjugate(C.T)*D*C
    return np.real(C*D*np.conjugate(C.T))

def real_D_wigner_from_euler(l_max, a, b, c):
    D = np.zeros(((l_max+1)**2, (l_max+1)**2))
    k = 0
    for l in range(l_max+1):
        D[k:k+(2*l+1), k:k+(2*l+1)] = real_D_wigner_(l, a, b, c)
        k += 2*l+1
    return D

def real_D_wigner_from_quaternion(l_max, q):
    r = Rotation(q)
    euler = r.as_euler('zyz')
    return real_D_wigner_from_euler(l_max, euler[0], euler[1], euler[2])






