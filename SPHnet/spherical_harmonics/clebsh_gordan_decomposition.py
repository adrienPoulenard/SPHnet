import numpy as np
import tensorflow as tf
from sphericalHarmonicsFlow.wigner_matrix import complex_wigner_, complex_D_wigner, real_D_wigner, euler_rot_zyz
from sphericalHarmonicsFlow.spherical_harmonics import complex_to_real_sh, real_to_complex_sh, complex_sh_

import scipy
from scipy import linalg, matrix, special

# https://en.wikipedia.org/wiki/Table_of_Clebsch%E2%80%93Gordan_coefficients#_j2=0

#  ⟨ j1 m1 j2 m2 | j3 m3 ⟩
#  ⟨ j1,j2; m1,m2 | j1,j2; J,M ⟩
# j1 = j1, j2 = j2, m1 = m1, m2 = m2, j3 = J, m3 = M

# symmetries:
# ⟨ j1 m1 j2 m2 | j3 m3 ⟩ = (-1)**(j3-j1-j2)*⟨ j1 -m1 j2 -m2 | j3 -m3 ⟩
# ⟨ j1 m1 j2 m2 | j3 m3 ⟩ = (-1)**(j3-j1-j2)*⟨ j2 m1 j1 m2 | j3 m3 ⟩

# when j2 = 0 the Clebsch–Gordan coefficients are given by deltaj3j1*deltam3m1



def clebsch_gordan_(j1, j2, J, m1, m2, M):
    # d = float((M == m1 + m2))
    if M != m1 + m2:
        return 0.0

    A = float((2*J+1)*np.math.factorial(J+j1-j2)*np.math.factorial(J-j1+j2)*np.math.factorial(j1+j2-J))
    A /= np.math.factorial(J+j1+j2+1)

    B = float(np.math.factorial(J+M)*np.math.factorial(J-M)*np.math.factorial(j1-m1)*
              np.math.factorial(j1+m1)*np.math.factorial(j2-m2)*np.math.factorial(j2+m2))
    C = 0.

    b0 = (j1+j2-J)
    b1 = (j1-m1)
    b2 = (j2+m2)

    a0 = 0
    a1 = (J-j2+m1)
    a2 = (J-j1-m2)

    k2 = np.min([b0, b1, b2])
    k1 = np.max([-a0, -a1, -a2])

    for k in range(k1, k2+1):
        a0_ = np.math.factorial(k+a0)
        a1_ = np.math.factorial(k+a1)
        a2_ = np.math.factorial(k+a2)

        b0_ = np.math.factorial(b0-k)
        b1_ = np.math.factorial(b1-k)
        b2_ = np.math.factorial(b2-k)

        C += ((-1)**k)/(float(a0_*a1_*a2_*b0_*b1_*b2_))

    return np.sqrt(A * B) * C


def clebsch_gordan(j1, j2, J, m1, m2, M):
    if M < 0:
        if j1 >= j2:
            return (-1.)**(J-j1-j2)*clebsch_gordan_(j1, j2, J, -m1, -m2, -M)
        else:
            return clebsch_gordan_(j2, j1, J, -m2, -m1, -M)
    else:
        if j1 >= j2:
            return clebsch_gordan_(j1, j2, J, m1, m2, M)
        else:
            return (-1.) ** (J - j1 - j2) * clebsch_gordan_(j2, j1, J, m2, m1, M)

# tensor product decomposition matrix
#  Q*kron(Dj1, Dj2)*Q.T = DJ
def Q_from_cb(j1, j2, J, dtype=np.float32):
    Q = np.zeros(shape=(2 * J + 1, (2 * j1 + 1) * (2 * j2 + 1)), dtype=dtype)
    for m1 in range(-j1, j1 + 1):
        for m2 in range(-j2, j2 + 1):
            m3 = m1 + m2
            if -J <= m3 <= J:
                Q[m3 + J, (2 * j2 + 1) * (m1 + j1) + (m2 + j2)] = clebsch_gordan(j1, j2, J, m1, m2, m3)
    return Q


def Q_from_cb_(j1, j2, J, dtype=np.float32):
    # Q = np.zeros(shape=(2 * J + 1, (2 * j1 + 1), (2 * j2 + 1)), dtype=dtype)
    Q = np.zeros(shape=(2 * J + 1, (2 * j1 + 1), (2 * j2 + 1)), dtype=dtype)
    for m1 in range(-j1, j1 + 1):
        for m2 in range(-j2, j2 + 1):
            m3 = m1 + m2
            if -J <= m3 <= J:
                Q[m3 + J, m1 + j1, m2 + j2] = clebsch_gordan(j1, j2, J, m1, m2, m3)
    return Q

# tensor product decomposition matrix for real Wigner
#  Q*kron(Dj1, Dj2)*Q.T = DJ
def real_Q_from_cb(j1, j2, J, dtype=np.float32):
    Q = np.zeros(shape=(2 * J + 1, (2 * j1 + 1) * (2 * j2 + 1)), dtype=dtype)
    for m1 in range(-j1, j1 + 1):
        for m2 in range(-j2, j2 + 1):
            m3 = m1 + m2
            if -J <= m3 <= J:
                Q[m3 + J, (2 * j2 + 1) * (m1 + j1) + (m2 + j2)] = clebsch_gordan(j1, j2, J, m1, m2, m3)

    CRj1 = complex_to_real_sh(j1)
    RCj1 = np.conjugate(CRj1.T)

    CRj2 = complex_to_real_sh(j2)
    RCj2 = np.conjugate(CRj2.T)

    CRJ = complex_to_real_sh(J)
    # RCJ = np.conjugate(CRJ.T)
    Q = np.matmul(np.matmul(CRJ, Q), np.kron(RCj1, RCj2))
    # print('uu')
    # print(np.linalg.norm(Q-np.real(Q)))
    return Q
    # return np.matmul(np.matmul(RCJ, Q), np.kron(CRj1, CRj2))

# tensor product decomposition matrix for real Wigner
#  Q*kron(Dj1, Dj2)*Q.T = DJ
def real_Q_from_cb_(j1, j2, J, dtype=np.float32):
    Q = Q_from_cb_(j1, j2, J, dtype=dtype)

    CRj1 = complex_to_real_sh(j1)
    RCj1 = np.conjugate(CRj1.T)

    CRj2 = complex_to_real_sh(j2)
    RCj2 = np.conjugate(CRj2.T)

    CRJ = complex_to_real_sh(J)
    # RCJ = np.conjugate(CRJ.T)
    Q = np.reshape(Q, newshape=(2*abs(J)+1, -1))
    Q = np.matmul(np.matmul(CRJ, Q), np.kron(RCj1, RCj2))
    Q = np.reshape(np.asarray(Q), newshape=(2*abs(J)+1, 2*abs(j1)+1, 2*abs(j2)+1))
    # print('uu')
    # print(np.linalg.norm(Q-np.real(Q)))
    return Q
    # return np.matmul(np.matmul(RCJ, Q), np.kron(CRj1, CRj2))



def tf_clebsh_gordan_matrices_(max_degree, dtype=tf.complex64):
    assert(max_degree >= 0)
    keys = []
    values = []
    for j in range(max_degree+1):
        for k in range(max_degree+1):
            for J in range(abs(k-j), k+j+1):
                # keys.append('('+str(j)+','+str(k)+','+str(J)+')')
                keys.append(str(j) + '_' + str(k) + '_' + str(J))
                QJ = Q_from_cb(j, k, J)
                values.append(tf.convert_to_tensor(QJ, dtype=dtype))

    return dict(zip(keys, values))

def tf_clebsh_gordan_matrices__(max_degree, dtype=tf.complex64):
    assert(max_degree >= 0)
    keys = []
    values = []
    for j in range(max_degree+1):
        for k in range(max_degree+1):
            for J in range(abs(k-j), k+j+1):
                # keys.append('('+str(j)+','+str(k)+','+str(J)+')')
                keys.append(str(j) + '_' + str(k) + '_' + str(J))
                QJ = Q_from_cb_(j, k, J)
                values.append(tf.convert_to_tensor(QJ, dtype=dtype))

    return dict(zip(keys, values))


class tfClebshGordanMatrices:
    def __init__(self, max_degree, dtype=tf.complex64):
        self.dtype = dtype
        self.Q = tf_clebsh_gordan_matrices__(max_degree, dtype=dtype)

    def getMatrix(self, j, k, J):
        key = str(j) + '_' + str(k) + '_' + str(J)
        if key in self.Q:
            return self.Q[key]
        else:
            Q_jkJ = Q_from_cb_(j, k, J, dtype=np.float32)
            self.Q[key] = tf.convert_to_tensor(Q_jkJ, dtype=self.dtype)
            return self.Q[key]


def np_clebsh_gordan_matrix(j, k, J, matrix_representation=True, is_real=False, dtype=np.complex64):
    if matrix_representation:
        if is_real:
            QJ = real_Q_from_cb(j, k, J, dtype=dtype)
        else:
            QJ = Q_from_cb(j, k, J, dtype=dtype)
    else:
        if is_real:
            QJ = real_Q_from_cb_(j, k, J, dtype=dtype)
        else:
            QJ = Q_from_cb_(j, k, J, dtype=dtype)
    return QJ

def np_clebsh_gordan_matrices__(max_degree, matrix_representation=True, is_real=False, dtype=np.complex64):
    assert(max_degree >= 0)
    keys = []
    values = []
    for j in range(max_degree+1):
        for k in range(max_degree+1):
            for J in range(abs(k-j), k+j+1):
                # keys.append('('+str(j)+','+str(k)+','+str(J)+')')
                keys.append(str(j) + '_' + str(k) + '_' + str(J))
                QJ = np_clebsh_gordan_matrix(j, k, J,
                                             matrix_representation=matrix_representation,
                                             is_real=is_real,
                                             dtype=dtype)
                values.append(QJ)
                print('j: ', j, ' k: ', k, ' J: ', J)
                # print(QJ)
                print(np.real(QJ)+np.imag(QJ))

    return dict(zip(keys, values))


class npClebshGordanMatrices:
    def __init__(self, max_degree, matrix_representation=True, real=False, dtype=np.complex64):
        self.dtype = dtype
        self.real = real
        self.matrix_representation = matrix_representation
        self.Q = np_clebsh_gordan_matrices__(max_degree,
                                             matrix_representation=matrix_representation,
                                             is_real=real,
                                             dtype=dtype)

    def getMatrix(self, j, k, J):
        key = str(j) + '_' + str(k) + '_' + str(J)
        if key in self.Q:
            return self.Q[key]
        else:
            Q_jkJ = np_clebsh_gordan_matrix(j, k, J,
                                            matrix_representation=self.matrix_representation,
                                            is_real=self.real,
                                            dtype=self.dtype)
            self.Q[key] = Q_jkJ
            return self.Q[key]


def tf_clebsh_gordan_matrices(max_degree, dtype=tf.complex64):
    assert(max_degree >= 0)
    Q = []
    idx_min = []
    idx_max = []
    for j in range(max_degree+1):
        for k in range(max_degree+1):
            J_max = int(np.min([max_degree + 1, j + k + 1]))
            Qjk = []
            a = 0
            b = 0
            for J in range(abs(k-j), J_max):
                a = b
                b += J
                QJ = Q_from_cb(j, k, J)
                Qjk.append(QJ)

            Qjk = np.concatenate(Qjk, axis=0)
            Q.append(tf.convert_to_tensor(Qjk, dtype=dtype))
    return Q





"""
def tf_clebsh_gordan_matrices(input_degrees , dtype=tf.complex64):
    assert(max_degree >= 0)
    keys = []
    values = []
    for j in range(max_degree+1):
        for k in range(max_degree+1):
            for J in range(abs(k-j), k+j+1):
                keys.append('('+str(j)+','+str(k)+','+str(J)+')')
                QJ = Q_from_cb(j, k, J)
                values.append(tf.convert_to_tensor(QJ, dtype=dtype))

    return dict(zip(keys, values))
"""


def real_conj(A, Q):
    return np.matmul(Q.T, np.matmul(A, Q))

def complex_conj(A, Q):
    return np.matmul(np.conjugate(Q.T), np.matmul(A, Q))

def unit_test4():

    j1 = 2
    j2 = 2
    J = 2
    # cb_dict = clebsch_gordan_dict()

    Q = np.asmatrix(Q_from_cb(j1, j2, J, dtype=np.complex64))
    # Q = np.sqrt(2.) * Q

    # Q_ = np.asmatrix(Q_from_cb_dict(j1, j2, J, cb_dict, dtype=np.complex64))
    # Q_ = np.sqrt(2.)*Q_

    # Q__ = tensorProductDecompose_(j1, j2, J)
    # Q__ = np.sqrt(1./2.49634557e-02)*Q__

    angles = np.random.rand(3)
    # angles = [1., 0., 0.]


    Dj1 = complex_wigner_(j1, angles[0], angles[1], angles[2])
    Dj2 = complex_wigner_(j2, angles[0], angles[1], angles[2])
    DJ = complex_wigner_(J, angles[0], angles[1], angles[2])



    print('eee')

    prod = np.kron(Dj1, Dj2)

    y = np.matmul(np.matmul(Q, prod), Q.T) - DJ

    # print(y)
    # print(np.matmul(Q.T, Q))
    # print(np.matmul(Q, Q.T))

    # print(np.real(prod))
    # print(np.real(Q))
    # print(np.real(Q__))
    # y = np.matmul(Q, prod) - np.matmul(DJ, Q)
    # y = np.matmul(y, Q.T)
    # print(np.linalg.norm(Q - Q_, 'fro'))
    print(np.linalg.norm(y))
    print(np.linalg.norm(DJ))




def unit_test5():
    j1 = 1
    j2 = 1

    angles = np.random.rand(3)
    # angles = [1.0, 0.0, 0.]

    D0 = np.asmatrix([[1.]], dtype=np.complex64)
    D1 = complex_wigner_(1, angles[0], angles[1], angles[2])
    D2 = complex_wigner_(2, angles[0], angles[1], angles[2])

    D = [D0, D1, D2]

    prod = np.kron(D[j1], D[j2])
    # prod = np.kron(D[j2], D[j1])

    c = 0.0
    for m1 in range(-j1, j1+1):
        for k1 in range(-j1, j1+1):
            for m2 in range(-j2, j2+1):
                for k2 in range(-j2, j2+1):
                    a = D[j1][j1 + m1, j1 + k1] * D[j2][j2 + m2, j2 + k2]
                    b = 0.
                    # b = prod[(2*j2+1)*(m1+j1) + (m2+j2), (2*j2+1)*(k1+j1) + (k2+j2)]

                    for J in range(abs(j1-j2), j1+j2+1):
                        if(2*J >= m1+m2+J >= 0 and 2*J >= k1+k2+J >= 0):
                            b += D[J][m1+m2+J, k1+k2+J]*clebsch_gordan(j1, j2, J, m1, m2, m1+m2)*clebsch_gordan(j1, j2, J, k1, k2, k1+k2)

                    print('zz')
                    print(a)
                    print(b)
                    print(a-b)

                    c += abs(np.real(a-b))*abs(np.real(a-b))+abs(np.imag(a-b))*abs(np.imag(a-b))

    print('rr')
    print(np.sqrt(c))



def unit_test6():
    angles = np.random.rand(3)
    # angles = [0., 1., 0.]

    Q0 = np.asmatrix(Q_from_cb(1, 1, 0, dtype=np.complex64))
    Q1 = np.asmatrix(Q_from_cb(1, 1, 1, dtype=np.complex64))
    Q2 = np.asmatrix(Q_from_cb(1, 1, 2, dtype=np.complex64))

    D0 = np.asmatrix([[1.]], dtype=np.complex64)
    D1 = complex_wigner_(1, angles[0], angles[1], angles[2])
    D2 = complex_wigner_(2, angles[0], angles[1], angles[2])

    y = np.kron(D1, D1) - real_conj(D2, Q2) - real_conj(D1, Q1) - real_conj(D0, Q0)

    print(np.linalg.norm(y))

def tensor_decomposition_unit_test___(j, k, J, a, b, c):
    Dj = complex_D_wigner(j, a, b, c)
    Dk = complex_D_wigner(k, a, b, c)
    DJ = complex_D_wigner(k, a, b, c)
    assert(j+k >= J >= abs(k-j))
    QJ = Q_from_cb(j, k, J)

    y = real_conj(np.kron(Dj, Dk), QJ.T) - DJ
    print(np.linalg.norm(y))


def tensor_decomposition_unit_test__(j, k, a, b, c):
    Dj = complex_D_wigner(j, a, b, c)
    Dk = complex_D_wigner(k, a, b, c)

    D_ = np.zeros(shape=((2*j+1)*(2*k+1), (2*j+1)*(2*k+1)), dtype=np.complex64)
    D = np.kron(Dj, Dk)

    for J in range(abs(k-j), k+j+1):
        print('j = ', j, 'k = ', k, 'J = ', J)
        DJ = complex_D_wigner(J, a, b, c)
        QJ = Q_from_cb(j, k, J)
        y = real_conj(np.kron(Dj, Dk), QJ.T) - DJ
        # print(np.linalg.norm(DJ))
        print(np.linalg.norm(y))
        D_ += real_conj(DJ, QJ)
    print('decompose j = ', j, 'k = ', k)
    print(np.linalg.norm(D - D_))


def tensor_decomposition_unit_test(l):
    for i in range(10):
        angles = np.random.rand(3)
        a = angles[0]
        b = angles[1]
        c = angles[2]
        for j in range(l+1):
            for k in range(l+1):
                tensor_decomposition_unit_test__(j, k, a, b, c)

def invariant_feature(equivariant_features, p, q, Q):
    # y = tf.einsum('bvqmrc,bvqnrc->bvqmnrc', equivariant_features[p[0]], equivariant_features[p[1]])
    # the equivariant channels must in the last dimesion

    #y = tf.einsum('bvqrcm,bvqrcn->bvqrcmn', equivariant_features[p[0]], equivariant_features[p[1]])
    """
    nb = y.get_shape()[0].value
    nv = y.get_shape()[1].value
    nq = y.get_shape()[2].value
    nr = y.get_shape()[3].value
    nc = y.get_shape()[4].value
    y = tf.reshape(y, shape=(nb, nv, nq, nr, nc, -1))
    """

def higher_product_matrix(p, q):


    Q = npClebshGordanMatrices(3)

    """
    res = np.eye((2*abs(q[0])+1)*(2*abs(p[1])+1))
    I = np.eye(1)
    res = np.real(np.reshape(Q.getMatrix(q[0], p[1], q[1]), newshape=(2*abs(q[1])+1, -1)))
    for i in range(len(p)-1):


        Qi_ = np.real(np.reshape(Q.getMatrix(q[i+1], p[i+2], q[i+2]), newshape=(2*abs(q[i+1])+1, -1)))
        Qi = np.kron(Qi_, I)
        res = np.matmul(Qi_, np.kron(res, I))
        I = np.kron(I, np.eye(2 * abs(p[i + 1]) + 1))
    """


    Q1 = np.reshape(Q.getMatrix(q[0], p[1], q[1]), newshape=(2*abs(q[1])+1, -1))
    Q2 = np.reshape(Q.getMatrix(q[1], p[2], q[2]), newshape=(2 * abs(q[2]) + 1, -1))
    M = np.real(Q1)
    M = np.kron(M, np.eye(2*abs(p[2])+1))
    M = np.matmul(np.real(Q2), M)
    print(M)
    print(np.matmul(M, M.transpose()))
    return


def higher_product(R, X, p, q, Q):

    # print(np.linalg.norm(y))
    # print(y)
    X = np.asmatrix(np.random.rand(1, 3))
    X /= (np.linalg.norm(X))
    X *= 10.0
    X_rot = (np.matmul(R.T, X.T)).T

    y = complex_sh_(abs(p[0]), X)
    y_rot = complex_sh_(abs(p[0]), X_rot)
    for i in range(len(p)-1):
        """
        print('uuu')
        print(X.shape)
        print(y.shape)
        print(p)
        print(complex_sh_(abs(p[i+1]), X).shape)
        print(Q.getMatrix(q[i], p[i+1], q[i+1]).shape)
        print('aaa')
        """
        """
        print('aaaaaa')
        print(q[i], p[i+1], q[i+1])
        print(Q.getMatrix(q[i], p[i+1], q[i+1]))
        print('bbbbbb')
        """
        X = np.asmatrix(np.random.rand(1, 3))
        X /= (np.linalg.norm(X))
        X *= 10.0
        X_rot = (np.matmul(R.T, X.T)).T


        z = np.einsum('jmn,jmn->j', Q.getMatrix(q[i], p[i+1], q[i+1]), Q.getMatrix(q[i], p[i+1], q[i+1]))
        print('qi, pi+1, qi+1 = ', q[i], p[i + 1], q[i + 1])
        print('norm z= ', np.linalg.norm(z))
        y = np.einsum('vm,vn->vmn', y, complex_sh_(abs(p[i+1]), X))
        y_rot = np.einsum('vm,vn->vmn', y_rot, complex_sh_(abs(p[i+1]), X_rot))
        y = np.einsum('jmn,vmn->vj', Q.getMatrix(q[i], p[i+1], q[i+1]), y)
        y_rot = np.einsum('jmn,vmn->vj', Q.getMatrix(q[i], p[i + 1], q[i + 1]), y_rot)
        # print(y)
        # print(np.linalg.norm(y))

    return y

def higher_tensor_decomposition_unit_test():
    Q = npClebshGordanMatrices(3)
    p = []
    q = []

    # degree 1 invariants
    p.append(np.zeros(shape=(1, 1), dtype=np.int32))
    q.append(np.zeros(shape=(1, 1), dtype=np.int32))

    p.append(np.array([[1, 1], [2, 2]], dtype=np.int32))
    q.append(np.array([[1, 0], [2, 0]], dtype=np.int32))

    p.append(np.array([[1, 1, 1],
                       [1, 1, 2],
                       [1, 2, 2],
                       [2, 2, 2]], dtype=np.int32))
    q.append(np.array([[1, 1, 0],
                       [1, 2, 0],
                       [1, 2, 0],
                       [2, 2, 0]], dtype=np.int32))
    for i in range(10):
        angles = np.random.rand(3)
        X = np.asmatrix(np.random.rand(1, 3))
        X /= (np.linalg.norm(X))
        a = angles[0]
        b = angles[1]
        c = angles[2]
        R = euler_rot_zyz(a, b, c)
        for d in range(len(p)):
            for j in range(np.size(p[d], 0)):
                print(p[d][j, :])
                print(q[d][j, :])
                z = higher_product(R, X, p[d][j, :], q[d][j, :], Q)
                print('norm output = ', np.linalg.norm(z))
                # print(np.linalg.norm(X))

def real_tensor_decomposition_unit_test__(j, k, a, b, c):

    # CRj = complex_to_real_sh(j)
    # CRk = complex_to_real_sh(k)

    # K = np.kron(CRj, CRk)
    # K_T = np.conjugate(K.T)

    Dj = real_D_wigner(j, a, b, c)
    Dk = real_D_wigner(k, a, b, c)

    D_ = np.zeros(shape=((2*j+1)*(2*k+1), (2*j+1)*(2*k+1)), dtype=np.complex64)
    D = np.kron(Dj, Dk)

    for J in range(abs(k-j), k+j+1):
        print('j = ', j, 'k = ', k, 'J = ', J)
        # CRJ = complex_to_real_sh(J)
        # RCJ = np.conjugate(CRJ.T)
        DJ = real_D_wigner(J, a, b, c)
        QJ = real_Q_from_cb(j, k, J, dtype=np.complex64)

        y = complex_conj(np.kron(Dj, Dk), np.conjugate(QJ.T)) - DJ
        # print(np.linalg.norm(DJ))
        print(np.linalg.norm(y))
        D_ += real_conj(DJ, QJ)
    print('decompose j = ', j, 'k = ', k)
    print(np.linalg.norm(D - D_))


def real_tensor_decomposition_unit_test(l):
    for i in range(3):
        angles = np.random.rand(3)
        a = angles[0]
        b = angles[1]
        c = angles[2]
        for j in range(l+1):
            for k in range(l+1):
                real_tensor_decomposition_unit_test__(j, k, a, b, c)






