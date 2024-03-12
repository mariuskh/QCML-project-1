# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:35:03 2024

@author: bendikst
"""

# from qiskit_ibm_provider import IBMProvider
# from qiskit import QuantumCircuit, transpile, assemble
# # Provide your IBM Quantum Experience API token here
# api_token = 'b75eb4e42166959dae7bce7f0385ac8affe33c7a1e799b2516346399959644b8d12ddfb19d8f13decea6e7ea5798a57d2c2a29af520e7de5491550aE17f671b5'
# IBMProvider.save_account(api_token, overwrite=True)
# provider = IBMProvider()
# # Run on the 127-qubit machine IBM-Osaka
# backend = provider.get_backend('ibm_osaka')

import numpy as np
import qiskit as qk
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.visualization import plot_histogram, plot_bloch_vector, visualize_transition
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
import seaborn as sns; sns.set_theme()
from tqdm import tqdm

from qc import One_qubit, Two_qubit, Four_qubit


sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

    
def one_qubit_data(E0 = 0, E1 = 4, V11 = 3, V22 = -3, V12 = 0.2, l_min = 0, l_max = 1):
    """
    Creates a list pf all the data for the one qubit case. Used in task b and c.
    """
    
    data = [E0, E1, V11, V22, V12, l_min, l_max]
    return data


def one_qubit_SES(E0 = 0, E1 = 4, V11 = 3, V22 = -3, V12 = 0.2, l_min = 0, l_max = 1, n = 1000):
    """
    The standard eigenvalue solver for the one qubit case.
    """
    
    # variabels and matrices which we use to create the Hamiltonians
    I   = np.eye(2)
    Eps = (E0+E1)/2
    Ome = (E0-E1)/2
    c   = (V11+V22)/2
    omz = (V11-V22)/2
    omx = V12
    
    # an array of all the λ's we are testing
    ls = np.linspace(l_min, l_max, n)
    
    # creates an [n,2,2] numpy array that represents the Hamiltonians
    H0 = Eps*I + Ome*sigma_z # non-interacting
    H1 = c*I + omz*sigma_z + omx*sigma_x # interacting
    Hamiltonians = H0 + ls[:,None,None]*H1[None]
    
    # calls the standard eigenvalue solver for the array of Hamiltonians
    eigen_vals,_ = SES(Hamiltonians)
    
    return ls, eigen_vals
    
def SES(Hamiltonians):
    """
    A generic standard eigenvalue solver.
    """
    
    # finds the eigenvalues
    eigen_vals, eigen_vecs = np.linalg.eig(Hamiltonians)
    
    # sorts the eigenvalues
    perm = eigen_vals.argsort(axis=1)
    eigen_vals.sort(axis=1)
    
    eigen_vecs = np.take_along_axis(eigen_vecs.transpose(0, 2, 1), perm[:,:,None], axis=1).transpose(0, 2, 1) # eigen_vecs[perm]
    
    return eigen_vals, eigen_vecs
    

# def plot_one_qubit(ls, eigen_vals):
#     """
#     Plots the results for the one qubit case.
#     """
    
#     plt.plot(ls, eigen_vals[:,0], label="E0")
#     plt.plot(ls, eigen_vals[:,1], label="E1")
#     plt.legend()
#     plt.xlabel(r"$\lambda$")
#     plt.ylabel("Energy")
#     plt.show()
    
    
    
def two_qubit_SES(l_min = 0, l_max = 1, n = 1000):
    
    # an array of all the λ's we are testing
    ls = np.linspace(l_min, l_max, n)
    
    Hx = 2.0
    Hz = 3.0
    # H_0
    Energiesnoninteracting = [0.0, 2.5, 6.5, 7.0]
    Hzs = Hz*np.array([1,-1,-1,1])
    
    Hamiltonians = np.zeros((n,4,4))
    for i in range(n):
        np.fill_diagonal(Hamiltonians[i], [Energiesnoninteracting[n]+Hzs[n]*ls[i] for n in range(len(Energiesnoninteracting))])
        np.fill_diagonal(np.fliplr(Hamiltonians[i]), Hx*ls[i])
    
    # print(Hamiltonians)
    
    eigen_vals, eigen_vecs = SES(Hamiltonians)
        
    plot_eigen_vals(ls, eigen_vals)    
    
    
    # ev = eigen_vecs[-1]
    
    entropies = np.zeros((n,4))
    for i in range(n):
        entropies[i] = find_entropy(eigen_vecs[i])
    
    print(entropies)
    print("\n")
    
    plot_eigen_vals(ls, entropies)
    
    
def find_entropy(ev):
    SAs = np.zeros(4)
    
    for index in range(4):
        rhoA = np.linalg.eigvalsh(trace_out(ev[:,index], 0))
        rhoA = np.ma.masked_equal(rhoA, 0).compressed()
        SA = - np.sum(rhoA*np.log2(rhoA))
        SAs[-1-index] = SA
    
    return SAs


def lipkin_model(eps=1, W=0, V=1, n=1000):
    
    H_J1 = np.zeros((n,3,3))
    
    Vs = np.linspace(0, 2, n)
    
    H_J1[:,0,0]   = -eps
    H_J1[:,-1,-1] =  eps
    H_J1[:,-1,0]  = -Vs
    H_J1[:,0,-1]  = -Vs
    
    eigen_vals_J1, eigen_vecs_J1 = SES(H_J1)
    
    plot_eigen_vals(Vs, eigen_vals_J1, xlabel="V/ε", ylabel="E/ε")
    
    H_J2 = np.zeros((n,5,5))
    
    W  = 0
    
    H_J2[:,0,0]   = -2*eps
    H_J2[:,-1,-1] =  2*eps
    H_J2[:,1,1]   = -eps+3*W
    H_J2[:,2,2]   =  4*W
    H_J2[:,3,3]   =  eps+3*W
    
    H_J2[:,0,2]  = -np.sqrt(6)*Vs
    H_J2[:,2,0]  = -np.sqrt(6)*Vs
    H_J2[:,4,2]  = -np.sqrt(6)*Vs
    H_J2[:,2,4]  = -np.sqrt(6)*Vs
    H_J2[:,1,3]  = -3*Vs
    H_J2[:,3,1]  = -3*Vs
    
    eigen_vals_J2, eigen_vecs_J2 = SES(H_J2)
    
    plot_eigen_vals(Vs, eigen_vals_J2, xlabel="V/ε", ylabel="E/ε")
    
    print(H_J2[-1])
    print(H_J2[0])
    
    
    def Hamiltonian(v):    
        qb = One_qubit()
        ZIII = np.kron(qb.Z, np.kron(qb.I, np.kron(qb.I, qb.I)))
        IZII = np.kron(qb.I, np.kron(qb.Z, np.kron(qb.I, qb.I)))
        IIZI = np.kron(qb.I, np.kron(qb.I, np.kron(qb.Z, qb.I)))
        IIIZ = np.kron(qb.I, np.kron(qb.I, np.kron(qb.I, qb.Z)))
    
        XXII = np.kron(qb.X, np.kron(qb.X, np.kron(qb.I, qb.I)))
        XIXI = np.kron(qb.X, np.kron(qb.I, np.kron(qb.X, qb.I)))
        XIIX = np.kron(qb.X, np.kron(qb.I, np.kron(qb.I, qb.X)))
        IXXI = np.kron(qb.I, np.kron(qb.X, np.kron(qb.X, qb.I)))
        IXIX = np.kron(qb.I, np.kron(qb.X, np.kron(qb.I, qb.X)))
        IIXX = np.kron(qb.I, np.kron(qb.I, np.kron(qb.X, qb.X)))
        
        YYII = np.kron(qb.Y, np.kron(qb.Y, np.kron(qb.I, qb.I)))
        YIYI = np.kron(qb.Y, np.kron(qb.I, np.kron(qb.Y, qb.I)))
        YIIY = np.kron(qb.Y, np.kron(qb.I, np.kron(qb.I, qb.Y)))
        IYYI = np.kron(qb.I, np.kron(qb.Y, np.kron(qb.Y, qb.I)))
        IYIY = np.kron(qb.I, np.kron(qb.Y, np.kron(qb.I, qb.Y)))
        IIYY = np.kron(qb.I, np.kron(qb.I, np.kron(qb.Y, qb.Y)))
    
        H = 0.5*(ZIII + IZII + IIZI + IIIZ) - \
            v/2 * (XXII + XIXI + XIIX + IXXI + IXIX + IIXX) + \
            v/2 * (YYII + YIYI + YIIY + IYYI + IYIY + IIYY)
        return H
    
    v_vals_ana = np.arange(0, 2, 0.01)
    eigvals_ana = np.zeros((len(v_vals_ana), 16))
    entropy = np.zeros((len(v_vals_ana), 16))
    for index, v in enumerate(v_vals_ana):
        H = Hamiltonian(v)
        if v == 1.99:
            print()
        eigen, eigvecs = np.linalg.eig(H)
        permute = eigen.argsort()
        eigvals_ana[index] = eigen[permute]
        eigvecs = eigvecs[:,permute]
    
    
    
def plot_eigen_vals(ls, eigen_vals, xlabel=r"$\lambda$", ylabel="Energy"):
    """
    Plots the eigen values for the n-qubit case.
    """
    
    for i in range(len(eigen_vals[0])):
        plt.plot(ls, eigen_vals[:,i], label=f"E{i}")
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
    
def main():
    
    # data = one_qubit_data()
    # ls, eigen_vals = one_qubit_SES(*data)
    # plot_eigen_vals(ls, eigen_vals)
    
    # two_qubit_SES()
    
    lipkin_model()



def Hamiltonian(lmb):    
    Hx = 2.0 
    Hz = 3.0
    H0Energiesnoninteracting = [0.0, 2.5, 6.5, 7.0]
    
    HI = Hz*np.kron(sigma_z, sigma_z) + Hx*np.kron(sigma_x, sigma_x)
    H0 = np.diag(H0Energiesnoninteracting)
    H = H0 + lmb*HI
    return H


def trace_out(state, index):
    # can we just take this? 
    ket0 = np.array([1, 0])
    ket1 = np.array([0, 1])
    I = np.eye(2)

    density = np.outer(state, np.conj(state))
    if index == 0:
        op0 = np.kron(ket0, I)
        op1 = np.kron(ket1, I)
    elif index == 1:
        op0 = np.kron(I, ket0)
        op1 = np.kron(I, ket1)
    return op0.conj() @ density @ op0.T + op1.conj() @ density @ op1.T # need to take conj() on first and .T on second since np.arrays are 


if __name__ == "__main__":
    main()  
    
    
    # lmbvalues_ana = [1] # np.arange(0.99, 1, 0.01)
    # eigvals_ana = np.zeros((len(lmbvalues_ana), 4))
    # entropy = np.zeros((len(lmbvalues_ana), 4))
    # for index, lmb in enumerate(lmbvalues_ana):
    #     H = Hamiltonian(lmb)
    #     # if lmb == 1:
    #     #     print(H)
    #     eigen, eigvecs = np.linalg.eig(H)
    #     permute = eigen.argsort()
    #     eigvals_ana[index] = eigen[permute]
    #     eigvecs = eigvecs[:,permute]
    #     # print(eigvals_ana)
    #     # print(eigvecs)
    #     for i in range(4):
    #         sub_density = trace_out(eigvecs[:, i], 0) # trace out qubit 0 from the ground state
    #         lmb_density = np.linalg.eigvalsh(sub_density)
    #         lmb_density = np.ma.masked_equal(lmb_density, 0).compressed() # remove zeros to avoid log(0)
    #         print(lmb_density)
    #         entropy[index, i] = -np.sum(lmb_density*np.log2(lmb_density))
            
    # print(entropy)
