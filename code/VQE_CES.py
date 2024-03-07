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
    eigen_vals = SES(Hamiltonians)
    
    return ls, eigen_vals
    
def SES(Hamiltonians):
    """
    A generic standard eigenvalue solver.
    """
    
    # finds the eigenvalues
    eigen_vals, eigen_vecs = np.linalg.eig(Hamiltonians)
    
    # sorts the eigenvalues
    eigen_vals_sorted = np.zeros_like(eigen_vals)
    eigen_vals_sorted[:,0] = np.min(eigen_vals, axis=1)
    eigen_vals_sorted[:,1] = np.max(eigen_vals, axis=1)
    
    return eigen_vals_sorted
    

def plot_one_qubit(ls, eigen_vals):
    """
    Plots the results for the one qubit case.
    """
    
    plt.plot(ls, eigen_vals[:,0], label="E0")
    plt.plot(ls, eigen_vals[:,1], label="E1")
    plt.legend()
    plt.show()
    
    
def main():
    
    data = one_qubit_data()
    ls, eigen_vals = one_qubit_SES(*data)
    plot_one_qubit(ls, eigen_vals)
    

if __name__ == "__main__":
    main()

