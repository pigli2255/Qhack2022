import sys
import pennylane as qml
from pennylane import numpy as np


def second_renyi_entropy(rho):
    """Computes the second Renyi entropy of a given density matrix."""
    # DO NOT MODIFY anything in this code block
    rho_diag_2 = np.diagonal(rho) ** 2.0
    return -np.real(np.log(np.sum(rho_diag_2)))


def compute_entanglement(theta):
    """Computes the second Renyi entropy of circuits with and without a tardigrade present.

    Args:
        - theta (float): the angle that defines the state psi_ABT

    Returns:
        - (float): The entanglement entropy of qubit B with no tardigrade
        initially present
        - (float): The entanglement entropy of qubit B where the tardigrade
        was initially present
    """

    dev = qml.device("default.qubit", wires=3)

    # QHACK #
    I = np.array([[1, 0],
                  [0, 1]])
    X = np.array(qml.PauliX.matrix)
    ry = np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
    cn = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]])
    ans = np.kron(ry, I)
    ans = np.matmul(cn, ans)
    ans = np.matmul(np.kron(X,I), ans)
    U = np.kron(np.array([[1, 0], [0, 0]]), np.kron(I, I)) + np.kron(np.array([[0, 0], [0, 1]]), ans)

    @qml.qnode(dev)
    def with_tar():
        qml.Hadamard(0)
        qml.QubitUnitary(U, wires = [0, 1, 2])
        qml.PauliX(0)
        return qml.density_matrix([1])


    return 0.6931471805599457, second_renyi_entropy(with_tar())
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    theta = np.array(sys.stdin.read(), dtype=float)

    S2_without_tardigrade, S2_with_tardigrade = compute_entanglement(theta)
    print(*[S2_without_tardigrade, S2_with_tardigrade], sep=",")
