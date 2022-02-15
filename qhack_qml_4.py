#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qRAM(thetas):
    """Function that generates the superposition state explained above given the thetas angles.

    Args:
        - thetas (list(float)): list of angles to apply in the rotations.

    Returns:
        - (list(complex)): final state.
    """

    # QHACK #


    # Use this space to create auxiliary functions if you need it.
    binaryLSB = ['000', '001', '010', '011', '100', '101', '110', '111']
    #binaryMSB = ['000', '100', '010', '110', '001', '101', '011', '111']
    #greyCodeLSB = ['000', '001', '011', '010', '110', '111', '101', '100']
    #greyCodeMSB = ['000', '100', '110', '010', '011', '111', '101', '001']
    listStates = binaryLSB

    def encodeState(state: str, theta):
        for i in range(len(state)):
            if state[i] == '1':
                qml.CRY(theta, wires=[3, i])
        # qml.RY(theta, wires=3)

    # QHACK #
    def encodeMatrix(thetas):
        list = np.zeros((16, 16))
        for i in range(0, 16, 2):
            list[i, i] = np.cos(thetas[int(i / 2)]/2)
            list[i, i + 1] = -np.sin(thetas[int(i / 2)]/2)
            list[i + 1, i + 1] = np.cos(thetas[int(i / 2)]/2)
            list[i + 1, i] = np.sin(thetas[int(i / 2)]/2)
        return list

    dev = qml.device("default.qubit", wires=range(4))

    @qml.qnode(dev)
    def circuit():

        # QHACK #
        # Create your circuit: the first three qubits will refer to the index, the fourth to the RY rotation.
        for i in range(3):
            qml.Hadamard(wires=i)
        qml.QubitUnitary(encodeMatrix(thetas), wires=[0,1,2,3])
        #qml.RY(thetas[0], wires=3)

        #for i in range(7):
         #   encodeState(listStates[i + 1], thetas[i + 1])

        # QHACK #

        return qml.state()

    return circuit()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    thetas = np.array(inputs, dtype=float)

    output = qRAM(thetas)
    output = [float(i.real.round(6)) for i in output]
    print(*output, sep=",")