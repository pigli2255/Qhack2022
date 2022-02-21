#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def deutsch_jozsa(fs):
    """Function that determines whether four given functions are all of the same type or not.

    Args:
        - fs (list(function)): A list of 4 quantum functions. Each of them will accept a 'wires' parameter.
        The first two wires refer to the input and the third to the output of the function.

    Returns:
        - (str) : "4 same" or "2 and 2"
    """

    # QHACK #
    dev = qml.device("default.qubit", wires=6, shots=1)

    @qml.qnode(dev)
    def circuit():
        """Implements the Deutsch Jozsa algorithm."""

        # QHACK #

        # Insert any pre-oracle processing here

        qml.Hadamard(0)
        qml.Hadamard(1)

        qml.Hadamard(3)
        qml.Hadamard(4)

        for i in [2,5]:
            qml.PauliX(i)
            qml.Hadamard(i)

        f4([0,1,2])
        f3([0,1,2])

        f2([3,4,5])
        f3([3,4,5])


        qml.Hadamard(0)
        qml.Hadamard(1)
        qml.Hadamard(3)
        qml.Hadamard(4)

        # QHACK #

        return [qml.expval(qml.PauliZ(0)),qml.expval(qml.PauliZ(3))]

    measurements = circuit()

    # QHACK #

    # From `sample` (a single call to the circuit), determine whether the function is constant or balanced.
    if sum(measurements) == 2:
        return "4 same"
    elif sum(measurements) == 0:
        return "2 and 2"

    else:
        print("Error!")
        return -1
    # QHACK #



if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]


    # Definition of the four oracles we will work with.

    def f1(wires):
        qml.CNOT(wires=[wires[numbers[0]], wires[2]])
        qml.CNOT(wires=[wires[numbers[1]], wires[2]])


    def f2(wires):
        qml.CNOT(wires=[wires[numbers[2]], wires[2]])
        qml.CNOT(wires=[wires[numbers[3]], wires[2]])


    def f3(wires):
        qml.CNOT(wires=[wires[numbers[4]], wires[2]])
        qml.CNOT(wires=[wires[numbers[5]], wires[2]])
        qml.PauliX(wires=wires[2])


    def f4(wires):
        qml.CNOT(wires=[wires[numbers[6]], wires[2]])
        qml.CNOT(wires=[wires[numbers[7]], wires[2]])
        qml.PauliX(wires=wires[2])


    output = deutsch_jozsa([f1, f2, f3, f4])
    print(f"{output}")