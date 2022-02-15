import sys
import pennylane as qml
from pennylane import numpy as np
import pennylane.optimize as optimize

DATA_SIZE = 250


def square_loss(labels, predictions):
    """Computes the standard square loss between model predictions and true labels.

    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)

    Returns:
        - loss (float): the square loss
    """

    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def accuracy(labels, predictions):
    """Computes the accuracy of the model's predictions against the true labels.

    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)

    Returns:
        - acc (float): The accuracy.
    """

    acc = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            acc = acc + 1
    acc = acc / len(labels)

    return acc


def classify_ising_data(ising_configs, labels):
    """Learn the phases of the classical Ising model.

    Args:
        - ising_configs (np.ndarray): 250 rows of binary (0 and 1) Ising model configurations
        - labels (np.ndarray): 250 rows of labels (1 or -1)

    Returns:
        - predictions (list(int)): Your final model predictions

    Feel free to add any other functions than `cost` and `circuit` within the "# QHACK #" markers
    that you might need.
    """

    # QHACK #

    num_wires = ising_configs.shape[1]
    dev = qml.device("default.qubit", wires=num_wires)

    # Define a variational circuit below with your needed arguments and return something meaningful
    @qml.qnode(dev)
    def circuit(x, thetas):
        # state preparation
        for i in range(num_wires):
            if x[i]==1:
                qml.PauliX(i)
        # variational circuit
        for j in range(num_layers):
            for i in range(num_wires):
                qml.Rot(thetas[j, i, 0], thetas[j, i, 1], thetas[j, i, 2], wires=i)
                if i < num_wires-1:
                    qml.CNOT(wires=[i,i+1])
                else:
                    qml.CNOT(wires=[i, 0])
        return [qml.expval(qml.PauliZ(n)) for n in range(num_wires)]
        #return [qml.expval(qml.PauliZ(0))]

    def variational_classifier(x, weights, bias):
        preds = circuit(x, weights)
        return np.sum(preds) + bias

    # Define a cost function below with your needed arguments
    def cost(thetas, bias, X, Y):

        # QHACK #

        # Insert an expression for your model predictions here
        #predictions = [circuit(x, thetas) for x in ising_configs]
        predictions = [variational_classifier(x, thetas, bias) for x in X]
        # QHACK #

        return square_loss(Y, predictions)  # DO NOT MODIFY this line




    # Training
    np.random.seed(5)
    num_qubits = 4
    num_layers = 2
    weights = np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
    bias = np.array(0.0, requires_grad=True)

    #opt = qml.AdamOptimizer(stepsize=0.3)
    opt = qml.NesterovMomentumOptimizer(0.3)
    epochs = 40
    batch_size=25


    for epoch in range(epochs):

        batch_index = np.random.randint(0, len(ising_configs), (batch_size,))
        X_batch = ising_configs[batch_index]
        Y_batch = labels[batch_index]


        predictions = [np.sign(variational_classifier(x, weights, bias)) for x in ising_configs]
        current_acc = accuracy(labels, np.sign(predictions))
        #print("Step ", epoch, "--> Cost: ", cost(weights, bias, ising_configs, labels), "Accuracy: ", current_acc)

        if current_acc >=0.9:
            break

        else:
            weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, Y_batch)
            weights = np.clip(weights, -2 * np.pi, 2 * np.pi)

    predictions = [int(np.sign(variational_classifier(x, weights, bias))) for x in ising_configs]
    # QHACK #
    #print("Accuracy: ", accuracy(labels, predictions))

    return predictions


if __name__ == "__main__":
    inputs = np.array(
        sys.stdin.read().split(","), dtype=int, requires_grad=False
    ).reshape(DATA_SIZE, -1)
    ising_configs = inputs[:, :-1]
    labels = inputs[:, -1]
    predictions = classify_ising_data(ising_configs, labels)
    print(*predictions, sep=",")