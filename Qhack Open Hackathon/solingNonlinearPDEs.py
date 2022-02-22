import torch
import pennylane as qml
from pennylane import numpy as np
# import matplotlib.pyplot as plt

n_qubits = 6
depth = 1
layers = 1

dev = qml.device("default.qubit", wires = n_qubits)

# Define Input Feature Map
def TChebyshev(x):
    #print(2*2*torch.arccos(x))
    for layer in range(layers):
        for i in range(n_qubits):
            qml.RY(2*torch.arccos(x), wires=i)

 # Define Variational quantum circuit
def HardwareEffAnsatz(params):
    for _ in range(depth):
        for l in range(n_qubits):
            #qml.RZ(params[l, 0], wires=l)
            qml.RX(params[l, 0], wires=l)
            #qml.RZ(params[l, 2], wires=l)
        for l in range(0, n_qubits - 1, 2):
            qml.CNOT(wires=[l, l + 1])
        for l in range(1, n_qubits - 1, 2):
            qml.CNOT(wires=[l, l + 1])

 # Combine QCircuits in qnode
@qml.qnode(dev, interface="torch")
def VQA_TC(inputs, params):
    TChebyshev(inputs)
    HardwareEffAnsatz(params)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]  # Use total magnetization as cost function

# Define torch model
weight_shapes = { 'params': (n_qubits, 1)}
VQA_layer = qml.qnn.TorchLayer(VQA_TC, weight_shapes)
model = torch.nn.Sequential(VQA_layer)

def compute_output(outs):
    return torch.mean(outs)

# Compute Residual (How much does solution differ from differential equation)
def residual(controlPoints):
    """ Calculate the residium of the control points"""
    x = controlPoints
    u = compute_output(model(x))
    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
        allow_unused=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True,
        allow_unused=True
    )[0]
    f = u_xx - 1

    return f

# cost function
def cost(pred):   # Later we can add additional boundary loss here
    return pred**2

controlPoints = torch.rand(100, requires_grad=True)
batches = torch.utils.data.DataLoader(controlPoints, batch_size=10, shuffle=True)

opt = torch.optim.Adam(model.parameters(), lr=1)

steps = 3
model.train()

for i in range(steps):
    for batch in batches:
        #print(batch)
        loss = 0
        for x in batch:
            opt.zero_grad()
            pred = residual(x)
            loss += cost(pred)
        loss.backward(retain_graph=True)
        opt.step()

    print("step: ", i, "--> loss = ", loss)
    print("Weights: ", model.state_dict())

