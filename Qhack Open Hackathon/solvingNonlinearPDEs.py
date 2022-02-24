import torch
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

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
def cost(points):
    x = points[0]
    xBdr = points[1]
    BdrValue = 0
    residualLoss = residual(x)**2
    boundaryLoss = (BdrValue - compute_output(model(xBdr)))**2
    return residualLoss + boundaryLoss

controlPoints = torch.rand(200, requires_grad=True)
controlPointsBdr = torch.tensor([-1,1]*100)
batches = torch.utils.data.DataLoader(torch.stack((controlPoints, controlPointsBdr),dim=1), batch_size=20, shuffle=True)

learning_rate = 0.1
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

steps = 20
model.train()

for i in range(steps):
    for batch in batches:
        #print(batch)
        loss = 0
        opt.zero_grad()
        for x in batch:
            loss += cost(x)
        loss.backward(retain_graph=True)
        opt.step()

    print("step: ", i, "--> loss = ", loss)
# print("Weights: ", model.state_dict())
# exact solution of the differential equation
def exactSolution(x):
    return 1/2*(x**2-1)


# Create Plot
model.eval()
numPlotPoints = 101

plotPoints = np.linspace(-1, 1, numPlotPoints).reshape(numPlotPoints, 1)
yExact = exactSolution(plotPoints)
yModel = [compute_output(model(Point[0])).detach().numpy() for Point in torch.tensor(plotPoints)]

plt.plot(plotPoints, yExact, label = "Exact Solution")
plt.plot(plotPoints, yModel, label = "Model Solution")
plt.xlabel("x", fontsize=14)
plt.ylabel("u(x)", fontsize=14)
plt.legend()
plt.savefig("ModelFit.png")