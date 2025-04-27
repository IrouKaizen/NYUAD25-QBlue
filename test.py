import pennylane as qml
from pennylane.qnn import TorchLayer
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── Hyperparameters ───────────────────────────────────────────────────────────
n_inputs     = 3          # [flow_rate, temperature, concentration]
n_hidden     = 2          # size of the quantum "hidden state"
n_qubits     = n_inputs + n_hidden
n_layers_rec = 2          # entangling layers in the QRNN cell
n_layers_out = 2          # entangling layers in the QNN readout
batch_size   = 16
lr           = 0.01
epochs       = 100        # adjust as you like

# ── Devices ───────────────────────────────────────────────────────────────────
dev_rec = qml.device("default.qubit", wires=n_qubits)
dev_out = qml.device("default.qubit", wires=n_hidden)

# ── 1) QRNN cell QNode ────────────────────────────────────────────────────────
@qml.qnode(dev_rec, interface="torch")
def qrnn_cell(inputs, weights):
    # inputs: 1D tensor of length n_inputs + n_hidden
    x = inputs[:n_inputs]
    h = inputs[n_inputs:]
    # embed
    for i in range(n_inputs):
        qml.RY(x[i], wires=i)
    for j in range(n_hidden):
        qml.RY(h[j], wires=n_inputs + j)
    # entangle + measure
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=n_inputs + j)) for j in range(n_hidden)]

qrnn_layer = TorchLayer(qrnn_cell, {"weights": (n_layers_rec, n_qubits, 3)})
qrnn_layer.batch_execute = False

# ── 2) QNN read-out QNode ──────────────────────────────────────────────────────
@qml.qnode(dev_out, interface="torch")
def qnn_readout(inputs, weights):
    # inputs: 1D tensor of length n_hidden
    for j in range(n_hidden):
        qml.RY(inputs[j], wires=j)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_hidden))
    return qml.expval(qml.PauliZ(wires=0))

qnn_out_layer = TorchLayer(qnn_readout, {"weights": (n_layers_out, n_hidden, 3)})
qnn_out_layer.batch_execute = False

# ── 3) Fully-Quantum RNN Model ─────────────────────────────────────────────────
class FullyQuantumRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.qrnn = qrnn_layer
        self.qout = qnn_out_layer

    def forward(self, x_seq):
        # x_seq: (time_steps, batch_size, n_inputs)
        batch = x_seq.shape[1]
        h = torch.zeros(batch, n_hidden)
        time_steps = x_seq.shape[0]

        # recurrent sweep
        for t in range(time_steps):
            xt = x_seq[t]                # (batch, n_inputs)
            new_h = torch.zeros(batch, n_hidden)
            for b in range(batch):
                vec = torch.cat([xt[b], h[b]], dim=0)  # (n_inputs + n_hidden,)
                new_h[b] = self.qrnn(vec)
            h = new_h

        # final quantum read-out
        out = torch.zeros(batch, 1)
        for b in range(batch):
            out[b] = self.qout(h[b])    # (1,)
        return out

# ── 4) Training Loop ──────────────────────────────────────────────────────────
model     = FullyQuantumRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn   = nn.MSELoss()

# Dummy training data (seq_len = 10)
training_seq_len = 10
X_train = torch.randn(training_seq_len, batch_size, n_inputs)
Y_train = torch.randn(batch_size, 1)

losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    preds = model(X_train)            # → (batch_size, 1)
    loss  = loss_fn(preds, Y_train)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}  Loss = {loss.item():.4f}")

# ── 5) Sweep Flow Rates ───────────────────────────────────────────────────────
temp_val = 25.0    # °C
conc_val = 35.0    # PSU
flows = torch.linspace(0.0, 10.0, 100).unsqueeze(1)  # (100,1)
temps = torch.full((100,1), temp_val)
concs = torch.full((100,1), conc_val)

# Create a 1-step sequence for the sweep
x_seq = torch.stack([torch.cat([flows, temps, concs], dim=1)], dim=0)

with torch.no_grad():
    eff_preds = model(x_seq).squeeze()  # (100,)

# ── 6) Plot Results ───────────────────────────────────────────────────────────
plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Epochs")

plt.figure()
plt.plot(flows.squeeze().numpy(), eff_preds.numpy())
plt.xlabel("Flow Rate (m³/s)")
plt.ylabel("Predicted Efficiency")
plt.title("Predicted Efficiency vs. Flow Rate")
plt.show()
