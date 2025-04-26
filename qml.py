# ================================================================
# Step 0: Imports
# ================================================================
import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ================================================================
# Step 1: Generate Quantum Sensor Data
# ================================================================

# Quantum device
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# Number of timesteps
timesteps = 300

# Noise parameters
sensor_noise_std = 0.02
flow_rate_noise_std = 0.05
efficiency_noise_std = 0.01

# Quantum sensor signal generator
def quantum_sensor_state(t):
    angles = [0.1*t + 0.05*np.sin(t/10), 0.2*t + 0.03*np.cos(t/15)]
    return angles

# Create flow rate series (m³/s)
flow_rate_series = 5.0 + 0.5*np.sin(np.linspace(0, 10*np.pi, timesteps))
flow_rate_series += np.random.normal(0, flow_rate_noise_std, size=timesteps)

# Quantum sensor readings
@qml.qnode(dev)
def sensor_measurement(angles):
    qml.RX(angles[0], wires=0)
    qml.RY(angles[1], wires=1)
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

sensor_data = []
for t in range(timesteps):
    angles = quantum_sensor_state(t)
    readings = sensor_measurement(angles)
    noisy_readings = readings + np.random.normal(0, sensor_noise_std, size=2)
    sensor_data.append(noisy_readings)

sensor_data = np.array(sensor_data)

# Plant efficiency function (%)
optimal_flow = 5.0

def efficiency_function(sensor_outputs, flow_rate):
    base_efficiency = 50.0  # %
    contribution = 30.0*(sensor_outputs[0] + sensor_outputs[1])
    penalty = 10.0*(flow_rate - optimal_flow)**2
    efficiency = base_efficiency + contribution - penalty
    return max(0.0, efficiency)

efficiency_series = np.array([
    efficiency_function(sensor_data[t], flow_rate_series[t])
    for t in range(timesteps)
])

efficiency_series += np.random.normal(0, efficiency_noise_std, size=timesteps)
efficiency_series = np.clip(efficiency_series, 0.0, None)

# ================================================================
# Step 2: Prepare Dataset
# ================================================================

# Features: [Sensor1, Sensor2, FlowRate]
X = np.hstack([sensor_data, flow_rate_series.reshape(-1,1)])
y = efficiency_series

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to Torch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ================================================================
# Step 3: Define Quantum-Classical Hybrid Model
# ================================================================

n_qubits_model = 4

# Quantum circuit
dev2 = qml.device("default.qubit", wires=n_qubits_model)

@qml.qnode(dev2, interface="torch")
def quantum_circuit(inputs, weights):
    # Encode inputs
    for i in range(3):
        qml.RX(inputs[i], wires=i)
    
    # Trainable layers
    qml.templates.AngleEmbedding(weights[0], wires=[0,1,2,3])
    qml.templates.BasicEntanglerLayers(weights[1], wires=[0,1,2,3])
    
    return qml.expval(qml.PauliZ(0))

# Hybrid Model
class HybridQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_net = nn.Linear(3, 3)  # classical preprocessing
        weight_shapes = {"weights": (2, 4)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.post_net = nn.Linear(1, 1)  # map quantum output to efficiency

    def forward(self, x):
        x = self.pre_net(x)
        x = self.qlayer(x)
        x = self.post_net(x)
        return x

model = HybridQNN()

# ================================================================
# Step 4: Training Loop
# ================================================================

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

epochs = 50

train_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_torch)
    loss = loss_fn(outputs, y_train_torch)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ================================================================
# Step 5: Prediction
# ================================================================

model.eval()
y_pred_torch = model(X_test_torch)
y_pred = y_pred_torch.detach().numpy().flatten()
y_true = y_test_torch.numpy().flatten()

# Calculate MSE
mse = mean_squared_error(y_true, y_pred)
print(f"Test MSE: {mse:.4f}")

# ================================================================
# Step 6: Plot True vs Predicted
# ================================================================

plt.figure(figsize=(12,6))
plt.plot(y_true, label="True Efficiency", color="purple")
plt.plot(y_pred, label="Predicted Efficiency", color="gold", linestyle='--')
plt.xlabel("Sample (Test Set)")
plt.ylabel("Efficiency (%)")
plt.title("True vs Predicted Plant Efficiency")
plt.legend()
plt.tight_layout()
plt.savefig("efficiency_prediction_vs_true.png", dpi=300)
print("Prediction figure saved as 'efficiency_prediction_vs_true.png")

