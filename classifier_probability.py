import jax
import matplotlib.pyplot as plt
import pennylane as qml
import pennylane.numpy as np
import pandas as pd
import jax.numpy as jnp
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utilities import map_cost_prob, predict_all_prob
import optax

seed = 42
np.random.seed(seed)

# Parameters
layers = 1  # Layers of ansatz
num_qubits = 1  # Number of qubits in ansatz
modify_data = True
epochs = 501
lr = 0.01
eps = 0.0001

training_data = pd.read_csv("mock_train_set.csv")
testing_data = pd.read_csv("mock_test_set.csv")

y_train = training_data["4"].to_numpy(dtype=int)
X_train = training_data[["0", "1", "2", "3"]].to_numpy()

y_test = testing_data["4"].to_numpy(dtype=int)
X_test = testing_data[["0", "1", "2", "3"]].to_numpy()

# Modified Dataset
if modify_data:
    X_train = jnp.log(X_train.prod(axis=1)).reshape(-1, 1)
    X_test = jnp.log(X_test.prod(axis=1)).reshape(-1, 1)

scaler1 = StandardScaler()
scaler2 = MinMaxScaler([-1, 1])
X_train = scaler1.fit_transform(X_train)
X_test = scaler1.transform(X_test)
X_train = scaler1.fit_transform(X_train)
X_test = scaler1.transform(X_test)

dev = qml.device("default.qubit.jax", wires=num_qubits, shots=None)


@jax.jit
@qml.qnode(dev, interface="jax")
def circuit_data_reupload_without_entanglement(params, features):
    """
    Create ansatz using data-re-upload method. Does not use entanglement.
    :param params: Variational Parameters
    :param features: Feature Vector
    :return: Probability values for qubit 0
    """
    thetas = params[:num_thetas]
    weights = params[num_thetas:]
    thetas = thetas.reshape([layers, num_qubits, 6])
    weights = weights.reshape([layers, num_qubits, 4])
    for i in range(layers):
        for j in range(num_qubits):
            qml.Rot(*(thetas[i][j][:3] + weights[i][j][:3] * features[:3]), wires=j)
            qml.Rot(
                thetas[i][j][-1] + weights[i][j][-1] * features[-1],
                thetas[i][j][-2],
                thetas[i][j][-3],
                wires=j,
            )
    return qml.probs(0)


@jax.jit
@qml.qnode(dev, interface="jax")
def circuit_data_reupload_with_entanglement(params, features):
    """
    Create ansatz using data-re-upload method. Uses entanglement.
    :param params: Variational Parameters
    :param features: Feature Vector
    :return: Probability values for qubit 0
    """
    thetas = params[:num_thetas]
    weights = params[num_thetas:]
    thetas = thetas.reshape([layers, num_qubits, 6])
    weights = weights.reshape([layers, num_qubits, 4])
    for i in range(layers):
        for j in range(num_qubits):
            qml.Rot(*(thetas[i][j][:3] + weights[i][j][:3] * features[:3]), wires=j)
            qml.Rot(
                thetas[i][j][-1] + weights[i][j][-1] * features[-1],
                thetas[i][j][-2],
                thetas[i][j][-3],
                wires=j,
            )
        if i != layers - 1 and num_qubits > 1:
            if i % 2 == 0:
                wire = 0
                for _ in range(num_qubits // 2):
                    qml.CZ(wires=[wire, wire + 1])
                    wire = wire + 2
            else:
                wire = 1
                for _ in range(num_qubits // 2 - 1):
                    qml.CZ(wires=[wire, wire + 1])
                    wire = wire + 2
                qml.CZ(wires=[dev.wires[0], dev.wires[-1]])
    return qml.probs(0)


# Choose circuit here
classifier_circuit = circuit_data_reupload_without_entanglement

num_thetas = layers * 6 * num_qubits
num_weights = layers * 4 * num_qubits
num_params = num_thetas + num_weights
variational_parameters = np.random.random(num_params)
variational_parameters = jnp.array(variational_parameters)


def fit(params, optimizer, map_cost, X_train, y_train, X_test, y_test, circuit):
    """
    Train the model
    :param params: Variational Parameters
    :param optimizer: Optimizer to use
    :param map_cost: Cost Function
    :param X_train: Training features
    :param y_train: Training labels
    :param X_test: Testing features
    :param y_test: Testing labels
    :param circuit: Circuit to use for classification
    :return: Trained parameters, Cost function values for training data, Cost function values for testing data
    """
    opt_state = optimizer.init(params)

    train_costs = []
    test_costs = []

    @jax.jit
    def step(params, opt_state, X, Y):
        loss_value, grads = jax.value_and_grad(map_cost)(params, X, Y, circuit, eps)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i in range(epochs):
        params, opt_state, loss_value = step(params, opt_state, X_train, y_train)
        train_costs.append(map_cost(params, X_train, y_train, circuit, eps))
        test_costs.append(map_cost(params, X_test, y_test, circuit, eps))

    return params, train_costs, test_costs


optimizer = optax.adam(lr)
variational_parameters, train_costs, test_costs = fit(
    variational_parameters,
    optimizer,
    map_cost_prob,
    X_train,
    y_train,
    X_test,
    y_test,
    classifier_circuit,
)

plt.plot(train_costs)
plt.plot(test_costs)
plt.legend(["Train Data", "Test Data"])
plt.title("Fidelity Cost")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.xticks(np.arange(0, epochs, epochs // 10), np.arange(0, epochs, epochs // 10) + 1)
plt.tight_layout()
# plt.savefig(f'plots/probability_{num_qubits}_{layers}_{modify_data}_cost_plot')
plt.show()

y_predict_train = predict_all_prob(X_train, variational_parameters, classifier_circuit)
y_predict_test = predict_all_prob(X_test, variational_parameters, classifier_circuit)

print("Accuracy after Training:")
print(
    "Training Data:",
    np.round(accuracy_score(y_pred=y_predict_train, y_true=y_train), 4) * 100,
)
print(
    "Testing Data:",
    np.round(accuracy_score(y_pred=y_predict_test, y_true=y_test), 4) * 100,
)
