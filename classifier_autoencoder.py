import jax
import matplotlib.pyplot as plt
import pennylane as qml
import pennylane.numpy as np
import pandas as pd
import jax.numpy as jnp
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utilities import U
import optax

seed = 42
np.random.seed(seed)

# Parameters
# encoding = 1 for AmplitudeEncoding
# encoding = 2 for AngleEncoding
encoding = 1
layers = 1  # Number of layers in encoder
epochs = 501
lr = 0.01

if encoding == 1:
    num_trash_bits = 1
    num_data_bits = 1
else:
    num_trash_bits = 3
    num_data_bits = 1

training_data = pd.read_csv("mock_train_set.csv")
testing_data = pd.read_csv("mock_test_set.csv")

y_train = training_data["4"].to_numpy(dtype=int)
X_train = training_data[["0", "1", "2", "3"]].to_numpy()

y_test = testing_data["4"].to_numpy(dtype=int)
X_test = testing_data[["0", "1", "2", "3"]].to_numpy()

scaler1 = StandardScaler()
X_train = scaler1.fit_transform(X_train)
X_test = scaler1.transform(X_test)
scaler2 = MinMaxScaler([-1, 1])
X_train = scaler2.fit_transform(X_train)
X_test = scaler2.transform(X_test)

class_0_train = X_train[y_train == 0]
class_1_train = X_train[y_train == 1]

num_wires = num_trash_bits + num_data_bits
trash_bits_encoding = list(range(num_trash_bits))
data_bits_encoding = list(range(num_trash_bits, num_wires))

num_weights = layers * (2 * 3 * (num_wires) + 3 * ((num_wires) - 1) * (num_wires))

weights_encoder = np.random.random(num_weights)
weights_encoder = np.random.uniform(-np.pi, np.pi, num_weights, requires_grad=True)

weights_encoder = jnp.array(weights_encoder)

dev = qml.device("default.qubit.jax", wires=num_wires, shots=None)
zero_state = jnp.zeros([2**num_trash_bits])
zero_state = zero_state.at[0].set(1)


@jax.jit
@qml.qnode(dev, interface="jax")
def circuit_encoding(params, state, encoding=encoding):
    """
    This creates the circuit for the autoencoder classifier.
    :param params: Variational Parameters
    :param state: Feature Vector
    :param encoding: 1 for Amplitude, 2 for Angle
    :return: Density Matrix of trash qubit state
    """
    params = params.reshape([layers, -1])

    # Crete Feature Encoding
    if encoding == 1:
        qml.AmplitudeEmbedding(state, wires=range(num_wires), normalize=True)
    else:
        qml.AngleEmbedding(state, wires=range(num_wires))

    # Create the encoder
    U(params, trash_bits_encoding, data_bits_encoding, layers)

    return qml.density_matrix(wires=range(num_trash_bits))


def encoding_fidelity(params, state):
    """
    This function measures the fidelity of the trash state with zero state
    :param params: Variational Parameters
    :param state: Feature Vector
    :return: Fidelity
    """
    trash = circuit_encoding(params, state)
    fid = jnp.dot(jnp.conj(zero_state), jnp.dot(trash, zero_state))
    return jnp.real(fid)


def cost_encoding(params, X):
    """
    Returns average fidelity for the dataset. This is negative as we want to maximize it.
    :param params: Variational Parameters
    :param X: Dataset Features
    :return: Cost
    """
    return -jnp.mean(jax.vmap(encoding_fidelity, in_axes=[None, 0])(params, X))


class_0_train_costs = []
class_1_train_costs = []


def fit(params, optimizer, class_0, class_1):
    """
    Train the model.
    :param params: Variational Parameters
    :param optimizer: Optimizer to be used
    :param class_0: Data Features with all label 0
    :param class_1: Data Features with all label 1
    :return: Trained Parameters
    """
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, X):
        loss_value, grads = jax.value_and_grad(cost_encoding)(params, X)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i in range(epochs):
        params, opt_state, loss_value = step(params, opt_state, class_0)
        class_0_train_costs.append(-cost_encoding(params, class_0))
        class_1_train_costs.append(-cost_encoding(params, class_1))

    return params


optimizer = optax.adam(lr)
weights_encoder = fit(weights_encoder, optimizer, class_0_train, class_1_train)

plt.plot(class_0_train_costs, label="Class 0")
plt.plot(class_1_train_costs, label="Class 1")
plt.legend()
plt.title("Encoder Fidelity of Training Data")
plt.xlabel("Epochs")
plt.ylabel("Fidelity")
plt.xticks(np.arange(0, epochs, epochs // 10), np.arange(0, epochs, epochs // 10) + 1)
plt.tight_layout()
# plt.savefig(f'plots/autoencoder_{encoding}_{layers}_cost_plot')
plt.show()


def get_fid_scores_encoded(params, X):
    """
    Gets fidelity values for the dataset
    :param params: Variational Parameters
    :param X: Dataset
    :return: Fidelity Values
    """
    return jax.vmap(encoding_fidelity, in_axes=[None, 0])(params, X)


class_0_train_fidelities = get_fid_scores_encoded(
    weights_encoder, class_0_train
).tolist()
class_1_train_fidelities = get_fid_scores_encoded(
    weights_encoder, class_1_train
).tolist()

plt.hist(
    class_1_train_fidelities, bins=100, label="Class 0", color="skyblue", alpha=0.4
)
plt.hist(class_0_train_fidelities, bins=100, label="Class 1", color="red", alpha=0.4)
plt.title("Classification of training data")
plt.xlabel("Fidelity")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
# plt.savefig(f'plots/autoencoder_{encoding}_{layers}_classification_plot')
plt.show()


def get_split():
    """
    Finds the best fidelity split for classification
    :return: accuracy for each split value, Index of best split
    """
    acc = []
    for split in np.linspace(0, 1, 101):
        class_1 = []
        for i in class_1_train_fidelities:
            if i < split:
                class_1.append(1)
            else:
                class_1.append(0)
        class_0 = []
        for i in class_0_train_fidelities:
            if i > split:
                class_0.append(1)
            else:
                class_0.append(0)
        acc.append((sum(class_1) + sum(class_0)) / (len(class_1) + len(class_0)))
    return acc, np.argmax(acc)


accuracies, split = get_split()
testing_data_fidelities = get_fid_scores_encoded(weights_encoder, X_test)

print("Accuracy after Training:")
print("Training Data:", np.round(accuracies[split], 4) * 100)
print(
    "Testing Data:",
    np.round(
        accuracy_score(
            y_true=y_test * 2 - 1,
            y_pred=-np.sign(testing_data_fidelities - split * 0.01),
        ),
        4,
    )
    * 100,
)
