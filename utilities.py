import jax
import pennylane.numpy as np
import jax.numpy as jnp
import pennylane as qml


def get_states(n):
    """
    This function returns two states for a given number of qubits that are orthogonal.
    :param n: Number of Qubits
    :return: |100..0> and |00..01>
    """
    s1 = np.zeros([2 ** n, 1])
    s1[0] = 1
    s2 = np.zeros_like(s1)
    s2[-1] = 1
    s1, s2 = jnp.array(s1), jnp.array(s2)
    return jnp.array([s1, s2])


def cost_fidelity(parameters, x, y, states, circuit):
    """
    This function calculates the fidelity of the state returned by the circuit and a label state.
    This is fidelity cost function.
    :param parameters: Variational parameters for the circuit
    :param x: Feature vector
    :param y: Label
    :param states: Set of orthogonal states
    :param circuit: Circuit used for modelling
    :return: 1-fidelity for 1 data point
    """
    y_state = states[y]
    guess_state = circuit(parameters, x)
    f = jnp.abs(jnp.dot(y_state.conj().transpose(), guess_state)) ** 2
    return 1 - f


def map_cost_fidelity(parameters, x, y, states, circuit):
    """
    This returns the fidedlity cost for the complete dataset.
    :param parameters: Variational parameters for the circuit
    :param x: Dataset features
    :param y: Dataset labels
    :param states: Set of orthogonal states
    :param circuit: Circuit used for modelling
    :return: Average fidelity cost for the data
    """
    return jnp.mean(
        jax.vmap(cost_fidelity, in_axes=[None, 0, 0, None, None])(
            parameters, x, y, states, circuit
        )
    )


def predict_one_fidelity(x, parameters, states, circuit):
    """
    This function predicts the label for a data point.
    It checks the fidelity of the data point with both states and assigns the one for which the fidelity was higher.
    :param x: Feature vector
    :param parameters: Variational Parameters
    :param states: Orthogonal states
    :param circuit: Circuit used for modelling
    :return: Predicted label
    """
    fids = []
    for state in states:
        guess_state = circuit(parameters, x)
        fids.append(jnp.abs(jnp.dot(state.conj().transpose(), guess_state)) ** 2)
    return jnp.argmax(jnp.array(fids))


def predict_all_fidelity(x, parameters, states, circuit):
    """
    Predicts labels for the complete dataset
    """
    return jax.vmap(predict_one_fidelity, in_axes=[0, None, None, None])(
        x, parameters, states, circuit
    )


def cost_prob(parameters, x, y, circuit, eps):
    """
    This function calculates log-loss cross entropy for a data point
    :param parameters: Variational parameters
    :param x: Feature vector
    :param y: Label
    :param circuit: Circuit used for modelling
    :param eps: Minimum error
    :return: Cross entropy for a single data point
    """
    p = circuit(parameters, x)
    p0 = jnp.max(jnp.array([eps, p[0]]))
    p1 = jnp.max(jnp.array([eps, p[1]]))
    return jnp.asarray(y * jnp.log(p1) + (1 - y) * jnp.log(p0))


def map_cost_prob(parameters, x, y, circuit, eps):
    """
    Gives cross entropy for the complete dataset
    """
    return -(
        jnp.mean(
            jax.vmap(cost_prob, in_axes=[None, 0, 0, None, None])(
                parameters, x, y, circuit, eps
            )
        )
    )


def predict_one_prob(x, parameters, circuit):
    """
    Predicts the label for a data point.
    :param x: Feature vector
    :param parameters: Variational Parameters
    :param circuit: Circuit used for modelling
    :return: Predicted Label based on higher probability
    """
    return jnp.argmax(circuit(parameters, x))


def predict_all_prob(x, parameters, circuit):
    """
    Predicts labels based on probabilities for complete dateset
    """
    return jax.vmap(predict_one_prob, in_axes=[0, None, None])(x, parameters, circuit)


def encoder_layer(params, trash, data):
    """
    This is 1 layer of the encoder
    :param params: Variational Parameters
    :param trash: Trash Qubits
    :param data: Data Qubits
    """
    wires = trash + data
    idx = 0
    for i in wires:
        qml.Rot(params[idx], params[idx + 1], params[idx + 2], wires=i)
        idx += 3

    for i in wires:
        for j in wires:
            if i != j:
                qml.CRot(params[idx], params[idx + 1], params[idx + 2], wires=[i, j])
                idx += 3

    for i in wires:
        qml.Rot(params[idx], params[idx + 1], params[idx + 2], wires=i)
        idx += 3


def U(params, trash, data, num_layers):
    """
    Makes the encoder with given number of layers
    """
    for l in range(num_layers):
        encoder_layer(params[l], trash, data)
