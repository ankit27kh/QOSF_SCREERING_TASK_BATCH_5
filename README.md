# QOSF COHORT 5 MENTORSHIP TASK 2

This repository contains the solution to task 2 of the screening tasks for batch 5. The task is the following:

Task 2 Encoding and Classifier

Encoding the following files in a quantum
circuit [mock_train_set.csv](https://drive.google.com/file/d/1PIcC1mJ_xi4u1-2gxyoStg2Rg_joSBIB/view?usp=sharing)
and [mock_test_set.csv](https://drive.google.com/file/d/1aapYE69pTeNHZ6u-qKAoLfd1HWZVWPlB/view?usp=sharing) in at least
two different ways (these could be basis, angle, amplitude, kernel or random encoding). Design a variational quantum
circuit for each of the encodings, uses the column 4 as the target, this is a binary class 0 and 1. You must use the
data from column0 to column3 for your proposed classifier. Consider the ansatz you are going to design as a layer and
find out how many layers are necessary to reach the best performance.

Analyze and discuss the results.

Feel free to use existing frameworks (e.g. PennyLane, Qiskit) for creating and training the circuits. This PennyLane
demo can be
useful: [Training a quantum circuit with Pytorch](https://pennylane.ai/qml/demos/tutorial_state_preparation.html), This
Quantum Tensorflow tutorial can be
useful: [Training a quantum circuit with Tensorflow](https://www.tensorflow.org/quantum/tutorials/mnist).

For the variational circuit, you can try any circuit you want. You can start from one with a layer of RX, RZ and CNOTs.

## SUBMISSION DETAILS

### File Details:

* classifier_autoencoder.py:

  This file uses a Quantum Autoencoder to classsify the data. For encoding the data, Angle and Amplitude encoding are
  used. Encoding to be used can be selected inside.

  For Angle Encoding the Autoencoder uses 4-1 encoding, and for Amplitude Encoding, it uses 2-1.

  The encoder is trained on only one class of data. Afterwards, the fidelity of the trash state is compared for both
  classes. The fidelity for the trained class is higher than the untrained class. This makes the classifier.
* autoencoder_results.xlsx

  This file contains the results for the autoencoder classifier using both enocodings for different number of layers.
* classifier_fidelity.py

  This file uses a Data-Re-Uploading classifier with fidelity cost function to classify the data. The dataset to be used
  can be either the original set or a modified dataset.
* fidelity_results.xlsx

  This file contains results for the data-re-upload classifier with fidelity cost function for original and modified
  dataset for different number of qubits and layers.
* classifier_probability.py

  This file uses a Data-Re-Uploading classifier with log-loss cost function to classify the data. The dataset to be used
  can be either the original set or a modified dataset.
* classifier_results.xlsx

  This file contains results for the data-re-upload classifier with log-loss cost function for original and modified
  dataset for different number of qubits and layers.
* mock_train_set.csv

  Training Data
* mock_test_set.csv

  Testing Data
* README.md
* requirements.txt
* utilities.py

  This file contains utility functions.

### Data Set Details:

Training data contains 300 data points.

Testing data contains 120 data points.

Original data has 4 features and a binary label.

The modified data makes a new feature which is the **log of the product of the 4 features**. This becomes the single
feature of the modified dataset. This single feature dataset can be trained better than the original dataset.

As the data range is very large, we use data standardisation. Data is first standardised and then range is changed
to [-1,1]. The scalers are fitted using training data and then both training and testing data is fitted.

### Results:

#### Plots folder:

This folder contains plots from all runs of all three files. The naming convention of the plots is:

* Autoencoder:

  `autoencoder_{encoding}_{num_layers}_{plot_type}`

  encoding = 1 for Amplitude Encoding

  encoding = 2 for Angle Encoding

  num_layers in [1,5]

  plot_type -> classification plot for training data OR cost plot for training

* Fidelity and Probability:

  `{circuit_type}_{num_qubits}_{num_layers}_{modified_data}_cost_plot`

  circuit_type -> fidelity OR probability

  num_qubits in [1,5]

  num_layers in [1,5]

  modified_data = True for modified data and False for original data

**AUTOENCODER**

Angle Encoding performs better than Amplitude Encoding with this data. This is likely due to the different number of
trash qubits in these two encodings. We also notice that there is virtually no change in performance when increasing
number of layers. This most likely means we have already reached the limit of training.

Best Test Data Score is **79.17%** for 1 layer with Amplitude Encoding.

Best Test Data Score is **85.83%** for 3 layer with Angle Encoding.

**DATA RE-UPLOAD**

We find that the modified data performs appreciably better than the original data. It also has the added benefit of
being a single feature which reduces circuit depth.

We also notice that increasing the number of qubits and layers does not improve the performance. It rather has a
negative impact on the accuracy for using more than 3 qubits with fidelity cost function. Accuracy remains more or less
same for all layer and qubit combinations for log-loss cost function. The reduced performance can be attributed to
increased parameter space that can result in the training loop getting stuck in a plateau.

Fidelity Cost:

Best Test Data Score is **86.67%** for 2 layers and 1 qubit with original data.

Best Test Data Score is **98.33%** for 2 layers and 4 qubit with modified data.

Log-Loss Cost:

Best Test Data Score is **86.67%** for 1 layer and 1 qubit with original data.

Best Test Data Score is **97.5%** for 1 layer and 1 qubit with modified data.

_**The accuracy is calculated as a ratio which is rounded to 4 significant places before being reported as a percent.**_

_The data re-upload files contain two ansatz, one with entanglement and one without. In this analysis only the ansatz
without entanglement is used._