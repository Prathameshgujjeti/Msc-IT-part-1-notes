#Practical 1a:  Design a simple linear neural network model.
# x=float(input("Enter value of x:")) 
# w=float(input("Enter value of weight w:")) 
# b=float(input("Enter value of bias b:")) 
 
# net = int(w*x+b) 
# if(net<0):     
#     out=0
# elif((net>=0)&(net<=1)): 
#     out =net
# else: 
#     out=1 
# print("net=",net) 
# print("output=",out)

#practical 1b : Calculate the output of neural net using both binary and bipolar sigmoidal function. 

# import numpy as np
# def binary_sigmoid(x):
#     return 1 / (1 + np.exp(-x))
# def bipolar_sigmoid(x):
#     return (2 / (1 + np.exp(-x))) - 1
# def neural_network(input_data, weights):
#     weighted_sum = np.dot(input_data, weights)
#     binary_sigmoid_output = binary_sigmoid(weighted_sum)
#     bipolar_sigmoid_output = bipolar_sigmoid(weighted_sum)
#     return binary_sigmoid_output, bipolar_sigmoid_output
# input_data = np.array([0.5, 0.3, 0.8])
# weights = np.array([0.2, 0.4, 0.7])
# binary_output, bipolar_output = neural_network(input_data, weights)
# print(f"Binary Sigmoid Output: {round(binary_output,3)}")
# print(f"Bipolar Sigmoid Output: {round(bipolar_output,3)}")

# 2nd code

# n = int(input("Enter number of elements: "))
# print("Enter the inputs")
# inputs = []
# for i in range(0, n):
#     ele = float(input())
#     inputs.append(ele)
# print("Entered inputs:", inputs)
# print("Enter the weights")
# weights = []
# for i in range(0, n):
#     ele = float(input())
#     weights.append(ele)
# print("Entered weights:", weights)
# print("The net input can be calculated as Yin = x1w1 + x2w2 + x3w3")
# Yin = []
# for i in range(0, n):
#     Yin.append(inputs[i] * weights[i])
# print("Net input (Yin):", round(sum(Yin), 3))

# 3rd code

# n = int(input("Enter number of elements: "))
# print("Enter the inputs:")
# inputs = [float(input()) for _ in range(n)]
# print("Entered inputs:", inputs)
# print("Enter the weights:")
# weights = [float(input()) for _ in range(n)]
# print("Entered weights:", weights)
# bias = float(input("Enter bias value:"))
# print("The net input can be calculated as Yin = b + x1w1 + x2w2:")
# Yin = [inputs[i] * weights[i] for i in range(n)]
# net_input = round(sum(Yin) + bias, 3)
# print("Net input (Yin):", net_input)

# Practical 2a : Generate AND/NOT function using McCulloch-Pitts neural net. 
# def mcculloch_pitts_neuron(inputs, weights, threshold):
#     weighted_sum = sum(i * w for i, w in zip(inputs, weights))

#     output = 1 if weighted_sum >= threshold else 0
#     return output

# def and_not_gate(input1, input2):
#     weights_and = [1, 1]
#     threshold_and = 2
#     output_and = mcculloch_pitts_neuron([input1, input2], weights_and, threshold_and)
#     output_and_not = 1 - output_and
#     return output_and_not

# result = and_not_gate(1, 0)
# print(f"AND/NOT(1, 0) = {result}")

# 2nd code

# num_ip = int(input("Enter the number of inputs: ")) 

# w1 = 1
# w2 = 1

# print("For the", num_ip, "inputs, calculate the net input using Yin = x1w1 + x2w2")

# x1 = [int(input("x1 = ")) for _ in range(num_ip)]
# x2 = [int(input("x2 = ")) for _ in range(num_ip)]

# print("x1 =", x1)
# print("x2 =", x2)

# n = [x * w1 for x in x1]
# m = [x * w2 for x in x2]

# Yin = [n[i] + m[i] for i in range(num_ip)]
# print("Yin =", Yin)

# Yin_exc_inh = [n[i] - m[i] for i in range(num_ip)]
# print("After assuming one weight as excitatory and the other as inhibitory Yin =", Yin_exc_inh)

# Y = [1 if y >= 1 else 0 for y in Yin_exc_inh]
# print("Y =", Y)

# Practical 2b : Generate XOR function using McCulloch-Pitts neural net. 

# def mcculloch_pitts_neuron(inputs, weights, threshold):
#     weighted_sum = sum(i * w for i, w in zip(inputs, weights))

#     output = 1 if weighted_sum >= threshold else 0
#     return output

# def xor_gate(input1, input2):
#     weights_and1 = [1, 1]
#     weights_and2 = [-1, -1]
#     weights_or = [1, 1]
#     threshold_and = 1
#     threshold_or = 1

#     output_and1 = mcculloch_pitts_neuron([input1, input2], weights_and1, threshold_and)
#     output_and2 = mcculloch_pitts_neuron([input1, input2], weights_and2, threshold_and)
#     output_or = mcculloch_pitts_neuron([output_and1, output_and2], weights_or, threshold_or)

#     return output_or

# input1 = int(input("Enter input 1 (0 or 1): "))
# input2 = int(input("Enter input 2 (0 or 1): "))

# result = xor_gate(input1, input2)

# print(f"XOR({input1}, {input2}) = {result}")

# 2nd code
# import numpy as np

# def get_weights_and_threshold():
#     print('Enter weights')
#     w11 = int(input('Weight w11='))
#     w12 = int(input('Weight w12='))
#     w21 = int(input('Weight w21='))
#     w22 = int(input('Weight w22='))
#     v1 = int(input('Weight v1='))
#     v2 = int(input('Weight v2='))

#     print('Enter Threshold Value')
#     theta = int(input('Theta='))

#     return w11, w12, w21, w22, v1, v2, theta

# def main():
#     x1 = np.array([0, 0, 1, 1])
#     x2 = np.array([0, 1, 0, 1])
#     z = np.array([0, 1, 1, 0])

#     w11, w12, w21, w22, v1, v2, theta = get_weights_and_threshold()

#     con = 1
#     y1 = np.zeros((4,))
#     y2 = np.zeros((4,))
#     y = np.zeros((4,))

#     while con == 1:
#         zin1 = x1 * w11 + x2 * w21
#         zin2 = x1 * w12 + x2 * w22

#         print("z1", zin1)
#         print("z2", zin2)

#         for i in range(4):
#             y1[i] = 1 if zin1[i] >= theta else 0
#             y2[i] = 1 if zin2[i] >= theta else 0

#         yin = y1 * v1 + y2 * v2

#         for i in range(4):
#             y[i] = 1 if yin[i] >= theta else 0

#         print("yin", yin)
#         print('Output of Net')
#         y = y.astype(int)
#         print("y", y)
#         print("z", z)

#         if np.array_equal(y, z):
#             con = 0
#         else:
#             print("Net is not learning. Enter another set of weights and threshold value")
#             w11, w12, w21, w22, v1, v2, theta = get_weights_and_threshold()

#     print("McCulloch-Pitts Net for XOR function")
#     print("Weights of Neuron Z1")
#     print(w11)
#     print(w21)
#     print("Weights of Neuron Z2")
#     print(w12)
#     print(w22)
#     print("Weights of Neuron Y")
#     print(v1)
#     print(v2)
#     print("Threshold value")
#     print(theta)

# if __name__ == "__main__":
#     main()

# Practical 3a : Write a program to implement Hebb’s rule.
# import numpy as np

# def hebbs_rule(inputs, learning_rate):
#     inputs = np.array(inputs)
#     outer_product = np.outer(inputs, inputs)
#     synaptic_weights = learning_rate * outer_product
#     return synaptic_weights

# pattern_size = int(input("Enter the size of the binary pattern: "))
# binary_pattern = []

# for i in range(pattern_size):
#     bit = int(input(f"Enter bit {i + 1} (0 or 1): "))
#     binary_pattern.append(bit)

# learning_rate = float(input("Enter the learning rate (e.g., 0.1): "))

# synaptic_weights = hebbs_rule(binary_pattern, learning_rate)

# print("\nLearned Synaptic Weights:")
# print(synaptic_weights)

# 2nd code 
# import numpy as np

# x1 = np.array([1, 1, 1, -1, 1, -1, 1, 1, 1])
# x2 = np.array([1, 1, 1, 1, -1, 1, 1, 1, 1])

# y = np.array([1, -1])

# wtold = np.zeros((9,))
# wtnew = np.zeros((9,))
# wtnew = wtnew.astype(int)
# wtold = wtold.astype(int)
# bais = 0
# print("First input with target = 1")
# for i in range(0, 9):
#     wtold[i] = wtold[i] + x1[i] * y[0]

# wtnew = wtold
# b = bais + y[0]
# print("new wt =", wtnew)
# print("Bias value", b)
# print("Second input with target = -1")
# for i in range(0, 9):
#     wtnew[i] = wtold[i] + x2[i] * y[1]

# b = b + y[1]
# print("new wt =", wtnew)
# print("Bias value", b)

# Practical 3b : Write a program to implement of delta rule.
# import numpy as np

# def delta_rule(inputs, target, weights, learning_rate):
#     inputs = np.array(inputs)
#     weights = np.array(weights)

#     predicted_output = np.dot(inputs, weights)

#     error = target - predicted_output

#     weights += learning_rate * error * inputs

#     return weights

# num_features = int(input("Enter the number of features: "))

# weights = np.zeros(num_features)

# target_output = float(input("Enter the target output: "))

# learning_rate = float(input("Enter the learning rate (e.g., 0.1): "))

# input_values = [float(input(f"Enter input feature {i + 1}: ")) for i in range(num_features)]

# weights = delta_rule(input_values, target_output, weights, learning_rate)

# print("\nUpdated Weights:")
# print(weghts)

# 2nd Code

# import numpy as np
# np.set_printoptions(precision=2)
# x = np.zeros((3,))
# weights = np.zeros((3,))
# desired = np.zeros((3,))
# actual = np.zeros((3,))

# for i in range(0, 3):
#     x[i] = float(input("Initial inputs:"))

# for i in range(0, 3):
#     weights[i] = float(input("Initial weights:"))

# for i in range(0, 3):
#     desired[i] = float(input("Desired output:"))

# a = float(input("Enter learning rate:"))

# actual = x * weights
# print("actual", actual)
# print("desired", desired)

# while True:
#     if np.array_equal(desired, actual):
#         break 
#     else:
#         for i in range(0, 3):
#             weights[i] = weights[i] + a * (desired[i] - actual[i])
#         actual = x * weights
#         print("weights", weights)
#         print("actual", actual)
#         print("desired", desired)
#         print("*" * 30)

# print("Final output")
# print("Corrected weights", weights)
# print("actual", actual)
# print("desired", desired)

# Practical 4a :Write a program for Back Propagation Algorithm 

# import numpy as np
# import math

# np.set_printoptions(precision=2)

# v1, v2, w = np.array([0.6, 0.3]), np.array([-0.1, 0.4]), np.array([-0.2, 0.4, 0.1])
# b1, b2, x1, x2, alpha = 0.3, 0.5, 0, 1, 0.25

# def calculate_net_input(b, x, weights):
#     return round(b + x.dot(weights[:2]), 4)

# def calculate_activation(input_value):
#     return round(1 / (1 + math.exp(-input_value)), 4)

# def calculate_net_output(inputs, weights):
#     return inputs.dot(weights)

# def update_weights(weights, alpha, delta, inputs):
#     return weights + alpha * delta * inputs

# zin1 = calculate_net_input(b1, np.array([x1, x2]), v1)
# print("z1 =", zin1)

# zin2 = calculate_net_input(b2, np.array([x1, x2]), v2)
# print("z2 =", zin2)

# z1, z2 = calculate_activation(zin1), calculate_activation(zin2)
# print("z1 =", z1)
# print("z2 =", z2)

# yin = calculate_net_output(np.array([1, z1, z2]), w)
# print("yin =", yin)

# y = calculate_activation(yin)
# print("y =", y)

# fyin= y * (1 - y)
# dk =(1 - y) * fyin
# print("dk =", dk)

# din1, din2 = dk * w[1], dk * w[2]
# print("din1 =", din1)
# print("din2 =", din2)

# fzin1, fzin2 = z1 * (1 - z1), z2 * (1 - z2)
# d1, d2 = din1 * fzin1, din2 * fzin2
# print("d1 =", d1)
# print("d2 =", d2)

# dv11, dv21, dv01 = alpha * d1 * x1, alpha * d1 * x2, alpha * d1
# dv12, dv22, dv02 = alpha * d2 * x1, alpha * d2 * x2, alpha * d2
# print("dv11 =", dv11)
# print("dv21 =", dv21)
# print("dv01 =", dv01)
# print("dv12 =", dv12)
# print("dv22 =", dv22)
# print("dv02 =", dv02)

# v1 += np.array([dv11, dv12])
# print("v =", v1)

# v2 += np.array([dv21, dv22])
# print("v2 =", v2)

# w[1:] = update_weights(w[1:], alpha, dk, np.array([z1, z2]))
# b1, b2, w[0] = b1 + dv01, b2 + dv02, w[0] + alpha * dk

# print("w =", w)
# print("bias b1 =",round(b1,3), " b2 =", round(b2,3))

# Practical 4b : Write a program for error Backpropagation algorithm. 

# import math
# a0, t = -1, -1

# w10 = float(input("Enter weight for the first network: "))
# b10 = float(input("Enter base for the first network: "))
# w20 = float(input("Enter weight for the second network: "))
# b20 = float(input("Enter base for the second network: "))
# c = float(input("Enter learning coefficient: "))

# n1 = w10 * c + b10
# a1 = math.tanh(n1)
# n2 = w20 * a1 + b20
# a2 = math.tanh(n2)

# e = t - a2
# s2 = -2 * (1 - a2**2) * e
# s1 = (1 - a1**2) * w20 * s2

# w21, w11 = w20 - c * s2 * a1, w10 - c * s1 * a0
# b21, b11 = b20 - c * s2, b10 - c * s1

# print("Updated weight of the first network w11 =", w11)
# print("Updated weight of the second network w21 =", w21)
# print("Updated base of the first network b10 =", b10)
# print("Updated base of the second network b20 =", b20)

# Practical 5a : Write a program for Hopfield Network. 
# class Neuron:
#     def __init__(self, weights):
#         self.activation = 0
#         self.weightv = weights

#     def act(self, inputs):
#         return sum(x * w for x, w in zip(inputs, self.weightv))

# class Network:
#     def __init__(self, weights_list):
#         self.nrn = [Neuron(weights) for weights in weights_list]
#         self.output = [0] * len(self.nrn)

#     @staticmethod
#     def threshold(value):
#         return 1 if value >= 0 else 0

#     def activation(self, pattern):
#         for i, neuron in enumerate(self.nrn):
#             neuron.activation = neuron.act(pattern)
#             self.output[i] = self.threshold(neuron.activation)

# def main():
#     wt1, wt2, wt3, wt4 = [0, -3, 3, -3], [-3, 0, -3, 3], [3, -3, 0, -3], [-3, 3, -3, 0]
#     h1 = Network([wt1, wt2, wt3, wt4])

#     patterns = [[1, 0, 1, 0], [0, 1, 0, 1]]

#     for patrn in patterns:
#         print(f"\nPresenting pattern {patrn}")
#         h1.activation(patrn)

#         for i, output in enumerate(h1.output):
#             match_status = "matches" if output == patrn[i] else "discrepancy occurred"
#             print(f"Pattern = {patrn[i]}, Output = {output}, Component {match_status}")

#         print("\n\n")

# if __name__ == "__main__":
#     main()

# Practical 5b : Write a program for Hopfield Network. 

# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.metrics import pairwise_distances_argmin_min
# import matplotlib.pyplot as plt

# class RadialBasisFunctionNetwork:
#     def __init__(self, num_centers, learning_rate=0.1):
#         self.num_centers = num_centers
#         self.learning_rate = learning_rate
#         self.centers = None
#         self.weights = None

#     def k_means_initialize(self, X):
#         kmeans = KMeans(n_clusters=self.num_centers)
#         kmeans.fit(X)
#         self.centers = kmeans.cluster_centers_

#     def calculate_phi(self, x):
#         distances = np.linalg.norm(x - self.centers, axis=1)
#         return np.exp(-0.5 * distances**2)

#     def train(self, X, y, epochs=100):
#         self.k_means_initialize(X)
#         self.weights = np.random.rand(self.num_centers)

#         for epoch in range(epochs):
#             for i in range(X.shape[0]):
#                 phi = self.calculate_phi(X[i])
#                 output = np.dot(phi, self.weights)
#                 error = y[i] - output
#                 self.weights += self.learning_rate * error * phi

#     def predict(self, X):
#         predictions = []
#         for i in range(X.shape[0]):
#             phi = self.calculate_phi(X[i])
#             output = np.dot(phi, self.weights)
#             predictions.append(output)
#         return np.array(predictions)

# def main():
#     np.random.seed(0)
#     X = np.sort(5 * np.random.rand(100, 1), axis=0)
#     y = np.sin(X).ravel()

#     rbf_network = RadialBasisFunctionNetwork(num_centers=10, learning_rate=0.1)
#     rbf_network.train(X, y, epochs=200)

#     X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
#     y_pred = rbf_network.predict(X_test)

#     plt.figure(figsize=(8, 6))
#     plt.plot(X, y, 'ro', label='Actual')
#     plt.plot(X_test, y_pred, 'b-', label='RBF Prediction')
#     plt.title('Radial Basis Function Network')
#     plt.legend()
#     plt.show()

# if __name__ == "__main__":
#     main()

# Practical 6a : Kohonen Self organizing map

# from minisom import MiniSom
# import matplotlib.pyplot as plt

# data = [
#     [0.80, 0.55, 0.22, 0.03],
#     [0.82, 0.50, 0.23, 0.03],
#     [0.80, 0.54, 0.22, 0.03],
#     [0.80, 0.53, 0.26, 0.03],
#     [0.79, 0.56, 0.22, 0.03],
#     [0.75, 0.60, 0.25, 0.03],
#     [0.77, 0.59, 0.22, 0.03]
# ]

# som = MiniSom(6, 6, 4, sigma=0.3, learning_rate=0.5)  # Initialization of 6x6 SOM
# som.train_random(data, 100)  # Trains the SOM with 100 iterations

# plt.imshow(som.distance_map())
# plt.show()

# Practical 6b : Adaptive resonance theory

# import numpy as np
# class ART:
#     def __init__(self, n=5, m=10, rho=0.5):
        
#         self.F1 = np.ones(n)
#         self.F2 = np.ones(m)
#         self.Wf = np.random.random((m, n))
#         self.Wb = np.random.random((n, m))
#         self.rho = rho
#         self.active = 0

#     def learn(self, X):
#         self.F2[...] = np.dot(self.Wf, X)
#         I = np.argsort(self.F2[:self.active].ravel())[::-1]
        
#         for i in I:
#             d = (self.Wb[:, i] * X).sum() / X.sum()
#             if d >= self.rho:
#                 self.Wb[:, i] *= X
#                 self.Wf[i, :] = self.Wb[:, i] / (0.5 + self.Wb[:, i].sum())
#                 return self.Wb[:, i], i
        
#         if self.active < self.F2.size:
#             i = self.active
#             self.Wb[:, i] *= X
#             self.Wf[i, :] = self.Wb[:, i] / (0.5 + self.Wb[:, i].sum())
#             self.active += 1
#             return self.Wb[:, i], i

#         return None, None

# if __name__ == '__main__':
#     np.random.seed(1)

# network = ART(5, 10, rho=0.5)
# data = [
#     " O ", " O O",
#     " O",
#     " O O",
#     " O",
#     " O O",
#     " O",
#     " OO O",
#     " OO ",
#     " OO O",
#     " OO ",
#     "OOO ",
#     "OO ",
#     "O ",
#     "OO ",
#     "OOO ",
#     "OOOO ",
#     "OOOOO",
#     "O ",
#     " O ",
#     " O ",
#     " O ",
#     " O",
#     " O O",
#     " OO O",
#     " OO ",
#     "OOO ",
#     "OO ",
#     "OOOO ",
#     "OOOOO"
# ]

# for i in range(len(data)):
#     X = np.zeros(len(data[i]))

#     for j in range(len(data[i])):
#         X[j] = (data[i][j] == 'O')

#     X = np.resize(X, network.Wf.shape[1])

#     Z, k = network.learn(X)
#     print("|%s|" % data[i], "-> class", k)

# def letter_to_array(letter):
#     ''' Convert a letter to a numpy array '''
#     shape = len(letter), len(letter[0])
#     Z = np.zeros(shape, dtype=int)
    
#     for row in range(Z.shape[0]):
#         for column in range(Z.shape[1]):
#             if column < len(letter[row]) and letter[row][column] == '#':
#                 Z[row][column] = 1

#     return Z
# def print_letter(Z):
#     ''' Print an array as if it was a letter '''
#     for row in range(Z.shape[0]):
#         for col in range(Z.shape[1]):
#             if Z[row, col]:
#                 print('#', end="")
#             else:
#                 print(' ', end="")
#         print()

# A = letter_to_array([' #### ',
#                     '# #',
#                     '# #',
#                     '######',
#                     '# #',
#                     '# #',
#                     '# #'])

# B = letter_to_array(['##### ',
#                     '# #',
#                     '# #',
#                     '##### ',
#                     '# #',
#                     '# #',
#                     '##### '])

# C = letter_to_array([' #### ',
#                     '# #',
#                     '# ',
#                     '# ',
#                     '# ',
#                     '# #',
#                     ' #### '])

# D = letter_to_array(['##### ',
#                     '# #',
#                     '# #',
#                     '# #',
#                     '# #',
#                     '# #',
#                     '##### '])

# E = letter_to_array(['######',
#                     '# ',
#                     '# ',
#                     '#### ',
#                     '# ',
#                     '# ',
#                     '######'])

# F = letter_to_array(['######',
#                     '# ',
#                     '# ',
#                     '#### ',
#                     '# ',
#                     '# ',
#                     '# '])

# samples = [A, B, C, D, E, F]
# network = ART(6 * 7, 10, rho=0.15)

# for i in range(len(samples)):
#     Z, k = network.learn(samples[i].ravel())
#     print("%c" % (ord('A') + i), "-> class", k)
#     print_letter(Z.reshape(7, 6))

# Testing art by creating array dataset
# import numpy as np
# def sigmoid(x):
#     output = 1 / (1 + np.exp(-x))
#     return output
# def sigmoid_output_to_derivative(output):
#     return output * (1 - output)

# X = np.array([
#     [0, 1],
#     [0, 1],
#     [1, 0],
#     [1, 0]
# ])
# y = np.array([[0, 0, 1, 1]]).T
# np.random.seed(1)
# synapse_0 = 2 * np.random.random((2, 1)) - 1
# for iter in range(10000):
#     layer_0 = X
#     layer_1 = sigmoid(np.dot(layer_0, synapse_0))
#     layer_1_error = layer_1 - y

#     layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
#     synapse_0_derivative = np.dot(layer_0.T, layer_1_delta)

#     synapse_0 -= synapse_0_derivative

# print("Output After Training:")
# print(layer_1)

# By providing data
# import math
# import sys

# class ART1_Example1:
#     def __init__(self, inputSize, numClusters, vigilance, numPatterns, numTraining, patternArray):
#         self.mInputSize = inputSize
#         self.mNumClusters = numClusters 
#         self.mVigilance = vigilance 
#         self.mNumPatterns = numPatterns 
#         self.mNumTraining = numTraining 
#         self.mPatterns = patternArray
#         self.bw = [] 
#         self.tw = [] 
#         self.f1a = []
#         self.f1b = []
#         self.f2 = []

#     def initialize_arrays(self):
#         sys.stdout.write("Weights initialized to:")
#         for i in range(self.mNumClusters):
#             self.bw.append([0.0] * self.mInputSize)
#             for j in range(self.mInputSize):
#                 self.bw[i][j] = 1.0 / (1.0 + self.mInputSize)
#                 sys.stdout.write(str(self.bw[i][j]) + ", ")
#             sys.stdout.write("\n")
#         sys.stdout.write("\n")

#         for i in range(self.mNumClusters):
#             self.tw.append([0.0] * self.mInputSize)
#             for j in range(self.mInputSize):
#                 self.tw[i][j] = 1.0
#                 sys.stdout.write(str(self.tw[i][j]) + ", ")
#             sys.stdout.write("\n")
#         sys.stdout.write("\n")

#         self.f1a = [0.0] * self.mInputSize
#         self.f1b = [0.0] * self.mInputSize
#         self.f2 = [0.0] * self.mNumClusters

#     def get_vector_sum(self, nodeArray):
#         total = sum(nodeArray)
#         return total

#     def get_maximum(self, nodeArray):
#         maximum = 0
#         foundNewMaximum = False
#         length = len(nodeArray)
#         done = False
#         while not done:
#             foundNewMaximum = False
#             for i in range(length):
#                 if i != maximum:
#                     if nodeArray[i] > nodeArray[maximum]:
#                         maximum = i
#                         foundNewMaximum = True
#             if not foundNewMaximum:
#                 done = True
#         return maximum

#     def test_for_reset(self, activationSum, inputSum, f2Max):
#         doReset = False
#         if float(activationSum) / float(inputSum) >= self.mVigilance:
#             doReset = False
#         else:
#             self.f2[f2Max] = -1.0  
#             doReset = True  
#         return doReset

#     def update_weights(self, activationSum, f2Max):
#         for i in range(self.mInputSize):
#             self.bw[f2Max][i] = (2.0 * float(self.f1b[i])) / (1.0 + float(activationSum))
#         for i in range(self.mNumClusters):
#             for j in range(self.mInputSize):
#                 sys.stdout.write(str(self.bw[i][j]) + ", ")
#             sys.stdout.write("\n")
#         sys.stdout.write("\n")

#         for i in range(self.mInputSize):
#             self.tw[f2Max][i] = self.f1b[i]
#         for i in range(self.mNumClusters):
#             for j in range(self.mInputSize):
#                 sys.stdout.write(str(self.tw[i][j]) + ", ")
#             sys.stdout.write("\n")
#         sys.stdout.write("\n")

#     def ART1(self):
#         inputSum = 0
#         activationSum = 0
#         f2Max = 0
#         reset = True
#         sys.stdout.write("Begin ART1:\n")
#         for k in range(self.mNumPatterns):
#             sys.stdout.write("Vector: " + str(k) + "\n\n")
            
#             for i in range(self.mNumClusters):
#                 self.f2[i] = 0.0

#             for i in range(self.mInputSize):
#                 self.f1a[i] = self.mPatterns[k][i]

#             inputSum = self.get_vector_sum(self.f1a) 
#             sys.stdout.write("InputSum (si) = " + str(inputSum) + "\n\n")

#             for i in range(self.mInputSize):
#                 self.f1b[i] = self.f1a[i]

#             for i in range(self.mNumClusters):
#                 for j in range(self.mInputSize):
#                     self.f2[i] += self.bw[i][j] * float(self.f1a[j]) 
#                     sys.stdout.write(str(self.f2[i]) + ", ")
#                 sys.stdout.write("\n")
#                 sys.stdout.write("\n")

#             reset = True
#             while reset == True:
#                 f2Max = self.get_maximum(self.f2)

#                 for i in range(self.mInputSize):
#                     sys.stdout.write(str(self.f1b[i]) + " * " + str(self.tw[f2Max][i]) + " = " + str(self.f1b[i] * self.tw[f2Max][i]) + "\n")
#                     self.f1b[i] = self.f1a[i] * math.floor(self.tw[f2Max][i])

#                 activationSum = self.get_vector_sum(self.f1b)
#                 sys.stdout.write("ActivationSum (x(i)) = " + str(activationSum) + "\n\n")
#                 reset = self.test_for_reset(activationSum, inputSum, f2Max)

#                 if k < self.mNumTraining:
#                     self.update_weights(activationSum, f2Max)

#                 sys.stdout.write("Vector #" + str(k) + " belongs to cluster #" + str(f2Max) + "\n\n")

#     def print_results(self):
#         sys.stdout.write("Final weight values:\n")
#         for i in range(self.mNumClusters):
#             for j in range(self.mInputSize):
#                 sys.stdout.write(str(self.bw[i][j]) + ", ")
#             sys.stdout.write("\n")
#             sys.stdout.write("\n")

#         for i in range(self.mNumClusters):
#             for j in range(self.mInputSize):
#                 sys.stdout.write(str(self.tw[i][j]) + ", ")
#             sys.stdout.write("\n")
#             sys.stdout.write("\n")


# if __name__ == '__main__':
#     N = 4
#     M = 3
#     VIGILANCE = 0.4
#     PATTERNS = 7
#     TRAINING_PATTERNS = 4
#     PATTERN_ARRAY = [
#         [1, 1, 0, 0],
#         [0, 0, 0, 1],
#         [1, 0, 0, 0],
#         [0, 0, 1, 1],
#         [0, 1, 0, 0],
#         [0, 0, 1, 0],
#         [1, 0, 1, 0]
#     ]

#     art1 = ART1_Example1(N, M, VIGILANCE, PATTERNS, TRAINING_PATTERNS, PATTERN_ARRAY)
#     art1.initialize_arrays()
#     art1.ART1()
#     art1.print_results()
# Practical 7a : Write a program for Linear separation. 
# import numpy as np
# import matplotlib.pyplot as plt

# def distance_function(a, b, c, x, y):
#     nom = a * x + b * y + c
#     pos = 0 if nom == 0 else -1 if (nom < 0 and b < 0) or (nom > 0 and b > 0) else 1
#     return np.absolute(nom) / np.sqrt(a ** 2 + b ** 2), pos

# points = [(3.5, 1.8), (1.1, 3.9)]

# fig, ax = plt.subplots()
# ax.set(xlabel="sweetness", ylabel="sourness", xlim=[-1, 6], ylim=[-1, 8])

# size = 10

# for index, (x, y) in enumerate(points):
#     marker = "o" if index == 0 else "oy"
#     color = "darkorange" if index == 0 else None
#     ax.plot(x, y, marker, color=color, markersize=size)

# step = 0.05

# for x in np.arange(0, 1 + step, step):
#     slope = np.tan(np.arccos(x))
#     Y = slope * np.arange(-0.5, 5, 0.1)
#     results = [distance_function(slope, -1, 0, point[0], point[1]) for point in points]

#     line_color = "g-" if results[0][1] != results[1][1] else "r-"
#     ax.plot(np.arange(-0.5, 5, 0.1), Y, line_color)

# plt.show()

# Practical 7b : b Write a program for Hopfield network model for associative memory

# from neurodynex.hopfield_network import network, pattern_tools, plot_tools
# import matplotlib.pyplot as plt

# def modified_plot_overlap_matrix(overlap_matrix, title="Overlap Matrix"):
#     fig, ax = plt.subplots()
#     im = ax.imshow(overlap_matrix, cmap="hot", interpolation="nearest", vmin=-1, vmax=1)
#     plt.title(title)
#     plt.colorbar(im)
#     plt.show()

# def main():
#     pattern_size = 5
#     hopfield_net = network.HopfieldNetwork(nr_neurons=pattern_size ** 2)
#     factory = pattern_tools.PatternFactory(pattern_size, pattern_size)

#     pattern_list = [factory.create_checkerboard()] + factory.create_random_pattern_list(nr_patterns=3, on_probability=0.5)

#     plot_tools.plot_pattern_list(pattern_list)

#     overlap_matrix = pattern_tools.compute_overlap_matrix(pattern_list)
#     modified_plot_overlap_matrix(overlap_matrix, title="Overlap Matrix")

#     hopfield_net.store_patterns(pattern_list)

#     noisy_init_state = pattern_tools.flip_n(pattern_list[0], nr_of_flips=4)
#     hopfield_net.set_state_from_pattern(noisy_init_state)

#     states = hopfield_net.run_with_monitoring(nr_steps=4)
#     states_as_patterns = factory.reshape_patterns(states)
#     plot_tools.plot_state_sequence_and_overlap(states_as_patterns, pattern_list, reference_idx=0, suptitle="Network dynamics")

# if __name__ == "__main__":
#     main()

# Practical 8a : Membership and Identity Operators | in, not in,
# fruits = ['apple', 'banana', 'cherry']

# print('banana' in fruits)
# print('orange' not in fruits)

# Practical 8b : Membership and Identity Operators is, is not
# x = 5
# y = 5.0
# z = x

# print(x is y)
# print(x is z)
# print(x is not y)

# Practical 9a : Find ratios using fuzzy logic
# from fuzzywuzzy import fuzz, process

# s1 = "I love fuzzysforfuzzys"
# s2 = "I am loving fuzzysforfuzzys"

# print("FuzzyWuzzy Ratio:", fuzz.ratio(s1, s2))
# print("FuzzyWuzzyPartialRatio:", fuzz.partial_ratio(s1, s2))
# print("FuzzyWuzzyTokenSortRatio:", fuzz.token_sort_ratio(s1, s2))
# print("FuzzyWuzzyTokenSetRatio:", fuzz.token_set_ratio(s1, s2))
# print("FuzzyWuzzyWRatio:", fuzz.WRatio(s1, s2), '\n\n')

# query = 'fuzzys for fuzzys'
# choices = ['fuzzy for fuzzy', 'fuzzy fuzzy', 'g. for fuzzys']

# print("List of ratios:")
# print(process.extract(query, choices), '\n')

# best_match = process.extractOne(query, choices)
# print("Best among the above list:", best_match)

# Practical 9b : Solve Tipping problem using fuzzy logic
# import numpy as np
# import skfuzzy as fuzz
# from skfuzzy import control as ctrl

# quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
# service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
# tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

# quality.automf(3)
# service.automf(3)

# tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
# tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
# tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

# quality['average'].view()
# service.view()
# tip.view()

# rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
# rule2 = ctrl.Rule(service['average'], tip['medium'])
# rule3 = ctrl.Rule(service['good'] | quality['good'], tip['high'])

# rule1.view()

# tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
# tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

# tipping.input['quality'] = 6.5
# tipping.input['service'] = 9.8

# tipping.compute()

# print(tipping.output['tip'])
# tip.view(sim=tipping)

# Practical 10a : Implementation of Simple genetic algorithm
# import random

# POPULATION_SIZE = 100

# GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP QRSTUVWXYZ
# 1234567890, .-;:_!"#%&/()=?@${[]}'''

# TARGET = "Mithilesh Chauhan"

# class Individual(object):
#     '''
#     Class representing an individual in the population
#     '''
#     def __init__(self, chromosome):
#         self.chromosome = chromosome
#         self.fitness = self.cal_fitness()

#     @classmethod
#     def mutated_genes(cls):
#         '''
#         Create random genes for mutation
#         '''
#         global GENES
#         gene = random.choice(GENES)
#         return gene

#     @classmethod
#     def create_gnome(cls):
#         '''
#         Create chromosome or string of genes
#         '''
#         global TARGET
#         gnome_len = len(TARGET)
#         return [cls.mutated_genes() for _ in range(gnome_len)]

#     def mate(self, par2):
#         '''
#         Perform mating and produce new offspring
#         '''
#         child_chromosome = []
#         for gp1, gp2 in zip(self.chromosome, par2.chromosome):
#             prob = random.random()
            
#             if prob < 0.45:
#                 child_chromosome.append(gp1)
            
#             elif prob < 0.90:
#                 child_chromosome.append(gp2)
            
#             else:
#                 child_chromosome.append(self.mutated_genes())

#         return Individual(child_chromosome)

#     def cal_fitness(self):
#         '''
#         Calculate fitness score, which is the number of characters in the string that differ from the target string.
#         '''
#         global TARGET
#         fitness = 0
#         for gs, gt in zip(self.chromosome, TARGET):
#             if gs != gt:
#                 fitness += 1
#         return fitness


# def main():
#     global POPULATION_SIZE

#     generation = 1

#     found = False
#     population = []

#     for _ in range(POPULATION_SIZE):
#         gnome = Individual.create_gnome()
#         population.append(Individual(gnome))

#     while not found:
#         population = sorted(population, key=lambda x: x.fitness)

#         if population[0].fitness <= 0:
#             found = True
#             break

#         new_generation = []

#         s = int((10 * POPULATION_SIZE) / 100)
#         new_generation.extend(population[:s])

#         s = int((90 * POPULATION_SIZE) / 100)
#         for _ in range(s):
#             parent1 = random.choice(population[:50])
#             parent2 = random.choice(population[:50])
#             child = parent1.mate(parent2)
#             new_generation.append(child)

#         population = new_generation

#         print("Generation: {}\tString: {}\tFitness: {}".format(generation, "".join(population[0].chromosome), population[0].fitness))
#         generation += 1

#     print("Generation: {}\tString: {}\tFitness: {}".format(generation, "".join(population[0].chromosome), population[0].fitness))

# if __name__ == '__main__':
#     main()

# Practical 10b : Create two classes: City and Fitness using Genetic algorithm

# import math
# import random

# class City:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def distance_to(self, other_city):
#         return math.sqrt((self.x - other_city.x)**2 + (self.y - other_city.y)**2)

#     def __repr__(self):
#         return f"City({self.x}, {self.y})"


# class Fitness:
#     def __init__(self, route):
#         self.route = route
#         self.distance = 0
#         self.fitness_score = 0.0

#     def calculate_distance(self):
#         total_distance = 0
#         for i in range(len(self.route) - 1):
#             total_distance += self.route[i].distance_to(self.route[i + 1])
#         total_distance += self.route[-1].distance_to(self.route[0])
#         self.distance = total_distance
#         return total_distance

#     def calculate_fitness(self):
        
#         self.fitness_score = 1 / float(self.calculate_distance())
#         return self.fitness_score
    
# city1 = City(0, 0)
# city2 = City(1, 2)
# city3 = City(4, 5)
# city4 = City(7, 8)

# route = [city1, city2, city3, city4]

# fitness = Fitness(route)

# total_distance = fitness.calculate_distance()
# fitness_score = fitness.calculate_fitness()

# print(f"Route: {route}")
# print(f"Total Distance: {total_distance}")
# print(f"Fitness Score: {fitness_score}")