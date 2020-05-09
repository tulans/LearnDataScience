import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Generate Random data set
observations = 1000
#Generate random inputs using/create a 2 variable linear model
xs = np.random.uniform(low=-10, high=10, size=(observations,1))
zs = np.random.uniform(-10,10, (observations, 1))
inputs = np.column_stack((xs, zs))
print(inputs.shape)


#Lets create the target in a linear model friendly way
noise = np.random.uniform(-1, 1, (observations, 1))
targets = 2*xs - 3*zs + 5 + noise
print(targets.shape)


#Plot the target
targets = targets.reshape(observations,)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, zs, targets)
ax.set_xlabel('xs')
ax.set_ylabel('zs')
ax.set_zlabel('Targets')
ax.view_init(azim=100)
plt.show()
targets = targets.reshape(observations, 1)


#init variables
init_range = 0.1
weights = np.random.uniform(-init_range, init_range, size=(2, 1))
biases = np.random.uniform(-init_range, init_range, size=1)
print(weights)
print(biases)

#set learning rate:
learning_rate = 0.02

for i in range(100):
    outputs = np.dot(inputs, weights) + biases
    deltas = outputs - targets
    loss = np.sum(deltas ** 2) / 2 / observations
    print(loss)
    deltas_scale = deltas / observations
    weights = weights - learning_rate * np.dot(inputs.T, deltas_scale)
    biases = biases - learning_rate * np.sum(deltas_scale)
