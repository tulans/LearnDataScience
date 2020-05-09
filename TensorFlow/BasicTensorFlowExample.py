import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#Generate Random data set
observations = 1000
#Generate random inputs using/create a 2 variable linear model
xs = np.random.uniform(low=-10, high=10, size=(observations,1))
zs = np.random.uniform(-10, 10, (observations, 1))
gen_inputs = np.column_stack((xs, zs))
print(gen_inputs.shape)


#Lets create the target in a linear model friendly way
noise = np.random.uniform(-1, 1, (observations, 1))
gen_targets = 2*xs - 3*zs + 5 + noise

np.savez('TF_intro', inputs=gen_inputs, targets=gen_targets)

training_data = np.load('TF_intro.npz')

input_size = 2
output_size = 1

#build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(output_size)
])


model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(training_data['inputs'], training_data['targets'], epochs=100, verbose=2)

print(model.layers[0].get_weights())

weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]

print('weights' + str(weights))
print('bias ' + str(bias))

#model.predict_on_batch(training_data['inputs']).round(1)
#training_data['targets'].rount(1)

plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))

plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()












