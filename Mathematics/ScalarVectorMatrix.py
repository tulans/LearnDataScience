import numpy as np

#Scalars
s = 5


#Vectors
v = np.array([5, -2, 4])


#Matrics
m = np.array([[5, 12, 6], [-3, 0, 14]])
print(m)

#Data Types
type(s)

type(v)

type(m)


s_array = np.array(5)
type(s_array)



#Data Shapes
m.shape

v.shape



#creating a column vector
v.reshape(1, 3)
print(v)

v.reshape(3, 1)
print(v)
