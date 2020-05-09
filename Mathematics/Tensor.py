import numpy as np

m1 = np.array([[5, 12, 6], [-3, 0, 14]])
print(m1)

m2 = np.array([[9, 8, 7], [1, 3, -5]])
print(m2)


t = np.array([m1,m2])
print(t)


print(m1+m2)


#Transpose of a matrix
print(m1.T)


#dot product
print('Dot product')
x = np.array([1,2,3])
y = np.array([3,2,3])
print(np.dot(x,y))


