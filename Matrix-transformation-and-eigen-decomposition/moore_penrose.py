import numpy as np

m = int(input("Enter the value of m: "))
n = int(input("Enter the value of n: "))


#produce a random n*m matrix
matrix = np.random.randint(-100, 100, (m,n))
print("The transformation matrix is: ")
print(matrix)


#single value decomposition
U, D, Vh = np.linalg.svd(matrix)
print("Using numpy library function")
print("The U matrix is: ")
print(U)
print("The D matrix is: ")
print(D)
print("The V matrix is: ")
print(Vh)
print(U.shape + D.shape + Vh.shape)



#using numpy library function
moore_penrose = np.linalg.pinv(matrix)
print("The moore penrose inverse matrix using numpy library function is: ")
print(moore_penrose)

#using the formula
reciprocal = np.reciprocal(D)
k = min(m, n)
#new numpy matrix of size m*n with zeros
dPLus = np.zeros((n,m))
for i in range(k):
    dPLus[i][i] = reciprocal[i]

pseudo_inverse = np.dot(dPLus, U.T)
pseudo_inverse = np.dot(Vh.T, pseudo_inverse)

print("The moore penrose inverse matrix using formula is: ")
print(pseudo_inverse)

print(np.allclose(pseudo_inverse, moore_penrose))