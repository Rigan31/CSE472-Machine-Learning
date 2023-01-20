import numpy as np

# taking the dimension of the matrix as input

dimension = int(input("Enter the dimension of the matrix: "))

#randomly generating integer values for a symmetric matrix
matrix = np.random.randint(-100, 100, (dimension, dimension))

# keep checking until the matrix is invertible
while(np.linalg.det(matrix) == 0):
    matrix = np.random.randint(-100, 100, (dimension, dimension))

print("The transformation matrix is: ")
print(matrix)



# finding the eigen values and eigen vectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print("The eigenvalues of the matrix are: ", eigenvalues)
print("The eigenvectors of the matrix are: ", eigenvectors)

#first find the inverse of the eigenvectors
#then find the diagonal matrix using eigen values
#then multiply the inverse of the eigenvectors with the diagonal matrix
#then multiply the product with the eigenvectors

inverse = np.linalg.inv(eigenvectors)
diagonal = np.diag(eigenvalues)
product = np.dot(diagonal, inverse)
result = np.dot(eigenvectors, product)
print("The reconstruction matrix is: ")
print(result)

# use np.allclose to compare the result with the original matrix
print(np.allclose(result, matrix))







