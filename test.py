import numpy as np
import scipy.io as sio
import scipy.optimize as sc
import heapq
import math
from PIL import Image

#-------funciones importantes-------
#crear el nuevo array del label
def createY(x):
    res = np.zeros(10)
    res[x] = 1.0
    return res

def sigmoid(z):
    return 1/(1+np.power(np.e,-z))

#derivada de la sigmoide
def sigmoidGradient(z):
    #f'(s) = f(s)*(1-f(s)) por regla de la cadena
    return np.multiply(sigmoid(z),(1-sigmoid(z)));

#generar numeros random para las thetas
def randomInitialization(i, epsilon=0.12):
    return np.random.rand(i,1)*2*epsilon-epsilon

#yvalue = train de y
#p = data
#convertir y a array
#backpropagation
def feedForward(theta, x):
    Theta1 = theta[0]
    Theta2 = theta[1]

    a1 = x
    z2 = Theta1 @ a1
    a2 = sigmoid(z2)
    z3 = Theta2 @ a2
    a3 = sigmoid(z3)

    return a3


def backProp(theta, data, num_input, num_hidden, num_labels, l=0.2):
    Theta1 = theta[0]
    Theta2 = theta[1]

    m = len(data)
    delta1 = np.zeros(theta[0].shape)
    delta2 = np.zeros(theta[1].shape)

    for t in range(m):
        # Feed Forward
        a1 = data[t][0]
        z2 = Theta1 @ a1
        a2 = sigmoid(z2)
        z3 = Theta2 @ a2
        a3 = sigmoid(z3)

        error3 = a3 - createY(data[t][1])
        error2 = Theta2.T @ error3 * a2 * (np.ones(len(a2)) - a2)

        delta1 += error2.reshape(num_hidden, 1) @ a1.reshape(1, num_input)
        delta2 += error3.reshape(num_labels, 1) @ a2.reshape(1, num_hidden)

    # print(Theta1.shape)
    return np.array([delta1, delta2])

#funcion de costos
# def J(theta, num_input, num_hidden, num_lables, X, yvalue, l=0.2):
#
#     m = len(X)
#     X = np.append(np.ones(shape=(X.shape[0],1)),X,axis=1)
#     J = 0
#     for i in range(m):
#         x = np.matrix(X[i])
#         w = np.zeros((10,1))
#         w[int(yvalue[i])-1] = 1
#         hx = sigmoid(Theta2*np.append([[1]], sigmoid(Theta1*x.transpose()), axis=0))
#         J += sum(-w.transpose()*np.log(hx)-(1-w).transpose()*np.log(1-hx))
#
#     J = J/m
#     J += (l/(2*m))*(sum(sum(Theta1[:,1:]**2)) + sum(sum(Theta2[:,1:]**2)))
#
#     return float(J)

#inicializar valores
counter = 0
num_hidden = 125
num_input = 784
num_labels = 10

#load data
circle = np.load('data/new/full_numpy_bitmap_circle.npy')[:4000]
face = np.load('data/new/full_numpy_bitmap_face.npy')[:4000]
house = np.load('data/new/full_numpy_bitmap_house.npy')[:4000]
square = np.load('data/new/full_numpy_bitmap_square.npy')[:4000]
tree = np.load('data/new/full_numpy_bitmap_tree.npy')[:4000]
triangle = np.load('data/new/full_numpy_bitmap_triangle.npy')[:4000]
mickeymouse = np.load('data/new/mickeymouse.npy')[:4000]
questionmark = np.load('data/new/questionmark.npy')[:4000]
sadface = np.load('data/new/sadface.npy')[:4000]
egg = np.load('data/new/egg.npy')[:4000]

# print(len(mickeymouse))
# print(len(questionmark))
# print(len(sadface))
# print(len(egg))
#union datasets
dataset = np.concatenate((circle, face, house, square, tree, triangle, mickeymouse, questionmark, sadface, egg))
dataset[dataset > 1] = 1
cant_img = 3000

#vector con los resultados que puede recibir
results = np.concatenate((
        np.repeat(0, len(circle)),
        np.repeat(1, len(face)),
        np.repeat(2, len(house)),
        np.repeat(3, len(square)),
        np.repeat(4, len(tree)),
        np.repeat(5, len(triangle)),
        np.repeat(6, len(mickeymouse)),
        np.repeat(7, len(questionmark)),
        np.repeat(8, len(sadface)),
        np.repeat(9, len(egg))
        ))

#se queda con int
data = list(map(lambda x, y: (x, y), dataset, results))
np.random.shuffle(data)
print(len(results))

threshold = math.trunc(len(data) * 0.8)

training_data = data[:threshold]
test_data = data[threshold:]

## Thetas
def random_theta(num):
    n = np.random.uniform(0, 0.1, num)
    return n

#matriz de theta, numeros random - iniciales
Theta1 = np.array([random_theta(num_input) for j in range(num_hidden)])
Theta2 = np.array([random_theta(num_hidden) for j in range(num_labels)])
Theta = [Theta1, Theta2]


for i in range(0, len(training_data), 10):
    batch = training_data[i:i+10]
    delta = backProp(Theta, batch, num_input, num_hidden, num_labels, l=0.2)
    #0.2 - para no tomar todo el error del bp
    #Este valor puede ser 0.05 y 0.3
    Theta = Theta - 0.2 * delta
    print(i)

# Theta = np.load('Theta.npy')

cont = 0
for i in range(len(test_data)):
    res = feedForward(Theta, test_data[i][0])
    if (test_data[i][1] == np.argmax(res)):
        cont +=1

print('Prueba: {0}/{1}'.format(cont, len(data)-threshold))

file = Image.open('test_1.bmp')
input = np.array(file)
inputArr = [0 if input[i][j][0] == 255 else 1 for i in range(28) for j in range(28)]
res = feedForward(Theta, inputArr)
res_label = ['circle', 'face', 'house', 'square', 'tree', 'triangle', 'mickeymouse', 'questionmark', 'sadface', 'egg']
print('Estoy {:2f} seguro que es un {}.'.format(max(res), res_label[np.argmax(res)]))
print(res)

# np.save('Theta', Theta)



#pasos para entrenar modelo
#1. ciclo para recorrer data
# 1.a backProp
# 1.b pesos-delta (backProp)

#2 ver acc
# np.save('Theta', theta)
