#
#Gabriel Pereira de Carvalho
#NUSP:11257668
#

#Exercicio 3: Modelo duas presas-um predador

import numpy as np
import math as mt
import matplotlib.pyplot as plt

#dados do problema
B = np.array([1.0, 1.0, -1.0])
to = np.array([0, 0, 0, 0, 0, 0]) #tempo inicial
tf = np.array([100.0, 100.0, 500.0, 500.0, 2000.0, 2000.0]) #tempo final
alfas = np.array([0.001, 0.002, 0.0033, 0.0036, 0.005, 0.0055])
n = 5000 #parametro de discretizacao
xo = 500.0 #populacao inicial de coelhos
yo = 500.0 #populacao inicial de lebres
zo = 10.0 #populacao inicial de raposas

def constroeA(alfa): #dado parametro alfa, constroe matriz A
    A = np.array([[0.001, 0.001, 0.015],
                  [0.0015, 0.001, 0.001],
                  [-alfa, -0.0005, 0.0]])
    return A

def f(t, u, A):
    return np.array([ u[0]*(B[0] - A[0][0]*u[0] - A[0][1]*u[1] - A[0][2]*u[2]),
                      u[1]*(B[1] - A[1][0]*u[0] - A[1][1]*u[1] - A[1][2]*u[2]),
                      u[2]*(B[2] - A[2][0]*u[0] - A[2][1]*u[1])])

def EulerExplicito(alfa, to, tf):#retorna vetor com aproximacoes obtidas
    
    h = (tf - to)/n #passo
    t = np.empty(n+1)    
    t[0] = to
    
    solucao = np.empty([3, n+1])
    solucao[0][0] = xo #solucao[0] armazena solucao numerica de x(t)
    solucao[1][0] = yo #solucao[1] armazena solucao numerica de y(t)
    solucao[2][0] = zo #solucao[2] armazena solucao numerica de z(t)
    
    A = constroeA(alfa)
    
    for i in range(0, n):
        anterior = np.array([solucao[0][i], solucao[1][i], solucao[2][i]]) #vetor u[k]
        novo = anterior + h*f(t[i], anterior, A) #vetor u[k+1]
        #recorrencia
        t[i+1] = t[i] + h
        solucao[0][i+1] = novo[0]
        solucao[1][i+1] = novo[1]
        solucao[2][i+1] = novo[2]
    
    return solucao

def RungeKutta(alfa, to, tf):#retorna vetor com aproximacoes obtidas
     h = (tf - to)/n #calculo do passo
     
     t = np.empty(n+1)
     t[0] = to #ponto inicial
     
     solucao = np.empty([3, n+1])
     solucao[0][0] = xo #solucao[0] armazena solucao numerica de x(t)
     solucao[1][0] = yo #solucao[1] armazena solucao numerica de y(t)
     solucao[2][0] = zo #solucao[2] armazena solucao numerica de z(t)
     
     A = constroeA(alfa)
     
     for i in range(0, n):
         u = np.array([solucao[0][i], solucao[1][i], solucao[2][i]])
         #calculo dos parametros k1,k2,k3,k4
         k1 = h*f(t[i], u, A)
         k2 = h*f(t[i] + h/2, u + k1/2, A)
         k3 = h*f(t[i] + h/2, u + k2/2, A)
         k4 = h*f(t[i] + h, u + k3, A)
         #recorrencia
         t[i+1] = t[i] + h
         solucao[0][i+1] = u[0] + (k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
         solucao[1][i+1] = u[1] + (k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
         solucao[2][i+1] = u[2] + (k1[2]+2*k2[2]+2*k3[2]+k4[2])/6
         
     return solucao

#vamos fazer os plots para cada alfa
for i in range(0, 6):
    
    #vamos construir vetor de tempos para eixo x dos graficos
    h = (tf[i] - to[i])/n
    t = np.empty(n+1)
    for j in range(0, n+1):
        t[j] = to[i] + j*h
        
    #Primeiro, vamos fazer o plot de cada populacao para observar a solucao
    solucaoEuler = EulerExplicito(alfas[i], to[i], tf[i])
    plt.plot(t, solucaoEuler[0], label="Coelhos", color="blue")
    plt.plot(t, solucaoEuler[1], label="Lebres", color="black")
    plt.plot(t, solucaoEuler[2], label="Raposas", color="red")
    plt.legend()
    titulo = "Euler Explicito alfa="+str(alfas[i])
    plt.title(titulo)
    plt.show()
    
    solucaoRungeKutta = RungeKutta(alfas[i], to[i], tf[i])
    plt.plot(t, solucaoRungeKutta[0], label="Coelhos", color="blue")
    plt.plot(t, solucaoRungeKutta[1], label="Lebres", color="black")
    plt.plot(t, solucaoRungeKutta[2], label="Raposas", color="red")
    plt.legend()
    titulo = "Runge Kutta alfa="+str(alfas[i])
    plt.title(titulo)
    plt.show()
    
    #agora vamos fazer os retratos de fase 3d
    ax = plt.axes(projection = '3d')
    ax.plot(solucaoEuler[0], solucaoEuler[1], solucaoEuler[2])
    ax.set_xlabel("Coelhos")
    ax.set_ylabel("Lebres")
    ax.set_zlabel("Raposas")
    titulo = "Euler Explicito alfa="+str(alfas[i])
    plt.title(titulo)
    plt.show()

    ax = plt.axes(projection = '3d')
    ax.plot(solucaoRungeKutta[0], solucaoRungeKutta[1], solucaoRungeKutta[2])
    ax.set_xlabel("Coelhos")
    ax.set_ylabel("Lebres")
    ax.set_zlabel("Raposas")
    titulo = "Runge Kutta alfa="+str(alfas[i])
    plt.title(titulo)
    plt.show()
    
    #agora vamos fazer os retratos de fase 2d
    plt.plot(solucaoEuler[0], solucaoEuler[1])
    plt.xlabel("Coelhos")
    plt.ylabel("Lebres")
    titulo = "Euler Explicito alfa="+str(alfas[i])
    plt.title(titulo)
    plt.show()
    
    plt.plot(solucaoEuler[0], solucaoEuler[2])
    plt.xlabel("Coelhos")
    plt.ylabel("Raposas")
    titulo = "Euler Explicito alfa="+str(alfas[i])
    plt.title(titulo)
    plt.show()
    
    plt.plot(solucaoEuler[1], solucaoEuler[2])
    plt.xlabel("Lebres")
    plt.ylabel("Raposas")
    titulo = "Euler Explicito alfa="+str(alfas[i])
    plt.title(titulo)
    plt.show()
    
    plt.plot(solucaoRungeKutta[0], solucaoRungeKutta[1])
    plt.xlabel("Coelhos")
    plt.ylabel("Lebres")
    titulo = "Runge Kutta alfa="+str(alfas[i])
    plt.title(titulo)
    plt.show()
    
    plt.plot(solucaoRungeKutta[0], solucaoRungeKutta[2])
    plt.xlabel("Coelhos")
    plt.ylabel("Raposas")
    titulo = "Runge Kutta alfa="+str(alfas[i])
    plt.title(titulo)
    plt.show()
    
    plt.plot(solucaoRungeKutta[1], solucaoRungeKutta[2])
    plt.xlabel("Lebres")
    plt.ylabel("Raposas")
    titulo = "Runge Kutta alfa="+str(alfas[i])
    plt.title(titulo)
    plt.show()
    
#Agora vamos fazer o teste de sensibilidade
xo = 37 #valor inicial de coelhos
yo = 75 #valor inicial de lebres
zo = 137 #valor inicial de raposas
h = (400.0 - 0.0)/n
t = np.empty(n+1)
for j in range(0, n+1):
    t[j] = j*h
solucaoEuler75 = EulerExplicito(0.005, 0, 400)
solucaoRungeKutta75 = RungeKutta(0.005, 0, 400)

print("Valores finais Euler y(0) = 75")
print(solucaoEuler75[0][n], " coelhos")
print(solucaoEuler75[1][n], " lebres")
print(solucaoEuler75[2][n], " raposas")
#alem de imprimir os valores, vamos plotar as populacoes
plt.plot(t, solucaoEuler75[0], label="Coelhos", color="blue")
plt.plot(t, solucaoEuler75[1], label="Lebres", color="black")
plt.plot(t, solucaoEuler75[2], label="Raposas", color="red")
plt.legend()
plt.title("Euler y(0)=75")
plt.show()

print("Valores finais Runge Kutta y(0) = 75")
print(solucaoRungeKutta75[0][n], " coelhos")
print(solucaoRungeKutta75[1][n], " lebres")
print(solucaoRungeKutta75[2][n], " raposas")
plt.plot(t, solucaoRungeKutta75[0], label="Coelhos", color="blue")
plt.plot(t, solucaoRungeKutta75[1], label="Lebres", color="black")
plt.plot(t, solucaoRungeKutta75[2], label="Raposas", color="red")
plt.legend()
plt.title("Runge Kutta y(0)=75")
plt.show()

yo = 74 #alterando populacao inicial de lebres para 74
solucaoEuler74 = EulerExplicito(0.005, 0, 400)
solucaoRungeKutta74 = RungeKutta(0.005, 0, 400)
print("Valores finais Euler y(0) = 74")
print(solucaoEuler74[0][n], " coelhos")
print(solucaoEuler74[1][n], " lebres")
print(solucaoEuler74[2][n], " raposas")
plt.plot(t, solucaoEuler74[0], label="Coelhos", color="blue")
plt.plot(t, solucaoEuler74[1], label="Lebres", color="black")
plt.plot(t, solucaoEuler74[2], label="Raposas", color="red")
plt.legend()
plt.title("Euler y(0)=74")
plt.show()

print("Valores finais Runge Kutta y(0) = 74")
print(solucaoRungeKutta74[0][n], " coelhos")
print(solucaoRungeKutta74[1][n], " lebres")
print(solucaoRungeKutta74[2][n], " raposas")
plt.plot(t, solucaoRungeKutta74[0], label="Coelhos", color="blue")
plt.plot(t, solucaoRungeKutta74[1], label="Lebres", color="black")
plt.plot(t, solucaoRungeKutta74[2], label="Raposas", color="red")
plt.legend()
plt.title("Runge Kutta y(0)=74")
plt.show()