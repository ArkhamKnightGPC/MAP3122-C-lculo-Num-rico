#
#Gabriel Pereira de Carvalho
#NUSP:11257668
#

#Exercicio 2: Modelo presa-predador

import numpy as np
import math as mt
import matplotlib.pyplot as plt

#dados do problema
lam = 2.0/3 #lambda
alf = 4.0/3 #alfa
bet = 1 #beta
gam = 1 #gama
xo = 1.5 #valor inicial x(t): Populacao de coelhos
yo = 1.5 #valor inicial y(t): Populacao de raposas
to = 0 #tempo inicial
tf = 10 #tempo final

def EulerExplicito(n):#retorna vetor com aproximacoes obtidas
    
    h = (tf - to)/n #passo
    t = np.empty(n+1)    
    t[0] = to
    
    solucao = np.empty([2, n+1])
    solucao[0][0] = xo #solucao[0] armazena solucao numerica de x(t)
    solucao[1][0] = yo #solucao[1] armazena solucao numerica de y(t)
    
    for i in range(0, n):
        #recorrencia
        t[i+1] = t[i] + h
        solucao[0][i+1] = solucao[0][i] + h*(lam*solucao[0][i] - alf*solucao[0][i]*solucao[1][i])
        solucao[1][i+1] = solucao[1][i] + h*(bet*solucao[0][i]*solucao[1][i] - gam*solucao[1][i])
    
    return solucao

def f1(t, x, y): #f = [f1, f2]
   return lam*x - alf*x*y

def f1delx(t, x, y): #derivada parcial de f1 com respeito a x
    return lam - alf*y

def f1dely(t, x, y): #derivada parcial de f1 com respeito a y
    return -alf*x

def f2(t, x, y): #f = [f1, f2]
    return bet*x*y - gam*y

def f2delx(t, x, y): #derivada parcial de f2 com respeito a x
    return bet*y

def f2dely(t, x, y): #derivada parcial de f2 com respeito a y
    return bet*x - gam

def f(t, u): #f = [f1, f2]
    return np.array([f1(t, u[0], u[1]), f2(t, u[0], u[1])])

def g(t, u, u_anterior): # g = u[k+1] - u[k] - h*f
    return u - u_anterior - h*f(t, u)

def jacobianoG(t, x, y): #jacobiano da funcao g
    return np.array([[1 - h*f1delx(t,x,y), -h*f1dely(t,x,y)],
                     [-h*f2delx(t,x,y), 1 - h*f2dely(t,x,y)]])

def inverteMatriz(mat): #inverte matrix 2x2
    #adjunta dividida pelo determinante
    determinante = mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0]
    cof = np.array([[mat[1][1], -mat[1][0]],
                    [-mat[0][1], mat[0][0]]])
    adj = np.transpose(cof)
    return adj/determinante


def MetodoNewton(t, pz, lim, tol, u_anterior):
    i = 1
    while(i <= lim):
        p = pz - np.matmul(g(t, pz, u_anterior),inverteMatriz(jacobianoG(t, pz[0], pz[1])))
        if(max(mt.fabs(p[0] - pz[0]), mt.fabs(p[1] - pz[1])) < tol):
            return p
        i = i+1
        pz = p
    return "Metodo de Newton falhou"

def EulerImplicito(n):
    
    h = (tf - to)/n #passo
    t = np.empty(n+1)
    t[0] = to
    
    solucao = np.empty([2, n + 1])
    solucao[0][0] = xo
    solucao[1][0] = yo
    u_anterior = np.array([solucao[0][0], solucao[1][0]]) #uk
    
    for i in range(0, n):
        t[i+1] = t[i] + h
        #usamos Euler Explicito para aproximacao inicial
        aprox_inicial = u_anterior + h*f(t[i], u_anterior)
        #a partir da aproximacao inicial, calculamos raiz com metodo de Newton
        raiz = MetodoNewton(t[i+1], aprox_inicial, 100, 10**-8, u_anterior)
        solucao[0][i+1] = raiz[0]
        solucao[1][i+1] = raiz[1]
        u_anterior = raiz #atualizamos uk para proxima iteracao
        
    return solucao

    
def RungeKutta(n):#retorna vetor com aproximacoes obtidas
     h = (tf - to)/n #calculo do passo
     
     t = np.empty(n+1)
     t[0] = to #ponto inicial
     
     solucao = np.empty([2, n+1])
     solucao[0][0] = xo #valor inicial
     solucao[1][0] = yo
     
     for i in range(0, n):
         u = np.array([solucao[0][i], solucao[1][i]])
         #calculo dos parametros k1,k2,k3,k4
         k1 = h*f(t[i], u)
         k2 = h*f(t[i] + h/2, u + k1/2)
         k3 = h*f(t[i] + h/2, u + k2/2)
         k4 = h*f(t[i] + h, u + k3)
         #recorrencia
         t[i+1] = t[i] + h
         solucao[0][i+1] = u[0] + (k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
         solucao[1][i+1] = u[1] + (k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
         
     return solucao
    
#MAIN
    
#Primeiro vamos fazer os plots EXPLICITOS

h = (tf - to)/5000 #passo
t = np.empty(5001)
for i in range(0, 5001):
    t[i] = to + i*h #tempo vai ser usado no eixo x

solucaoExplicita = EulerExplicito(5000)

#retrato de fase
plt.plot(solucaoExplicita[0], solucaoExplicita[1])
plt.title("Euler Explicito n=5000")
plt.show()

#populacoes
plt.plot(t, solucaoExplicita[0], label = "Coelhos", color="blue")
plt.plot(t, solucaoExplicita[1], label = "Raposas", color="red")
plt.title("Euler Explicito n=5000")
plt.legend()
plt.show()

#agora vamos fazer os plots IMPLICITOS

h = (tf - to)/500 #passo
t = np.empty(501)
for i in range(0, 501):
    t[i] = to + i*h #tempo vai ser usado no eixo x

solucaoImplicita = EulerImplicito(500)

#retrato de fase
plt.plot(solucaoImplicita[0], solucaoImplicita[1])
plt.title("Euler Implicito n=500")
plt.show()

#populacoes
plt.plot(t, solucaoImplicita[0], label = "Coelhos", color="blue")
plt.plot(t, solucaoImplicita[1], label = "Raposas", color="red")
plt.legend()
plt.title("Euler Implicito n=500")
plt.show()

#agora vamos plotar os erros
n = np.array([250, 500, 1000, 2000, 4000])
for i in range(0, 5):
    #calculamos solucao para cada n
    solucaoExplicita = EulerExplicito(n[i])
    solucaoImplicita = EulerImplicito(n[i])
    erro = solucaoImplicita - solucaoExplicita
    
    h = (tf - to)/n[i] #passo
    t = np.empty(n[i] + 1)
    for j in range(0, n[i] + 1):
        t[j] = to + j*h #tempo vai ser usado no eixo x
    plt.plot(t, erro[0], label="Coelhos", color="blue")
    plt.plot(t, erro[1], label="Raposas", color="red")
    titulo = "Erro n=" +  str(n[i]) #montamos titulo do grafico
    plt.title(titulo)
    plt.legend()
    plt.show()

#agora vamos fazer os plots com RUNGE KUTTA
h = (tf - to)/500 #passo
t = np.empty(501)
for i in range(0, 501):
    t[i] = to + i*h #tempo vai ser usado no eixo x
    
solucaoRungeKutta = RungeKutta(500)
    
#retrato de fase
plt.plot(solucaoRungeKutta[0], solucaoRungeKutta[1])
plt.title("Runge Kutta n=500")
plt.show()

#populacoes
plt.plot(t, solucaoRungeKutta[0], label = "Coelhos", color="blue")
plt.plot(t, solucaoRungeKutta[1], label = "Raposas", color="red")
plt.legend()
plt.title("Runge Kutta n=500")
plt.show()

#vamos plotar todos os retratos de fase juntos
plt.plot(solucaoExplicita[0], solucaoExplicita[1], label = "Euler Explicito", color="blue")
plt.plot(solucaoImplicita[0], solucaoImplicita[1], label = "Euler Implicito", color="red")
plt.plot(solucaoRungeKutta[0], solucaoRungeKutta[1], label = "Runge Kutta", color="green")
plt.legend()
plt.title("Comparando retratos de fase")
plt.show()