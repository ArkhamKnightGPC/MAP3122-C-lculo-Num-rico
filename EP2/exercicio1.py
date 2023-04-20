#
#Gabriel Pereira de Carvalho
#NUSP:11257668
#

#Exercicio 1: Testes

import numpy as np
import math as mt
import matplotlib.pyplot as plt


def ExercicioRungeKutta():

#dados do problema
    xo = np.array([1, 1, 1, -1]) #valor inicial
    A = np.array([[-2, -1, -1, -2],
                  [1, -2, 2, -1],
                  [-1, -2, -2, -1],
                  [2, -1, 1, -2]]) #matriz usada para calculo da funcao f
    n = np.array([20, 40, 80, 160, 320, 640]) #valores do parametro de discretizacao
    to = 0 #tempo inicial
    tf = 2 #tempo final
    
    def f(ti, xi):
        return np.matmul(A, xi)
    
    def RungeKutta(n):#retorna vetor com aproximacoes obtidas
        h = (tf - to)/n #calculo do passo
        
        t = np.empty(n+1)
        t[0] = to #ponto inicial
        
        solucao = np.empty([n+1, xo.size])
        solucao[0] = xo #valor inicial
        
        for i in range(0, n):
            #calculo dos parametros k1,k2,k3,k4
            k1 = h*f(t[i], solucao[i])
            k2 = h*f(t[i] + h/2, solucao[i] + k1/2)
            k3 = h*f(t[i] + h/2, solucao[i] + k2/2)
            k4 = h*f(t[i] + h, solucao[i] + k3)
            #recorrencia
            t[i+1] = t[i] + h
            solucao[i+1] = solucao[i] + (k1+2*k2+2*k3+k4)/6
            
        return solucao
        
    def SolucaoExata(n):
        h = (tf - to)/n #calculo do passo
        exata = np.empty([n+1, 4])
        for i in range(0, n+1): #usando formulas dadas
            exata[i][0] = mt.exp(-to - i*h)*mt.sin(to + i*h) + mt.exp(-3*to - 3*i*h)*mt.cos(3*to + 3*i*h)
            exata[i][1] = mt.exp(-to - i*h)*mt.cos(to + i*h) + mt.exp(-3*to - 3*i*h)*mt.sin(3*to + 3*i*h)
            exata[i][2] = -mt.exp(-to - i*h)*mt.sin(to + i*h) + mt.exp(-3*to - 3*i*h)*mt.cos(3*to + 3*i*h)
            exata[i][3] = -mt.exp(-to - i*h)*mt.cos(to + i*h) + mt.exp(-3*to - 3*i*h)*mt.sin(3*to + 3*i*h)
        return exata
    
    #comparacao das solucoes exata e aproximada para cada n dado
    erro_maximo = np.empty(n.size) #maximo erro para cada n
    R = np.empty(n.size - 1) #razao entre erros
    for i in range(0, 6):
        erro_maximo[i] = -1 # vai ser substituido apos uso de max
        
        aproximacao = RungeKutta(n[i])
        exata = SolucaoExata(n[i])
        
        h = (tf - to)/n[i]
        t = np.empty(n[i] + 1)
        erro = np.empty(n[i] + 1)
        
        for j in range(0, n[i]+1):
            erro[j] = -1
            t[j] = to + j*h
            for k in range(0, xo.size):
                erro[j] = max(erro[j], mt.fabs(exata[j][k] - aproximacao[j][k]))
            erro_maximo[i] = max(erro_maximo[i], erro[j])
            
        plt.plot(t, erro)
        plt.title("Erro n="+str(n[i]))
        plt.xlabel("t")
        plt.ylabel("Erro")
        plt.show()
        if i>0:
            R[i-1] = erro_maximo[i-1]/erro_maximo[i] #calculamos razao entre os erros (erro relativo)
    for i in range(0, 5):
        print("R[",i,"] = ",R[i])

def ExercicioEulerImplicito():
    
    #dados do problema
    xo = -8.79 #valor inicial
    to = 1.1 #tempo inicial
    tf = 3.0 #tempo final
    n = 5000 #parametro de discretizacao
    h = (tf - to)/n #passo
    t = np.empty(n + 1) #array para armazenar tempos
    
    def f(t, x, x_anterior):
        return x - x_anterior - h*(2*t + (x - t**2)**2)
    
    def fp(t, x, x_anterior): #derivada de f
        return 1 - h*2*(x - t**2)
    
    def MetodoNewton(t, pz, lim, tol, x_anterior):
        i = 1
        while(i <= lim):
            p = pz - f(t, pz, x_anterior)/fp(t, pz, x_anterior)
            if(mt.fabs(p - pz) < tol): #encontramos raiz
                return p
            i = i+1
            pz = p
        return "Metodo de Newton falhou"
    
    def EulerImplicito():
        
        solucao = np.empty(n + 1)
        solucao[0] = xo
        
        for i in range(0, n):
            #usamos Euler Explicito para aproximacao inicial
            aprox_inicial = solucao[i] + h*(2*t[i] + (solucao[i] - t[i]**2)**2)
            #a partir da aproximacao inicial, calculamos raiz com metodo de Newton
            solucao[i+1] = MetodoNewton(t[i+1], aprox_inicial, 7, 10**-8, solucao[i])
            
        return solucao
        
    def SolucaoExata():
        
        solucao = np.empty(n + 1)
        for i in range(0, n+1):
            solucao[i] = (to + i*h)**2 + 1/(1 - to - i*h)
        
        return solucao
    
    #calculo dos tempos (eixo x dos graficos)
    for i in range(0, n+1):
        t[i] = to + i*h
        
    solucaoExata = SolucaoExata()
    plt.plot(t, solucaoExata)
    plt.title("Solucao Exata")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.show()
    
    solucaoNumerica = EulerImplicito()
    plt.plot(t, solucaoNumerica)
    plt.title("Solucao Numerica")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.show()
    
    erro = np.empty(n + 1)
    for i in range(0, n + 1):
        erro[i] = mt.fabs(solucaoExata[i] - solucaoNumerica[i])
    plt.plot(t, erro)
    plt.title("Erro")
    plt.xlabel("t")
    plt.ylabel("Erro")
    plt.show()
    
#Escolher qual exercicio executar
print("Para executar Teste Runge Kutta 4, digite 1")
print("Para exercutar Teste Euler Implicito, digite 2")
decisao = input()
if decisao == "1":
    ExercicioRungeKutta()
else:
    ExercicioEulerImplicito()