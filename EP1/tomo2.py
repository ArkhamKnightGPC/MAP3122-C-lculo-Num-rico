#
#Gabriel Pereira de Carvalho
#NUSP:11257668
#

import numpy as np
import matplotlib as mp

EPS = 10**-9

def ElimGauss(A, b): #resolve sistema de m equacoes e m variaveis
    
    #dimensao da matriz quadrada A
    m = np.size(A, 0)
    #criar vetor coluna das solucoes
    x = np.empty(m)
    
    #Vamos implementar com Condensacao Pivotal
    for coluna in range(0, m):
        
        #selecionando pivo
        sel = coluna
        for i in range(coluna + 1, m):
            if(abs(A[i][coluna]) > abs(A[sel][coluna])):
                sel = i
                
        #vamos trocar as linhas
        A[[coluna, sel]] = A[[sel, coluna]]
        b[[coluna,sel]] = b[[sel, coluna]]
        
        #se pivo eh nulo, sistema nao tem solucao
        if(abs(A[coluna][coluna]) < EPS):
            return False
        
        #hora de eliminar
        for linha in range(coluna + 1, m):
            multiplicador = A[linha][coluna]/A[coluna][coluna]
            A[linha] -= multiplicador*A[coluna]
            b[linha] -= multiplicador*b[coluna]
                
    linha = m - 1 #retrosubstituicao: comecamos na ultima linha e subimos
    while linha > -1:
        x[linha] = b[linha]
        for anterior in range(linha+1, m):
            x[linha] -= A[linha][anterior]*x[anterior]
        x[linha] /= A[linha][linha]
        linha -= 1
        
    return x

    
def diag_decr(i, n):
    if(i%n == 0): #atingimos borda superior
        return n - 1 + i/n
    elif i < n :#atingimos borda lateral
        return n-i-1
    else:
        return diag_decr(i - n - 1, n)#chamamos elemento anterior da progressão

def diag_cres(i, n):
    if i<n :#atingimos borda lateral
        return i
    elif i%n==n-1:#atingimos borda inferior
        return n-1 + i/n
    else:
        return diag_cres(i - n + 1, n)#chamamos elemento anterior da progressão

def constroeA(n):#constroe matriz A
    n = int(n)
    A = np.zeros((6*n-2,n*n))
    #vamos pensar na contribuicao de cada f_i
    for i in range(0, n*n):
        
        #vamos determinar qual linha f_i contribui
        linha =  int(i%n)
        A[linha][i] += 1
        
        #vamos determinar qual coluna f_i contribui
        coluna = int(i/n)
        A[n + coluna][i] += 1
        
        #vamos determinar qual diagonal descrescente f_i contribui
        A[int(2*n + diag_decr(i,n))][i] += 1
        
        #vamos determinar qual diagonal crescente f_i contribui
        A[int(4*n-1+diag_cres(i,n))][i] += 1
        
    return A

    
def exercicio2(p, n, delta):
    
    A = constroeA(n)
    At = np.copy(A)
    At = np.transpose(At)
    
    #vamos obter sistema normal a partir dos dados
    A_normal = np.add(np.matmul(At, A), np.multiply(delta, np.identity(n*n)))
    p_normal = np.matmul(At, p)
    
    #vamos calcular os determinates pedidos aqui
    det = np.linalg.det(A_normal)
    #print(n, delta, det)
    
    
    fsol = ElimGauss(A_normal, p_normal)
    
    #foi encontrado pivo nulo durante eliminacao
    if(type(fsol) == 'bool'):
        print("SISTEMA IMPOSSIVEL")    
        
    #transformar vetor coluna em matriz no formato dado
    fsol = np.reshape(fsol, (n,n))
    fsol = np.transpose(fsol)
    
    #hora de gerar imagem da solucao
    #print(fsol)
    mp.pyplot.matshow(fsol)
    mp.pyplot.show()

#vamos importar o arquivo p2.npy
print("Por digite o caminho do arquivo p2.npy")
caminho_p = input()
p = np.load(caminho_p)

#obter parametro n
n = round((np.size(p, 0) + 2)/6)

#vamos gerar solucao do problema para cada delta
delta = np.array([10**-3, 10**-2, 10**-1])
for i in range(0, np.size(delta,0)):
    exercicio2(p, n, delta[i])
    
#Agora vamos exibir a imagem original
print("Por favor digite o caminho do arquivo im.png")
caminho_f = input()
f = mp.pyplot.imread(caminho_f)
#print(f)
mp.pyplot.matshow(f)
mp.pyplot.show()