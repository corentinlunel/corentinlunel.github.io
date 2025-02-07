from copy import copy,deepcopy
from random import randint,randrange
from matplotlib import pylab
from math import cos,sin,pi
from time import clock
import sys
sys.setrecursionlimit(100000)

#sont variables globales temps,parcoursi et predr
#si temps, les algorithmes affichent certaines étapes

odr = [1,4,6,7,8,9,10]        # autorisés à droite si brin
oga = [1,3,5,7,8,9,10]        # autorisés à gauche si brin
oho = [2,3,4,7,8,9,10]        # autorisés au dessus si brin
oba = [2,5,6,7,8,9,10]        # autorisés au dessous si brin
fdr = [0,2,3,5]               # autorisés à droite si non brin
fga = [0,2,4,6]               # autorisés à gauche si non brin
fho = [0,1,5,6]               # autorisés au dessus si non brin
fba = [0,1,3,4]               # autorisés au dessous si non brin

#"gauche,droite,dessus,dessous"

correct = [[fga, fdr, fho, fba],
        [oga, odr, fho, fba],
        [fga, fdr, oho, oba],
        [fga, odr, fho, oba],
        [oga, fdr, fho, oba],
        [fga, odr, oho, fba],
        [oga, fdr, oho, fba],
        [oga, odr, oho, oba],
        [oga, odr, oho, oba],
        [oga, odr, oho, oba],
        [oga, odr, oho, oba]]


chaise = [[0,3,4,0,0],[3,7,8,4,0],[5,8,7,8,4],[0,5,8,6,2],[0,0,5,1,6]]
huit = [[0,3,4,0,0],[3,7,8,1,4],[2,5,7,4,2],[2,3,8,6,2],[5,6,5,1,6]]
trefle = [[0,3,4,0],[3,8,7,4],[5,7,9,6],[0,5,6,0]]
exemple_1 = [[0, 3, 4, 0, 0, 3, 4, 0, 3, 1, 1, 4, 3, 4, 0, 0, 0], [0, 5, 7, 4, 3, 10, 7, 4, 2, 0, 0, 5, 8, 7, 1, 4, 0], [0, 3, 8, 9, 6, 2, 5, 7, 10, 1, 1, 1, 7, 8, 4, 2, 0], [3, 6, 5, 8, 1, 7, 1, 10, 7, 4, 3, 4, 2, 5, 6, 5, 4], [2, 3, 1, 10, 4, 2, 3, 9, 8, 7, 8, 9, 8, 1, 4, 0, 2], [5, 6, 0, 2, 2, 5, 7, 10, 7, 10, 7, 8, 7, 4, 2, 3, 6], [0, 0, 3, 10, 7, 4, 2, 5, 8, 7, 10, 7, 8, 7, 6, 2, 0], [3, 1, 8, 6, 2, 5, 8, 4, 5, 8, 7, 9, 9, 6, 3, 9, 4], [2, 0, 5, 4, 5, 1, 6, 5, 1, 9, 6, 5, 8, 1, 10, 8, 6], [5, 1, 1, 10, 4, 3, 4, 3, 1, 9, 1, 4, 2, 0, 5, 7, 4], [0, 0, 0, 2, 2, 5, 7, 6, 3, 7, 4, 2, 5, 1, 1, 9, 6], [3, 1, 1, 8, 7, 1, 6, 0, 2, 2, 5, 8, 1, 4, 3, 7, 4], [2, 0, 3, 9, 8, 4, 0, 3, 7, 8, 4, 5, 4, 2, 2, 2, 2], [2, 0, 5, 9, 7, 8, 1, 10, 6, 2, 5, 1, 7, 8, 10, 6, 2], [2, 0, 3, 6, 5, 10, 4, 5, 4, 5, 4, 3, 9, 7, 10, 4, 2], [5, 4, 5, 4, 3, 8, 7, 1, 6, 3, 6, 2, 5, 9, 6, 5, 6], [0, 5, 1, 6, 5, 6, 5, 1, 1, 6, 0, 5, 1, 6, 0, 0, 0]]

def mva(m):
    c = ""
    for i in m :
        c += str(i[0])
        for j in range(1,len(i)) :
            c += " & "
            c += str(i[j])
        c += " \\ "
    return c

def mvm(l):
    c = ""
    for i in l :
        c += str(i)
        c += " "
    return c


def mouline(n,i,c):

    """cherche un noeud a au moins c composantes premières et moins de i isthmes"""

    global m
    l = detor(mgauss(m)[0])
    if l != [] :
        l = nprem(l)
    else :
        m = aualean(n)
        return mouline(n,i,c)
    u,v = 0,0 
    for j in l : 
        if len(j) > 1 :
            u += 1
        else : 
            v += 1
    if u >= c and v <= i:
        return m
    else : 
        m = aualean(n)
        return mouline(n,i,c)


def t(f,x):

    """temps d'execution de f(x)"""

    t_0 = clock()
    f(x)
    return clock()-t_0

global temps
temps = True

####### Génération de noeuds

def entoure(M) :
    
    """entoure la matrice de 0"""

    m = [[0 for i in range(len(M[0])+2)]]
    for i in M : 
        m += [[0]+i+[0]]
    m += [[0 for i in range(len(M[0])+2)]]
    return m


def est_noeud(m) :

    """vérifie si m est un noeud"""

    M = entoure(m)
    n,p = len(M),len(M[0])
    t = [[False for i in range(p)] for i in range(n)]

    def embranchement_correct (M,i,j) :
        n = M[i][j]
        if t[i][j] or n == 0:
            return True
        else :
            t[i][j] = True
            return (M[i][j-1] in correct[n][0]) and (M[i][j+1] in correct[n][1]) and (M[i-1][j] in correct[n][2]) and (M[i+1][j] in correct[n][3]) and embranchement_correct(M,i-1,j) and embranchement_correct(M,i+1,j) and embranchement_correct(M,i,j-1) and embranchement_correct(M,i,j+1) 

    b = False
    for i in range(n) :
        for j in range(p) :
            if not(t[i][j]) and M[i][j] != 0 :
                if not( embranchement_correct(M,i,j)) :
                    return False
                else :
                    b = True
    return b



def inter(a,b,c=[],d=[],boule = False):

    """réalise l'intersection de a et b et c et d si non vides"""
    
    if not(boule):
        L = [] 
        for i in a :
            if i in b :
                L += [i]
        if c != [] :
            return inter(L,c,d,[])
        else : 
            return L
    else :
        L = [] 
        i,j = 0,0
        while i < len(a) and j< len(b):
            if a[i] == b[j]:
                L += [a[i]]
                i += 1
                j += 1
            elif a[i] > b[j]:
                j += 1
            else : 
                i += 1
        if c != [] :
            return inter(L,c,d,[],True)
        else :
            return L


def interl(l) :

    """intersection des listes l"""
    
    if l != [] :
        t = l[0]
        for i in l :
            t = inter(t,i)
        return t
    else : 
        return []


def premant(x,l):

    """Premier antécédant de x dans l"""

    c = 0 
    while x != l[c]:
        c += 1
    return c


def alean_original(n,p=0) : 

    """renvoie un entrelas aléatoire dans une matrice de taille n"""

    if p == 0 :
        p = n
    cor = correct+[[[i for i in range(11)],[i for i in range(11)],[i for i in range(11)],[i for i in range(11)]]]
    m = [[(11) for i in range(p)] for i in range(n)]
    m = entoure(m)
    for i in range(1,n+1):
        for j in range(1,p+1):
            l = inter(cor[m[i][j-1]][1],cor[m[i][j+1]][0],cor[m[i-1][j]][3],cor[m[i+1][j]][2])
            m[i][j]=l[randint(0,len(l)-1)]
    return [[m[i][j] for j in range(1,p+1)] for i in range(1,n+1)]


def alean(n,p=0) : 

    """renvoie un entrelas aléatoire dans une matrice de taille n"""
    
    t0 = clock()
    if p == 0 :
        p = n
    cor = correct+[[[i for i in range(11)],[i for i in range(11)],[i for i in range(11)],[i for i in range(11)]]]
    m = [[(11) for i in range(p)] for i in range(n)]
    m = entoure(m)
    for i in range(1,n):
        for j in range(1,p):
            l = inter(cor[m[i][j-1]][1],cor[m[i-1][j]][3])
            m[i][j]=l[randint(0,len(l)-1)]
        l = inter(cor[m[i][p-1]][1],cor[m[i][p+1]][0],cor[m[i-1][p]][3])
        m[i][p]=l[randint(0,len(l)-1)]
    for j in range(1,p+1):
        l = inter(cor[m[n][j-1]][1],cor[m[n][j+1]][0],cor[m[n-1][j]][3],cor[m[n+1][j]][2])
        m[n][j]=l[randint(0,len(l)-1)]
    if temps :
        print("entrelac",clock()-t0)
    return [[m[i][j] for j in range(1,p+1)] for i in range(1,n+1)]


def tressen(n,p=0) : 

    """renvoie un entrelas aléatoire dans une matrice de taille n"""
    if p == 0 :
        p = n
    cor = correct+[[[i for i in range(11)],[i for i in range(11)],[i for i in range(11)],[i for i in range(11)]]]
    m = [[(11) for i in range(p)] for i in range(n)]
    m = entoure(m)
    for i in range(1,p+1):
        m[1][i] = 2
        m[n][i] = 2
    for i in range(2,n):
        for j in range(1,p+1):
            l = inter(cor[m[i][j-1]][1],cor[m[i][j+1]][0],cor[m[i-1][j]][3],cor[m[i+1][j]][2])
            m[i][j]=l[randint(0,len(l)-1)]
    return [[m[i][j] for j in range(1,p+1)] for i in range(1,n+1)]


def compose(m,l):

    """crèe une composante suivant l"""
    
    for i in range(1,len(l)-1):
        a,b = l[i]
        L = moinsl([(a+1,b),(a-1,b),(a,b+1),(a,b-1)],[l[i+1]])
        (a1,b1),(a2,b2),(a3,b3) = L[0],L[1],L[2]
        if m[a][b] == 0:
            m[a][b] = inter(correct[m[a1][b1]][posrel(m,a,b,a1,b1)],correct[m[a2][b2]][posrel(m,a,b,a2,b2)],correct[m[a3][b3]][posrel(m,a,b,a3,b3)])[0]
        elif m[a][b] in [1,2] :
            m[a][b] = randint(7,8)
        elif m[a][b] in [3,6]:
            m[a][b] = 9
        elif m[a][b] in [4,5]:
            m[a][b] = 10
    return m


def aleaun(n,p=0):

    """noeud à un brin dans une matrice de taille n,p """

    m = alean(n,p)
    t = [[0 for i in range(len(m[0])+2)] for i in range(len(m)+2)]
    l = maxl([i[0] for i in parcours(m)])
    l = l + [l[0]]
    for i in range(len(l)):
        a,b = l[i]
        l[i] = (a+1,b+1)
    a,b=l[0]
    t[a][b] = 3
    t = compose(t,l)
    return [[t[i][j] for j in range(1,len(t[0])-1)] for i in range(1,len(t)-1)] 


def aaleaun(n,p=0) :

    """noeud à un brin alterné de taille n,p"""

    return alterne(aleaun(n,p))


def estjoin(a,b):

    """true si a et b ont un élément en commun"""

    for i in a : 
        for j in b : 
            if i ==j :
                return True
    return False


def estjoint(a,b):

    """true si a et b triées ont un élément en commun"""
    
    if a == [] or b == []:
        return False
    i,j = 0,0
    while i < len(a) and j < len(b) :
        if a[i] == b[j]:
            return True
        if R(a[i],b[j]) :
            j += 1 
        else :
            i += 1
    return False


def conn(k,L) :

    """graphe des contacts  non orientés de la composante connexe de la kieme liste de L"""
    
    v = [False for i in range(len(L))]
    v[k] = True
    g = [[] for i in range(len(L))]
    for i in range(len(L)):
        for j in range(len(L)) :
            if i != j:
                if not(i in g[j]):
                    if estjoint(L[i],L[j]):
                        g[i] += [j]
                        g[j] += [i]
    def parcours(i):
        for j in range(len(g[i])):
            if not(v[g[i][j]]):
                v[g[i][j]]=True
                parcours(g[i][j])
    parcours(k)
    return v,g


def ualean(n,p=0):

    """noeud aléatoire de taille n,p"""
    
    return lie(alean(n,p)) 


def lie(m) :

    """lie les composantes connexes reliables de m, peu de rapport avec les crochets"""
   
    t0 = clock()
    l = composanteco(m) 
    if len(l) != 1:
        ind,lenm = 0,len(l[0][0])
        for i in range(1,len(l)) :
            if len(l[i][0]) > lenm:
                ind = i
                lenm = len(l[i][0])
        supp = []
        sec = [trif(i) for i in secl(l)]
        v,g = conn(ind,sec)
        N = len(g)
        compo = [i for i in range(N)]
        for i in range(N):
            if v[i] and g[i] != []:
                rel = [compo[i]]
                for j in range(len(g[i])):
                    k = g[i][j]
                    pos = intert(sec[i],sec[k])
                    a,b = pos[randint(0,len(pos)-1)]
                    if m[a][b] in [7,8]:
                        m[a][b] = randint(9,10)
                    else :
                        m[a][b] = randint(7,8)
                    rel += [compo[k]]
                connecte =[]
                for h in range(N):
                    if compo[h] in rel :
                        connecte += [h]
                rep = min(connecte)
                for h in range(N) :
                    if h in connecte :
                        compo[h] = rep
                        g[h] = moinsl(g[h],connecte)
                    else : 
                        if v[h] :
                           g[h] = unocc(g[h],connecte)
            if not(v[i]) :
                for i in l[i][0]:
                    a,b = i 
                    m[a][b] = 0
    if temps :
        print("lié",clock()-t0)
    return m


def alterne(m):

    """alterne le noeud m"""

    n,p = len(m),len(m[0])
    marq = [[0 for i in range(p)] for i in range(n)]
    t0 = clock()
    l,der = [],False
    for i in range(n) :
        for j in range(p):
            if m[i][j] != 0 and marq[i][j] in [0,1]:
                a,b,x,y = i+1,j,i,j
                ref = (i,j)
                while (a,b) != ref :
                    if m[a][b] in [7,8,9,10] : 
                        if marq[a][b] == 0:
                            marq[a][b] = 1
                            if m[a][b] in [7,8]:
                                if der :
                                    m[a][b] = 7
                                else : 
                                    m[a][b] = 8
                                der = not(der)
                        elif marq[a][b] == 1:
                            marq[a][b] = 2
                            if m[a][b] in [7,8]:
                                der = not(der)
                    else : 
                        marq[a][b] = 2
                    if m[a][b] in [1,2,7,8]:
                        a,b,x,y = ((2*a)-x),((2*b)-y),a,b
                    elif m[a][b] in [4,5,10] :
                        a,b,x,y = (a+(b-y)),(b+(a-x)),a,b
                        der = not(der)
                    else :
                        a,b,x,y = (a-(b-y)),(b-(a-x)),a,b
                        der = not(der)
    if temps :
        print("alterné",clock()-t0)
    return m


def aualean(n,p=0):

    """noeud alterné a un seul brin dans une matrice de taille n,p ou n,n si p = 0"""

    return alterne(ualean(n,p))


def aleaprem(n,p=0):

    """noeud premier de taille n"""

    global m
    m = reduit(aualean(n,p))
    l = nprem(m)
    k = 0
    ma = 0
    for i in range(len(l)):
        if len(l[i]) > ma : 
            k = i 
            ma = len(l[i])
    return isole(deepcopy(m),k,l)


def isole(m,c,l=[]):

    """isole la composante première c de m"""

    if l == []:
        m = reduit(m)
        l = nprem(m)
    if len(l) in[0,1] :
        return m
    parc = parcourss(m)[0]
    boule = True
    marq = [[False for i in range(len(m[0]))] for i in range(len(m))]
    for i in l[c]:
        a,b = i
        marq[a][b] = True
    k = 0 
    a,b = parc[0]
    while not(marq[a][b]):
        k += 1
        a,b = parc[k] 
    parc = parc[(k+1):]+parc[:(k+1)]
    e,f = a,b
    for i in range(len(parc)) : 
        a,b = parc[i] 
        if m[a][b] in [7,8]:
            if boule and not(marq[a][b]):
                u,v = parc[i-1]
                m = supp(m,a,b,posrel(m,u,v,a,b))
            if not(boule) and marq[a][b]:
                m = relie(m,a,b,e,f)
            boule = marq[a][b]
            if marq[a][b] :
                e,f = a,b
        if not(boule) :
            if m[a][b] in [9,10] : 
                c,d = parc[i-1]
                h = posrel(m,a,b,c,d)
                if m[a][b] == 9 :
                    m[a][b] = 6-3*(h%2)
                else : 
                    m[a][b] = 4 + int(((h%3)+1)/2)
            else : 
                m[a][b] = 0
    return reduit(m)
        

########### Fonctions de parcours

def parcourss(m) :

    """Renvoie la liste des parcours de chacun des noeuds de m"""

    n,p = len(m),len(m[0])
    marq = [[0 for i in range(p)] for i in range(n)]
    l = []
    for i in range(n) :
        for j in range(p):
            if m[i][j] != 0 and (marq[i][j] in [0,1]):
                a,b,x,y = i+1,j,i,j
                acc = [(i,j)]
                while (a,b) != acc[0] :
                    if m[a][b] == 9 :
                        if marq[a][b] == 0:
                            marq[a][b] = 1
                        elif marq[a][b] == 1:
                            marq[a][b] = 2
                    else :
                        marq[a][b] = 2
                    acc += [(a,b)]
                    if m[a][b] in [1,2,7,8]:
                        a,b,x,y = ((2*a)-x),((2*b)-y),a,b
                    elif m[a][b] in [4,5,10] :
                        a,b,x,y = (a+(b-y)),(b+(a-x)),a,b
                    else :
                        a,b,x,y = (a-(b-y)),(b-(a-x)),a,b
                l += [acc]
    return l


def parcours(m) :

    """Renvoie la liste des parcours de chacun des noeuds de m ainsi  que la liste des intersections propres à chaque noeud"""
    
    n,p = len(m),len(m[0])
    marq = [[0 for i in range(p)] for i in range(n)]
    l,c =[],0
    for i in range(n) :
        for j in range(p):
            if m[i][j] != 0 and (marq[i][j] >= 0):
                c += 1
                a,b,x,y = i+1,j,i,j
                acc = [(i,j)]
                acci = []
                while (a,b) != acc[0] :
                    if m[a][b] in [7,8] and marq[a][b] == c:
                        acci += [(a,b)]
                    if m[a][b] in [7,8,9,10] :
                        if marq[a][b] == 0:
                            marq[a][b] = c
                        elif marq[a][b] >= 0:
                            marq[a][b] = -1
                    else :
                        marq[a][b] = -1
                    acc += [(a,b)]
                    if m[a][b] in [1,2,7,8]:
                        a,b,x,y = ((2*a)-x),((2*b)-y),a,b
                    elif m[a][b] in [4,5,10] :
                        a,b,x,y = (a+(b-y)),(b+(a-x)),a,b
                    else :
                        a,b,x,y = (a-(b-y)),(b-(a-x)),a,b
                l += [[acc,acci]]
    return l


def composante(m):

    """Renvoie la liste des parcours et case correspondante de chacun des noeuds de m"""

    n,p = len(m),len(m[0])
    marq = [[0 for i in range(p)] for i in range(n)]
    l = []
    for i in range(n) :
        for j in range(p):
            if m[i][j] != 0 and (marq[i][j] in [0,1]):
                a,b,x,y = i+1,j,i,j
                acc = [[(i,j),m[i][j]]]
                ref = (i,j)
                while (a,b) != ref :
                    if m[a][b] == 9 :
                            if marq[a][b] == 0:
                                marq[a][b] = 1
                            elif marq[a][b] == 1:
                                marq[a][b] = 2
                    else :
                        marq[a][b] = 2
                    acc += [[(a,b),m[a][b]]]
                    if m[a][b] in [1,2,7,8]:
                        a,b,x,y = ((2*a)-x),((2*b)-y),a,b
                    elif m[a][b] in [4,5,10] :
                        a,b,x,y = (a+(b-y)),(b+(a-x)),a,b
                    else :
                        a,b,x,y = (a-(b-y)),(b-(a-x)),a,b
                l += [acc] 
    return l


def composanteco(m):

    """Renvoie la liste des parcours de chacun des noeuds de m ainsi  que la liste des contacts entre chaque composante"""
    
    n,p = len(m),len(m[0])
    marq = [[0 for i in range(p)] for i in range(n)]
    l,cont = [],[]
    c = 0
    for i in range(n) :
        for j in range(p):
            if m[i][j] != 0 and (marq[i][j] >= 0):
                c += 1
                a,b,x,y = i+1,j,i,j
                acc = [(i,j)]
                if m[i][j] == 9 :
                    doubi,contint = [(i,j)],[(i,j)]
                    marq[i][j] = -1
                else :
                    doubi,contint = [],[]
                while (a,b) != acc[0] :
                    if m[a][b] in [7,8,9,10]:
                        if marq[a][b] ==0:
                            doubi += [(a,b)]
                            marq[a][b] = c
                        else :
                            if marq[a][b] != c :
                                contint += [(a,b)]
                            marq[a][b] = -1
                    else : 
                        marq[a][b] = -1
                    acc += [(a,b)]
                    if m[a][b] in [1,2,7,8]:
                        a,b,x,y = ((2*a)-x),((2*b)-y),a,b
                    elif m[a][b] in [4,5,10] :
                        a,b,x,y = (a+(b-y)),(b+(a-x)),a,b
                    else :
                        a,b,x,y = (a-(b-y)),(b-(a-x)),a,b
                for k in doubi :
                    a,b = k
                    if marq[a][b] == c:
                        contint += [(a,b)]
                l += [[acc,contint]]
    return l

########## Manipulation des listes

def maxdouble(l) :

    """couple du maximum de chacun des couples de l"""

    c,d = 0,0
    for i in l : 
        a,b = i
        if a > c : 
            c = a
        if b > d : 
            d = b
    return c,d


def mindouble(l):

    """couple du minimum de chacun des couples de l"""

    c,d = l[0]
    for i in l :
        a,b = i 
        if a < c :
            c = a
        if b < d : 
            d = b 
    return c,d


def maxl(l):

    """liste de plus grande taille de l"""

    L = []
    for i in l :
        if i != None :
            L += [len(i)]
        else : 
            L += [0]
    return l[premant(max(L),L)]


def preml(l) :

    """liste des premiers éléments de chaque couple de l"""

    L = []
    for i in l :
        L += [i[0]]
    return L


def secl(l) :

    """liste des second éléments de chaue couple de l"""

    L = []
    for i in l :
        L += [i[1]]
    return L


def R(x,y):

    """ x si plus grand que y pour ordre lexicographique"""

    x1,x2 = x
    y1,y2 = y
    if x1 > y1 :
        return True
    elif x1 == y1 and x2 >= y2 :
        return True
    else :
        return False


def decoupe(l):

    """découpe l en deux parties"""

    m,n = [],[]
    while len(l) > 1 :
        m += [l.pop()]
        n += [l.pop()]
    if l !=[]:
        m += [l.pop()]
    return m,n


def recole(a,b) :

    """liste triée des couples de a et de b"""

    L = []
    while a != [] and b != []:
        if R(a[0],b[0]):
            L += [b.pop(0)]
        else :
            L += [a.pop(0)]
    return L+a+b


def trif(l):

    """tri fusion de l liste de couples"""

    if len(l) <= 1:
        return l 
    else :
        m,n = decoupe(l)
        return recole(trif(m),trif(n))


def intert(a,b):

    """intersection de a et b listes de couples triées"""

    i,j = 0,0
    L = []
    while i < len(a) and j < len(b) :
        if a[i] == b[j] :
            L += [a[i]]
            i += 1
            j += 1
        elif R(a[i],b[j]):
            j += 1
        else :
            i += 1
    return L


def rev(l):

    """pareil que l.reverse() sans toucher la liste"""

    return [l[i] for i in range(len(l)-1,-1,-1)]


def moinsl(a,b):

    """liste a sans les éléments de b"""

    if a == [] :
        return a 
    else :
        return [i for i in a if not(i in b)]


def unocc(a,b):

    """liste a avec un seul des éléments de b quand c'est possible, a sinon"""

    pos = inter(a,b)
    if len(pos) <= 1 :
        return a
    else : 
        c = pos[randint(0,len(pos)-1)]
        return moinsl(a,suppr(c,b))


def suppr(x,l):

    """liste l sans l'élément x"""

    return [i for i in l if i != x]


def rev(l):

    """pareil que l.reverse() sans toucher la liste"""

    return [l[i] for i in range(len(l)-1,-1,-1)]

########## Outils sur les noeuds 

def mgauss_original(m):

    """ Renvoie la liste de chacun des mots de gauss associés à chacun des noeuds de m""" 

    l = parcours(m)
    g = []
    for i in l : 
        noeu = i[1]
        dia = []
        for k in i[0] :
            if k in noeu :
                j = 0 
                while noeu[j] != k : 
                    j += 1
                dia += [j]
        g += [dia]
    for i in range(len(g)):
        g[i] = renum(g[i])
    return g


def mgauss(m,boule = False,nprem = False):

    """liste de chacun des mots de gauss de chacun des noeuds de m, si boule ne numérote pas les sommets""" 
   
    n,p = len(m),len(m[0])
    marq = [[0 for i in range(p)] for i in range(n)]
    l,c =[],0
    t0 = clock()
    for i in range(n) :
        for j in range(p):
            if m[i][j] in [3,9] and (marq[i][j] >= 0):
                c += 1
                a,b,x,y = i+1,j,i,j
                ref = (i,j)
                acc = []
                while (a,b) != ref:
                    if m[a][b] in [7,8]:
                        acc += [(a,b)]
                        if marq[a][b] == 0:
                            marq[a][b] = c
                        elif marq[a][b] != c:
                            marq[a][b] = -1
                    elif m[a][b] == 9 and  marq[a][b] == 0 :
                        marq[a][b] = c
                    else :
                        marq[a][b] = -1
                    if m[a][b] in [1,2,7,8]:
                        a,b,x,y = ((2*a)-x),((2*b)-y),a,b
                    elif m[a][b] in [4,5,10] :
                        a,b,x,y = (a+(b-y)),(b+(a-x)),a,b
                    else :
                        a,b,x,y = (a-(b-y)),(b-(a-x)),a,b
                l += [acc]
    L = []
    if not(boule):
        corr = [[(-1) for i in range(p)] for i in range(n)]
        for i in l :
            c = 0 
            mg = []
            for j in i :
                a,b = j
                if marq[a][b] >= 0 :
                    if corr[a][b] >= 0 :
                        mg += [corr[a][b]]
                    else :
                        corr[a][b] = c 
                        mg += [c]
                        c += 1
            L += [mg]
    else :
        for i in l:
            mg = []
            for j in i :
                a,b = j
                if marq[a][b] >= 0:
                    mg += [j]
            L += [mg]
    if temps :
        print("mgauss",clock()-t0)
    if nprem :
        return L,[[corr[i][j],(i,j)] for i in range(n) for j in range(p)]
    return L


def colonne(m,i):

    """renvoie la ieme colonne de m"""

    l = []
    for k in m :
        l += [k[i]]
    return l


def icolonne(i,m):

    """insère une colonne en i"""

    M = []
    for k in m:
        v = 0
        if k[i-1] in oga and k[i] in odr :
            v = 1
        M += [k[:i] + [v] + k[i:]]
    return M


def iligne(i,m):

    """insère une ligne en i"""

    M = []
    for k in range(i):
        M += [m[k]]
    l = []
    for k in range(len(m[0])):
        if m[i-1][k] in oho and m[i][k] in oba :
            l += [2]
        else :
            l += [0]
    M += [l]
    for k in range(i,len(m)):
        M += [m[k]]
    return M


def rpdoc(m,i,j) :

    """remplace un 9 ou 10 en m[i][j] par des chevrons"""

    M = m.copy()            
    if M[i][j] in [9,10] :
        M = icolonne(j+1,M)
        if M[i][j] == 9 :
            M[i][j+1] = 3
            M[i][j] = 6
            for k in range(i+1,len(M)):
                M[k][j+1]=M[k][j]
                if M[k][j+1] in odr and M[k][j-1] in oga :
                    M[k][j]= 1
                else :
                    M[k][j] = 0
        else : 
            M[i][j+1] = 5
            M[i][j] = 4
            for k in range(0,i):
                M[k][j+1]=M[k][j]
                if M[k][j+1] in odr and M[k][j-1] in oga :
                    M[k][j]= 1
                else :
                    M[k][j] = 0
    return M


def rpdo(m) :

    """remplace les 9 et 10 dans m par des chevrons"""

    M = m.copy()
    i,j = 0,0
    while i<len(M) :
        while j<len(M[0]) :
            if M[i][j] in [9,10]:
                M = rpdoc(M,i,j)
            j += 1
        i += 1 
        j = 0
    return M


def reduc(m):

    """supprime les colonnes et lignes de 0 ou ne comprenant que des 1 ou que 2 avec des 0"""
    
    M = []
    for i in m :
        c = [0 for i in range(11)]
        for j in i :
            c[j] += 1
        if not(c[0]+c[1]+c[2] == len(i) and (c[1] ==0 or c[2]==0)):
            M += [i]
    l = []
    for i in range(len(M[0])):
        c = [0 for i in range(11)]
        for j in range(len(M)):
            c[M[j][i]]+= 1
        if not(c[0]+c[1]+c[2] == len(M) and (c[1] ==0 or c[2]==0)):
            l += [i]
    return [[M[i][j] for j in l] for i in range(len(M))]


def miroir(m):

    """miroir de m"""

    M = []
    for i in m :
        l = []
        for j in i : 
            if j == 8 : 
                l += [7]
            elif j == 7 : 
                l += [8]
            else : 
                l += [j]
        M += [l]
    return M


def equiv(l,L):
    
    """vrai si l équivalent à L"""
    
    l,L = detor(l),detor(L)
    if len(l) != len(L) :
        return False
    lm = renum(rev(l))
    for i in range(len(l)):
        L = renum(L[1:]+[L[0]])
        if l == L or lm == L : 
            return True
    return False

######## Réduction

def encercle(t):
    
    """entoure t d'un noeud trivial tout à fait tendance"""

    m = deepcopy(t)
    for i in range(len(m)):
        m[i] = [2]+m[i]+[2]
    return [[3]+[1 for i in range(len(m[0])-2)]+[4]]+m+[[5]+[1 for i in range(len(m[0])-2)]+[6]]


def ouvertureh(m,x,y):

    """-1 si ouvert en haut, 1 en bas, 0 sinon"""

    if x > 0 and not(m[x][y] in correct[m[x-1][y]][3] and m[x-1][y] in correct[m[x][y]][2]):
        return -1
    if x < len(m) and not(m[x][y] in correct[m[x+1][y]][2] and m[x+1][y] in correct[m[x][y]][3]):
        return 1
    return 0


def ouverturec(m,x,y) :

    """-1 si ouvert à gauche, 1 à droite, 0 sinon"""

    if y > 0 and not(m[x][y] in correct[m[x][y-1]][1] and m[x][y-1] in correct[m[x][y]][0]):
        return -1
    if y < len(m[0]) and not(m[x][y] in correct[m[x][y+1]][0] and m[x][y+1] in correct[m[x][y]][1]):
        return 1
    return 0


def ouverture(m,x,y) : 

    """couple de 1,0,-1 indiquant une ouverture de m[x][y]"""
    
    return (ouvertureh(m,x,y),ouverturec(m,x,y))


def posrel(m,a,b,x,y):
    
    """ 0 si (a,b) à gauche de (x,y), 1 à droite, 2 dessus, 3 dessous"""

    if a == x and y > 0 and b == y-1 : 
        return 0
    elif a == x and y < len(m[0]) and b == y+1 :
        return 1
    elif b == y and x > 0 and a == x-1 :
        return 2
    elif b == y and x < len(m) and a == x+1 :
        return 3
    else : 
        return None


def relie(m,di,dj,ai,aj,boule = False):

    """complète m sur un chemin de taille minimum entre (di,dj) et (ai,aj) points du noeud"""

    if di == ai and aj == dj :
        h,c = ouverture(m,di,dj)
        if c == 1 :
            m[ai][aj+1] = 5-h
        else :
            m[ai][aj-1] = 4-h
        return relie(m,ai,aj,di,dj+c)
    di,dj,ai,aj = di+ouvertureh(m,di,dj),dj+ouverturec(m,di,dj),ai+ouvertureh(m,ai,aj),aj+ouverturec(m,ai,aj)
    if boule :
        return join(m,di,dj,ai,aj)
    else : 
        return join_rec(m,di,dj,ai,aj)


def join_rec(m,di,dj,ai,aj):
    
    """ chemin minimum entre (di,dj) et (ai,aj) non points du noeud"""
    
    global predr
    m = encercle(m)
    n,p = len(m),len(m[0])
    ai,aj,di,dj = ai+1,aj+1,di+1,dj+1
    p = [(di,dj)]
    predr = [[0 for i in range(len(m[0]))] for j in range(len(m))]
    def parcours(p):
        global predr
        if p == [] :
            return False
        else : 
            a,b = p[0]
            if (a,b) == (ai,aj) :
                return True
            l = []
            if m[a-1][b] in [0,5,6] and predr[a-1][b] == 0:
                predr[a-1][b] = (a,b)
                l += [(a-1,b)]
            if m[a+1][b] in [0,3,4] and predr[a+1][b] == 0:
                predr[a+1][b] = (a,b)
                l += [(a+1,b)]
            if m[a][b-1] in [0,4,6] and predr[a][b-1] == 0:
                predr[a][b-1] = (a,b)
                l += [(a,b-1)]
            if m[a][b+1] in [0,3,5] and predr[a][b+1] == 0:
                predr[a][b+1] = (a,b)
                l += [(a,b+1)]
            return parcours(p[1:]+l)
    if parcours (p):
        def recchem(x,y,acc):
            global predr
            if (x,y)== (di,dj):
                return acc
            else :
                a,b = predr[x][y]
                return recchem(a,b,[(a,b)]+acc)
        chem = [(di,dj)]+recchem(ai,aj,[(ai,aj)])+[(ai,aj)]
        m = compose(m,chem)
    return [[m[j][i] for i in range(1,len(m[0])-1)] for j in range(1,len(m)-1)]


def join(m,di,dj,ai,aj):
    
    """ chemin minimum entre (di,dj) et (ai,aj) non points du noeud"""
   
    n,p = len(m),len(m[0])
    a,b = di,dj
    p = [(di,dj)]
    predr = [[0 for i in range(len(m[0]))] for j in range(len(m))]
    while (a,b) != (ai,aj):
        if p == []:
            return "ya comprend pas"
        a,b = p.pop(0)
        if m[a-1][b] in [0,5,6] and predr[a-1][b] == 0:
            predr[a-1][b] = (a,b)
            p += [(a-1,b)]
        if m[a+1][b] in [0,3,4] and predr[a+1][b] == 0:
            predr[a+1][b] = (a,b)
            p += [(a+1,b)]
        if m[a][b-1] in [0,4,6] and predr[a][b-1] == 0:
            predr[a][b-1] = (a,b)
            p += [(a,b-1)]
        if m[a][b+1] in [0,3,5] and predr[a][b+1] == 0:
            predr[a][b+1] = (a,b)
            p += [(a,b+1)]
    def recchem(x,y,acc):
        if (x,y)== (di,dj):
            return acc
        else :
            a,b = predr[x][y]
            return recchem(a,b,[(a,b)]+acc)
    chem = [(di,dj)]+recchem(ai,aj,[(ai,aj)])+[(ai,aj)]
    m = compose(m,chem)
    return m


def supp(m,di,dj,direc,parc = [],num = 0):

    """supprime les éléments de m après (di,dj) dans la direction direc jusqu'au croisemet suivant 0 gauche, 1 droite, 2 dessus et 3 dessous"""
    
    if parc == [] :
        parc = parcours(m)[num][0]
    parc = parc*2 + rev(parc)*2
    boule = False
    for i in range(len(parc)):
        a,b = parc[i]
        if boule :
            if m[a][b] in [9,10] : 
                c,d = parc[i-1]
                h = posrel(m,a,b,c,d)
                if m[a][b] == 9 :
                    m[a][b] = 6-3*(h%2)
                else : 
                    m[a][b] = 4 + int(((h%3)+1)/2)
            elif m[a][b] in [7,8] :
                return m
            else : 
                m[a][b] = 0
        if a == di and b==dj :
            c,d = parc[i+1]
            if posrel(m,c,d,a,b) == direc:
                boule = True
    return m


def detor(m,num= 0,l = []):

    """enlève les torsades de m mot de gauss ou m noeud"""

    if type(m[0]) in [int,tuple]:
        l = m.copy()
        i = 0
        while i < len(l)-1 :
            if l[i] == l[i+1] :
                l.remove(l[i])
                l.remove(l[i])
                if i > 0 : 
                    i = i-1
            else : 
                i += 1
        while len(l) >0 and l[0] == l[len(l)-1]:
            l.remove(l[len(l)-1])
            l.remove(l[len(l)-1])
        if l != []:
            if type(m[0]) == tuple :
                return l
            else :
                return renum(l,True)
        else : 
            return []
    else :
        if l == [] :
            l = parcours(m)[num]
        mot = mgauss(m,True)[0]
        d = dict([(i,[-1,-1]) for i in mot])
        marq = [[False for i in range(len(m[0]))] for i in range(len(m))]
        for i in l[1]:
            a,b = i
            marq[a][b] = True
        for i in range(len(l[0])):
            a,b = l[0][i]
            if marq[a][b] :
                if d[l[0][i]][0] == -1 :
                    d[l[0][i]][0]=i
                else: 
                    d[l[0][i]][1] = i
        i = 0
        while i < len(mot)-1 :
            if mot[i] == mot[i+1] :
                a,b = mot[i]
                e,f = l[0][d[mot[i]][0]+1]
                g,h = l[0][d[mot[i]][1]-1]
                dep = posrel(m,a,b,e,f)%2
                arr = posrel(m,a,b,g,h)%2
                if (dep,arr) == (0,0) or (dep,arr) == (1,1):
                    m[a][b] = 10
                else :
                    m[a][b] = 9
                mot.remove(mot[i])
                mot.remove(mot[i])
                if i > 0 : 
                    i = i-1
            else : 
                i += 1
        while len(mot)>0 and mot[0] == mot[len(mot)-1]:
                a,b = mot[0]
                e,f = l[0][d[mot[0]][0]+1]
                g,h = l[0][d[mot[0]][1]-1]
                dep = posrel(m,a,b,e,f)%2
                arr = posrel(m,a,b,g,h)%2
                if (dep,arr) == (0,0) or (dep,arr) == (1,1):
                    m[a][b] = 10
                else :
                    m[a][b] = 9
                mot.remove(mot[len(mot)-1])
                mot.remove(mot[len(mot)-1])
        return m


def detors(m):

    """détorsade les sous-noeuds de m"""
    
    l = parcours(m)
    for i in range(len(l)):
        m = detor(m,i,l[i])
    return m


def sup_bis(m,i,parc):

    """supprime les éléments de m après (di,dj) dans la direction direc jusqu'au croisemet suivant 0 gauche, 1 droite, 2 dessus et 3 dessous"""
    
    boule = True
    while boule:
        a,b = parc[i]
        if m[a][b] in [9,10] : 
            c,d = parc[i-1]
            h = posrel(m,a,b,c,d)
            if m[a][b] == 9 :
                m[a][b] = 6-3*(h%2)
            else : 
                m[a][b] = 4 + int(((h%3)+1)/2)
        elif m[a][b] in [7,8] :
            boule = False
        else : 
            m[a][b] = 0
        i += 1
    return m,a,b



def reduit(m,num=0, l= []) :

    """ version simplifiée de m"""
    
    t2 = clock()
    t = clock()
    m = encercle(detor(deepcopy(m)))
    if temps :
        print("detor",clock() -t)
    if l == []:
        l = parcours(m)[num+1]
    if l[1] == [] : 
        return [[3,4],[5,6]]
    else : 
        mot = mgauss(m,True)[num+1]
        parc = l[0]
        supm = [[0 for i in range(len(m[0]))] for i in range(len(m))]
        for i in l[1]:
            a,b = i
            supm[a][b] = [0,0,0,0]
        t = clock()
        for i in range(1,len(parc)-1):
            a,b = parc[i]
            if supm[a][b] != 0:
                c,d = parc[i+1]
                supm[a][b][posrel(m,c,d,a,b)]=i
                c,d = parc[i-1]
                supm[a][b][posrel(m,c,d,a,b)]=3*len(parc)-1-i
        parc = 2*parc+(rev(parc))*2
        if temps :
            print("matrice",clock()-t)
        t=clock()
        t00 = 0
        t11 = 0
        for k in range(len(l[0])-1):
            a,b = l[0][k]
            if supm[a][b] != 0:
                c,d = l[0][k+1]
                if supm[c][d] == 0:
                    t = clock()
                    m,e,f = sup_bis(m,(supm[a][b][posrel(m,c,d,a,b)])+1,parc)
                    t00 += clock()-t
                    t =clock()
                    m = relie(m,a,b,e,f,True)
                    t11 += clock()-t
        if temps :
            print("reducted",clock()-t2)
            print("   supp",t00)
            print("   relie",t11)
        return reduc([[m[i][j] for j in range(1,len(m[0])-1)] for i in range(1,len(m)-1)])


def rev(l):

    """pareil que l.reverse() sans toucher la liste"""

    return [l[i] for i in range(len(l)-1,-1,-1)]


def translate(m,a,b,l=[],vois=[]):

    """tentative de réduction poussée"""

    if l == [] :
        l = parcours(m)[0]
    if vois == [] :
        mot = []
        for k in l[0]  :
            if k in l[1] :
                mot += [k]
        for i in range(len(mot)):
            if mot[i] == (a,b) :
                if i < len(mot)-1:
                    vois += [mot[i+1]]
                else : 
                    vois += [mot[0]]
                if i > 0 :
                    vois += [mot[i-1]]
                else : 
                    vois += [mot[len(mot)-1]]
    i = 0
    boule = False
    while l[0][i] != (a,b):
        i += 1 
    l = l + l[:i]
    col,lig = [],[]
    while l[0][i+1] != vois[0] :
        c,d = l[0][i+1]
        if c == 0 :
            if not(0 in lig):
                lig += [0]
        else : 
            if m[c-1][b]:
                ()
        if c == len(m)-1 : 
            lig += len(m)-1

########### Décomposition

def emboit(m):
    
    """emboitement des sommet du mot de gauss de m ou du mot de gauss m"""

    if type(m[0]) == list :
        m = mgauss(m)[0]
    t = [(-1) for i in range(int(len(m)/2))]
    for i in range(len(m)):
        if t[m[i]] == -1 :
            t[m[i]] = i
        else : 
            t[m[i]] = i-t[m[i]]-1
            if len(m)-t[m[i]]-2 < t[m[i]]:
                t[m[i]] = len(m)-t[m[i]]-2
    return t


def renum(l,boule = False):

    """renumérote le mot de gauss l en croissant"""
    
    n = int(len(l)/2)
    if boule : 
        m = max(l)+1
    else : 
        m = n
    t = [-1 for i in range(m)]
    c = 0
    for i in range(2*n) :
        if t[l[i]] == -1 :
            t[l[i]] = c
            c += 1
    for i in range(2*n) : 
        l[i] = t[l[i]]
    return l


def gremboit(m):
    
    """graphe d'emboitement du mot de gauss (de) m"""

    t_0 = clock()
    if type(m[0]) == list :
        m = mgauss(m)[0]
    n = int(len(m)/2)
    g = [[] for i in range(n)]
    t = [(-1) for i in range(n)]
    for i in range(len(m)):
        if t[m[i]] == -1 :
            t[m[i]] = i
        else : 
            dej = [False for j in range(n)]
            pos = t[m[i]]
            t[m[i]] = i-t[m[i]]-1
            if len(m)-t[m[i]]-2 < t[m[i]]:
                t[m[i]] = len(m)-t[m[i]]-2
                for j in range(pos):
                    if not(dej[m[j]]) : 
                        dej[m[j]] = True
                        g[m[i]] += [m[j]]
                for j in range(i+1,len(m)):
                    if not(dej[m[j]]) : 
                        dej[m[j]] = True
                        g[m[i]] += [m[j]]
            else : 
                for j in range(pos+1,i):
                    if not(dej[m[j]]) : 
                        dej[m[j]] = True
                        g[m[i]] += [m[j]]
    if temps :
        print("graphe",clock()-t_0)
    return g


def revg(g):
    
    """co-graphe de g"""

    t = [[] for i in range(len(g))]
    for i in range(len(g)):
        for j in g[i] : 
            t[j] += [i] 
    return t


def extno(g):
    
    """liste des listes de sommets des sous-graphes"""

    v = [-1 for i in range(len(g))]
    c = [0]
    def parcours(i):
        v[i] = c[0]
        c[0] += 1
        for j in range(len(g[i])):
            if v[g[i][j]] == -1 :
                parcours(g[i][j])
        v[i] = c[0]
        c[0] += 1
    for i in range(len(g)):
        if v[i] == -1 :
            parcours(i)
    t = revg(g)
    c = [0]
    comp = [-1 for i in range(len(g))]
    def indm(l,kind):
        k = kind
        while k < len(v) :
            if vu[l[k]]:
                return k
            else :
                k += 1
        return k
    def sparcours(i) :
        comp[i] = c[0]
        vu[i] = False
        for j in range(len(t[i])):
            if comp[t[i][j]] == -1 :
                sparcours(t[i][j])
    def recole(a,b) :
        L = []
        while a != [] and b != []:
            if v[a[0]] <= v[b[0]]:
                L += [b.pop(0)]
            else :
                L += [a.pop(0)]
        return L+a+b
    def trifu(l):   
        if len(l) <= 1:
            return l 
        else :
            m,n = decoupe(l)
            return recole(trifu(m),trifu(n))
    l = trifu([i for i in range(len(v))])
    vu = [True for i in range(len(v))]
    ind = 0
    ref = [0 for i in range(len(v))]
    while ind < len(v) :
        sparcours(l[ind])
        ind = indm(l,ind)
        c[0] += 1
    L = [[] for i in range(max(comp)+1)]
    for i in range(len(g)):
        L[comp[i]] += [i]
    return L


def extno_originale(g):
    
    """liste des listes de sommets des sous-graphes"""

    v = [-1 for i in range(len(g))]
    c = [0]
    def parcours(i):
        v[i] = c[0]
        c[0] += 1
        for j in range(len(g[i])):
            if v[g[i][j]] == -1 :
                parcours(g[i][j])
        v[i] = c[0]
        c[0] += 1
    for i in range(len(g)):
        if v[i] == -1 :
            parcours(i)
    t = revg(g)
    c = [0]
    comp = [-1 for i in range(len(g))]
    def indm(v):
        k = 0
        m = v[0]
        for i in range(len(v)):
            if v[i] > m:
                m = v[i]
                k = i
        return k
    def sparcours(i) :
        comp[i] = c[0]
        v[i] = 0
        for j in range(len(t[i])):
            if comp[t[i][j]] == -1 :
                sparcours(t[i][j])
    ref = [0 for i in range(len(v))]
    while v != ref :
        sparcours(indm(v))
        c[0] += 1
    L = [[] for i in range(max(comp)+1)]
    for i in range(len(g)):
        L[comp[i]] += [i]
    return L


def nprem(m):

    """si m mot de gauss, sous-mots de gauss, si m noeuds, listes des sous noeuds données par intersections"""

    if type(m[0]) == list :
        t0 = clock() 
        l,corr = mgauss(m,False,True)
        l = l[0]
        d = dict(corr)
        if l != [] :
            L = extno(gremboit(l))
        else : 
            L = []
        for i in range(len(L)):
            for j in range(len(L[i])):
                L[i][j] = d[L[i][j]]
        if temps :
            print("nprem",clock()-t0)
        return L
    else : 
        return extno(gremboit(m))


def coprem(m,boule = True,multi = False,fg = False):

    """affiche la coloration des composantes premières de m réduit ou sous diagramme si m mot, si boule reduit m, si multi calcule la figure sans l'afficher, fg pour fullgraph"""

    if type(m[0]) == list :
        if boule :
            m = reduit(m)
        if len(m) ==2 :
            affiche(m)
            return
        parc,comp = parcourss(m)[0],nprem(m)
        nb = 0 
        for i in comp :
            if len(i) > 1 : 
                nb += 1
        if nb >7 : 
            meth = [True]
        else : 
            meth = [False]
        t0 = clock()
        k = 0
        a,b = parc[k]
        while not(m[a][b] in [7,8]):
            k += 1
            a,b = parc[k]
        parc = parc+parc[:k]
        col = ['royalblue','olivedrab','darkred','purple','peru','dimgrey','teal']
        #col = ['royalblue','orange','blue','peru','cyan']
        coul = [-1 for i in range(len(comp))]
        couleur = [['' for i in range(len(m[0]))] for i in range(len(m))]
        k,isthme = [-1,-1],False
        def aux(a,b):
            for i in range(len(comp)):
                if (a,b) in comp[i]:
                    if len(comp[i]) == 1:
                        couleur[a][b] = 'black'
                        return True
                    else :
                        if coul[i] == -1 :
                            if meth[0] :
                                k[0] = (k[0]+1)%5
                                coul[i] = k[0]
                            else : 
                                k[1] = k[1] +1
                                k[0] = k[1]
                                coul[i]= k[1]
                        else :
                            k[0] = coul[i]
                        return False
        for i in range(len(parc)) : 
            a,b = parc[i] 
            if m[a][b] in [7,8]:
                isthme = aux(a,b)
            if m[a][b] in [9,10]:
                if couleur[a][b] == '' :
                    couleur[a][b] = ['','']
                    x2,y2 = parc[i-1]
                    x1,y1 = parc[i+1]
                    if posrel(m,a,b,x1,y1) == 2 or posrel(m,a,b,x2,y2) == 2:
                        couleur[a][b][0] = col[k[0]]
                    else : 
                        couleur[a][b][1] = col[k[0]]
                else : 
                    if couleur[a][b][0] == '' :
                        couleur[a][b][0] = col[k[0]]
                    else : 
                        couleur[a][b][1] = col[k[0]]
            else :
                if not(isthme) :
                    couleur[a][b] = col[k[0]]
                else : 
                    isthme = False
        if temps :
            print("colorié",clock()-t0)
        if fg :
            return comp,coul,col
        affprem(m,couleur,True,0.075,multi)
    else :
        m = detor(m)
        if m != [] :
            l = nprem(m)
            ism = []
            for i in range(len(l)):
                if len(l[i]) ==1 :
                    ism += l[i]
            l = [l[i] for i in range(len(l)) if len(l[i]) >1]
            v = [0 for i in range(int(len(m)/2))]
            for i in range(len(l)):
                for j in l[i]:
                    v[j] = i
            L = [[] for i in range(len(l))]
            for i in m :
                if not(i in ism):
                    L[v[i]] += [i]
            if fg : 
                return L
            corde(L,False,multi)
        else : 
            corde([[3,4],[5,6]])

############ Permutations

def cycle(l,i):

    """liste du  cycle de i dans l"""

    ref = i
    L = []
    while l[i] != ref :
        L = [l[i]]+L
        i = l[i]
    L = [ref]+L
    return L


def repcycle(l):

    """liste de la décomposition en cycle non triviaux du mot de gauss l ou les éléments sont les positions réelles"""

    t = l.copy()
    n = int(len(l)/2)
    m = [False for i in range(n)]
    for i in range(2*n) :
        if m[t[i]] :
            t[i] = t[i]+n
        else :
            m[t[i]] = True
    cy = []
    m = [True for i in range(2*n)]
    for i in range(2*n):
        if m[i] :
            L = cycle(t,i)
            for k in L : 
                m[k] = False
            if len(L) > 1 :
                cy += [L]
    return cy


def cyclem(m):

    """décomposition en cycle de la permutation associé au mot de gauss de m"""

    return repcycle(mgauss(m)[0])


def genS(n):

    """liste des permutations de sn"""

    return perm([i for i in range(n)])


def perm(l):

    """permute de toute les manière possible l"""

    if len(l) <= 1:
        return [l]
    else :
        spar = []
        for i in l :
            t = l.copy()
            t.remove(i)
            for j in perm(t):
                spar.append([i]+j)
        return spar
       

def suml(l,i=0,j=0):

    """somme des éléments de l entre i et j"""

    if j ==0 :
        j = len(l)
    c = 0
    for k in range(i,j):
        c += l[k]
    return c


def gpvn(l):

    """crèe un noeud à partir d'un mot de gauss"""
    
    n = int(len(l)/2)
    corr = [(-1) for i in range(n)]
    m = [[0 for i in range(n)]]
    conn = [(-1) for i in range(n)]
    corres = [] 
    c = 0
    while c < n :
        a = (randint(0,n-1),randint(0,n-1))
        if not(a in corres):
            c += 1
            corres += [a]
    m = [[0 for i in range(n)] for i in range(n)]
    for i in corres :
        a,b = i 
        m[a][b] = 7
    m = encercle(m)
    predr = [[0 for i in range(len(m[0]))] for j in range(len(m))]
    return [[m[j][i] for i in range(1,len(m[0])-1)] for j in range(1,len(m)-1)]

########### Anciennes fonctions récursives

def parcours_rec(m) :

    """Renvoie la liste des parcours de chacun des noeuds de m ainsi  que la liste des intersections propres à chaque noeud"""

    n,p = len(m),len(m[0])
    global parcoursi
    parcoursi,l= [],[]
    marq = [[False for i in range(p)] for i in range(n)]

    def suivre(i,j,x,y,acc, acci):
        global parcoursi
        if i == x and j == y :
            acc.append((i,j))
            return suivre(i+1,j,i,j,acc,acci)
        marq[i][j] = True
        if (i,j) in acc:
            if (i,j) == acc[0] :
                return [acc,acci]
            else :
                if m[i][j] in [7,8] :
                    acci.append((i,j))
        acc.append((i,j))
        if m[i][j] in [7,8,9,10] and not((i,j) in parcoursi):
            parcoursi.append((i,j))
            marq[i][j] = False
        if m[i][j] in [1,2,7,8]:
            return suivre(((2*i)-x),((2*j)-y),i,j,acc,acci)
        if m[i][j] in [3,9,6] :
            return suivre(i-(j-y),j-(i-x),i,j,acc,acci)
        if m[i][j] in [4,5,10] :
            return suivre(i+(j-y),j+(i-x),i,j,acc,acci)

    for i in range(n) :
        for j in range(p):
            if not(marq[i][j]) and m[i][j] != 0:
                l.append(suivre(i,j,i,j,[],[]))
    return l


def composante_rec(m):

    """Renvoie la liste des parcours avec chiffre de chaque case de chacun des noeuds de m"""

    n,p = len(m),len(m[0])
    global parcoursi
    parcoursi,l = [],[]
    marq = [[False for i in range(p)] for i in range(n)]

    def suivre(i,j,x,y,acc,desc):
        global parcoursi
        if i == x and j == y :
            acc.append((i,j))
            desc.append([(i,j),m[i][j]])
            return suivre(i+1,j,i,j,acc,desc)
        marq[i][j] = True
        if (i,j) in acc:
            if (i,j) == acc[0] :
                return desc
        acc.append((i,j))
        desc.append([(i,j),m[i][j]])
        if m[i][j] in [7,8,9,10] and not((i,j) in parcoursi):
            parcoursi.append((i,j))
            marq[i][j] = False
        if m[i][j] in [1,2,7,8]:
            return suivre(((2*i)-x),((2*j)-y),i,j,acc,desc)
        if m[i][j] in [3,9,6] :
            return suivre(i-(j-y),j-(i-x),i,j,acc,desc)
        if m[i][j] in [4,5,10] :
            return suivre(i+(j-y),j+(i-x),i,j,acc,desc)

    for i in range(n) :
        for j in range(p):
            if not(marq[i][j]) and m[i][j] != 0:
                l.append(suivre(i,j,i,j,[],[]))
    return l


def composanteco_rec(m):

    """Renvoie la liste des parcours de chacun des noeuds de m ainsi  que la liste des contacts entre chaque composante"""

    n,p = len(m),len(m[0])
    global parcoursi
    parcoursi,l,cont= [],[],[]
    marq = [[False for i in range(p)] for i in range(n)]

    def suivre(i,j,x,y,acc,desc):
        global parcoursi
        if i == x and j == y :
            acc.append((i,j))
            desc.append([(i,j),m[i][j]])
            return suivre(i+1,j,i,j,acc,desc)
        marq[i][j] = True
        if (i,j) in acc:
            if (i,j) == acc[0] :
                return desc
        acc.append((i,j))
        desc.append([(i,j),m[i][j]])
        if m[i][j] in [7,8,9,10] and not((i,j) in parcoursi):
            parcoursi.append((i,j))
            marq[i][j] = False
        if m[i][j] in [1,2,7,8]:
            return suivre(((2*i)-x),((2*j)-y),i,j,acc,desc)
        if m[i][j] in [3,9,6] :
            return suivre(i-(j-y),j-(i-x),i,j,acc,desc)
        if m[i][j] in [4,5,10] :
            return suivre(i+(j-y),j+(i-x),i,j,acc,desc)

    for i in range(n) :
        for j in range(p):
            if not(marq[i][j]) and m[i][j] != 0:
                L = suivre(i,j,i,j,[],[])
                case = preml(L)
                for k in parcoursi :
                    a,b = k 
                    if not(marq[a][b]) : 
                        cont.append((a,b))
                l.append([L,inter(case,cont)])
    return l


def alterne_rec(m):

    """alterne le noeud m"""

    n,p = len(m),len(m[0])
    global parcoursi
    parcoursi= []
    marq = [[False for i in range(p)] for i in range(n)]

    def suivre(i,j,x,y,acc,der):
        global parcoursi
        if i == x and j == y :
            acc.append((i,j))
            suivre(i+1,j,i,j,acc,der)
        marq[i][j] = True
        if (i,j) in acc:
            if (i,j) == acc[0] :
                return
        acc.append((i,j))
        if m[i][j] in [7,8,9,10] : 
            if not((i,j) in parcoursi):
                parcoursi.append((i,j))
                marq[i][j] = False
                if m[i][j] in [7,8] :
                    if der : 
                        m[i][j] = 7
                        der = False
                    else : 
                        m[i][j] = 8
                        der = True
            elif m[i][j] in [7,8]:
                der = not(der)
        if m[i][j] in [1,2,7,8]:
            suivre(((2*i)-x),((2*j)-y),i,j,acc,der)
        if m[i][j] in [3,9,6] :
            suivre(i-(j-y),j-(i-x),i,j,acc,not(der))
        if m[i][j] in [4,5,10] :
            suivre(i+(j-y),j+(i-x),i,j,acc,not(der))

    for i in range(n) :
        for j in range(p):
            if not(marq[i][j]) and m[i][j] != 0:
                suivre(i,j,i,j,[],True)
    return m

########### Fonctions d'affichages


def montre(m,b = True) :
    
    """affiche la matrice du noeud plus lisiblement"""
    
    if b :
        for i in m :
            c =""
            for j in i :
                c += " "
                if j != 10 :
                    c += str(j)
                    c +=" "
                else : 
                    c += "10"
            print(c)
    else : 
        for i in m :
            c =""
            for j in i :
                c += " "
                if j != 10 and j != 0 :
                    c += str(j)
                    c +=" "
                elif j == 0 :
                    c += "  "
                else : 
                    c += "10"
            print(c)


def nec(m):

    """affiche le noeud et le digramme de corde"""

    affiche(m,True,0.075,True)
    corde(m,True,True)
    pylab.show()


def necp(m):

    """affiche diagramme de corde, noeud decomposé et mot de gauss décomposé"""

    coprem(m,True,True)
    corde(m,True,True)
    coprem(mgauss(m)[0],True,True)
    pylab.show()


def fullgraph(m):

    """donne tout et plus encore"""

    def suivant(l,i,boule = False) :
        k = 0
        b = False
        for j in range(len(l)) : 
            if l[j] == i :
                if b : 
                    if boule :
                        return k,j
                    return j
                else : 
                    b = True
                    k = j
    affiche(m,True,0.075,True)
    m = reduit(m)
    gauss = coprem(mgauss(m)[0],False,True,True)
    comp,couleur,vectcouleur = coprem(m,False,True,True)
    F = pylab.figure().gca()
    G = pylab.figure().gca()
    j = -1
    g = mgauss(m)[0]
    N = len(g)
    G.add_patch(pylab.Circle([0,0],radius = 1,fill = False))
    for i in range(len(comp)) :
        if len(comp[i]) > 1 :
            j += 1
            F.add_patch(pylab.Circle([2*j,0],radius = 1, fill = False,color = vectcouleur[couleur[i]] ))
            n = len(gauss[j])
            acc = [] 
            for k in range(n) :
                if not(gauss[j][k] in acc) :
                    c_1,c_2 = k*2*pi/n,(suivant(gauss[j],gauss[j][k]))*2*pi/n
                    F.add_patch(pylab.Polygon([(2*j+cos(c_1),sin(c_1)),(2*j+cos(c_2),sin(c_2))],color = vectcouleur[couleur[i]] ))
                    c_1,c_2 = suivant(g,gauss[j][k],True)
                    G.add_patch(pylab.Polygon([(cos(c_1*2*pi/N),sin(c_1*2*pi/N)),(cos(c_2*2*pi/N),sin(c_2*2*pi/N))],color = vectcouleur[couleur[i]] ))
                    acc += [gauss[j][k]]
        else :
            c_1,c_2 = suivant(g,gauss[i][0],True)
            G.add_patch(pylab.Polygon([(cos(c_1*2*pi/N),sin(c_1*2*pi/N)),(cos(c_2*2*pi/N),sin(c_2*2*pi/N))]))     
    F.axis('scaled')
    G.axis('scaled')
    coprem(m,True,True)
    pylab.show()


def corde(m,boule = True,multi = False) : 

    """affiche le diagramme de corde de chacun des noeuds de m, si boule comprend m comme une liste de mot de gauss,  si multi calcule la figure sans l'afficher"""

    def suivant(l,i) :
        b = False
        for j in range(len(l)) : 
            if l[j] == i :
                if b : 
                    return j
                else : 
                    b = True
    if type(m[0]) == list:
        if boule :
            gauss = mgauss(m)
        else : 
            gauss = m
    else :
        gauss = [m]
    F = pylab.figure().gca()
    for i in range(len(gauss)) :
        F.add_patch(pylab.Circle([2*i,0],radius = 1, fill = False))
        n = len(gauss[i])
        acc = [] 
        for k in range(n) :
            if not(gauss[i][k] in acc) :
                c_1,c_2 = k*2*pi/n,(suivant(gauss[i],gauss[i][k]))*2*pi/n
                F.add_patch(pylab.Polygon([(2*i+cos(c_1),sin(c_1)),(2*i+cos(c_2),sin(c_2))]))
                acc += [gauss[i][k]]
    pylab.axis('scaled')
    if not(multi) :
        pylab.show()


def affgraphe(g,epsi=0.025) : 

    """affiche le graphe g"""

    marq = [[False for i in range(len(g))] for i in range(len(g))]
    F = pylab.gca()
    for i in range(len(g)) :
        F.add_patch(pylab.Circle([i,i],radius = 3*epsi, fill = False))
        for j in range(len(g[i])) :
            if not(marq[i][j]):
                k = g[i][j]
                if k < i :
                    y,k = k,i
                else : 
                    y = i
                marq[i][j],marq[j][i] = True,True
                if i in g[k]:
                    F.add_patch(pylab.Polygon([(y-epsi,y-epsi),(y-epsi,k+epsi),(k+epsi,k+epsi),(k+epsi,k-epsi),(y+epsi,k-epsi),(y+epsi,y-epsi)]))
                else :
                    F.add_patch(pylab.Polygon([(y-epsi,y-epsi),(k+epsi,y-epsi),(k+epsi,k+epsi),(k-epsi,k+epsi),(k-epsi,y+epsi),(y-epsi,y+epsi)]))
    pylab.axis('scaled')
    pylab.show()


def affiche(m,boule = True, epsi = 0.075, multi = False) :

    """affiche un noeud sur pylab, si boule pas de cadrillage, epsi taille du trait, si multi calcule la figure sans l'afficher"""

    F = pylab.figure().gca()
    aff = [(lambda x,y : pylab.Polygon([(x,y+1/2-epsi),(x,y+1/2+epsi),(x+1,y+1/2+epsi),(x+1,y+1/2-epsi)],fill=True)),
           (lambda x,y : pylab.Polygon([(x+1/2-epsi,y),(x+1/2+epsi,y),(x+1/2+epsi,y+1),(x+1/2-epsi,y+1)],fill=True)),
           (lambda x,y :pylab.Polygon([(x+1/2-epsi,y),(x+1/2+epsi,y),(x+1/2+epsi,y+1/2-epsi),(x+1,y+1/2-epsi),(x+1,y+1/2+epsi),(x+1/2-epsi,y+1/2+epsi)],fill=True)),
           (lambda x,y : pylab.Polygon([(x,y+1/2-epsi),(x,y+1/2+epsi),(x+1/2+epsi,y+1/2+epsi),(x+1/2+epsi,y),(x+1/2-epsi,y),(x+1/2-epsi,y+1/2-epsi)],fill=True)),
           (lambda x,y :pylab.Polygon([(x+1/2-epsi,y+1),(x+1/2+epsi,y+1),(x+1/2+epsi,y+1/2+epsi),(x+1,y+1/2+epsi),(x+1,y+1/2-epsi),(x+1/2-epsi,y+1/2-epsi)],fill=True)),
           (lambda x,y :pylab.Polygon([(x+1/2-epsi,y+1),(x+1/2+epsi,y+1),(x+1/2+epsi,y+1/2-epsi),(x,y+1/2-epsi),(x,y+1/2+epsi),(x+1/2-epsi,y+1/2+epsi)],fill=True)),
           (lambda x,y : pylab.Polygon([(x+1/2-epsi,y),(x+1/2+epsi,y),(x+1/2+epsi,y+1/2-3*epsi),(x+1/2-epsi,y+1/2-3*epsi),(x+1/2-epsi,y+1/2-epsi),(x,y+1/2-epsi),(x,y+1/2+epsi),(x+1/2-epsi,y+1/2+epsi),(x+1/2-epsi,y+1),(x+1/2+epsi,y+1),(x+1/2+epsi,y+1/2+3*epsi),(x+1/2-epsi,y+1/2+3*epsi),(x+1/2-epsi,y+1/2+epsi),(x+1,y+1/2+epsi),(x+1,y+1/2-epsi),(x+1/2-epsi,y+1/2-epsi)],fill=True)),
            (lambda x,y :pylab.Polygon([(x+1/2-epsi,y),(x+1/2+epsi,y),(x+1/2+epsi,y+1/2-epsi),(x+1,y+1/2-epsi),(x+1,y+1/2+epsi),(x+1/2+3*epsi,y+1/2+epsi),(x+1/2+3*epsi,y+1/2-epsi),(x+1/2+epsi,y+1/2-epsi),(x+1/2+epsi,y+1),(x+1/2-epsi,y+1),(x+1/2-epsi,y+1/2-epsi),(x,y+1/2-epsi),(x,y+1/2+epsi),(x+1/2-3*epsi,y+1/2+epsi),(x+1/2-3*epsi,y+1/2-epsi),(x+1/2-epsi,y+1/2-epsi)],fill=True)),
            (lambda x,y :pylab.Polygon([(x,y+1/2-epsi),(x,y+1/2+epsi),(x+1/2-epsi,y+1),(x+1/2+epsi,y+1)],fill=True)),
            (lambda x,y :pylab.Polygon([(x+1/2-epsi,y),(x,y+1/2-epsi),(x,y+1/2+epsi),(x+1/2+epsi,y)],fill=True))]
    n,p = len(m),len(m[0])
    for i in range(n):
        for j in range(p) :
            if boule :
                F.add_patch(pylab.Rectangle((j,-i),1,1,fill = False))
            if m[i][j] != 0 :
                F.add_patch(aff[m[i][j]-1](j,-i))
            if m[i][j] == 9:
                F.add_patch(pylab.Polygon([(j+1/2+epsi,-i),(j+1,-i+1/2-epsi),(j+1,-i+1/2+epsi),(j+1/2-epsi,-i)],fill = True))
            if m[i][j] == 10 :
                F.add_patch(pylab.Polygon([(j+1/2-epsi,-i+1),(j+1/2 + epsi, -i+1),(j+1, -i+1/2+epsi),(j+1, -i+1/2-epsi)],fill = True))

    pylab.axis('scaled')
    if not(multi):
        pylab.show()


def affprem(m,couleur,boule = True, epsi = 0.075, multi = False) :

    """affiche un noeud selon les colorations de la fonction couleur, si boule pas de cadrillage, epsi taille du trait, si multi calcule la figure sans l'afficher"""

    
    F = pylab.figure().gca()
    aff = [(lambda x,y : pylab.Polygon([(x,y+1/2-epsi),(x,y+1/2+epsi),(x+1,y+1/2+epsi),(x+1,y+1/2-epsi)],fill=True,color=couleur[-y][x])),
           (lambda x,y : pylab.Polygon([(x+1/2-epsi,y),(x+1/2+epsi,y),(x+1/2+epsi,y+1),(x+1/2-epsi,y+1)],fill=True,color=couleur[-y][x])),
           (lambda x,y :pylab.Polygon([(x+1/2-epsi,y),(x+1/2+epsi,y),(x+1/2+epsi,y+1/2-epsi),(x+1,y+1/2-epsi),(x+1,y+1/2+epsi),(x+1/2-epsi,y+1/2+epsi)],fill=True,color=couleur[-y][x])),
           (lambda x,y : pylab.Polygon([(x,y+1/2-epsi),(x,y+1/2+epsi),(x+1/2+epsi,y+1/2+epsi),(x+1/2+epsi,y),(x+1/2-epsi,y),(x+1/2-epsi,y+1/2-epsi)],fill=True,color = couleur[-y][x])),
           (lambda x,y :pylab.Polygon([(x+1/2-epsi,y+1),(x+1/2+epsi,y+1),(x+1/2+epsi,y+1/2+epsi),(x+1,y+1/2+epsi),(x+1,y+1/2-epsi),(x+1/2-epsi,y+1/2-epsi)],fill=True,color = couleur[-y][x])),
           (lambda x,y :pylab.Polygon([(x+1/2-epsi,y+1),(x+1/2+epsi,y+1),(x+1/2+epsi,y+1/2-epsi),(x,y+1/2-epsi),(x,y+1/2+epsi),(x+1/2-epsi,y+1/2+epsi)],fill=True,color = couleur[-y][x])),
           (lambda x,y : pylab.Polygon([(x+1/2-epsi,y),(x+1/2+epsi,y),(x+1/2+epsi,y+1/2-3*epsi),(x+1/2-epsi,y+1/2-3*epsi),(x+1/2-epsi,y+1/2-epsi),(x,y+1/2-epsi),(x,y+1/2+epsi),(x+1/2-epsi,y+1/2+epsi),(x+1/2-epsi,y+1),(x+1/2+epsi,y+1),(x+1/2+epsi,y+1/2+3*epsi),(x+1/2-epsi,y+1/2+3*epsi),(x+1/2-epsi,y+1/2+epsi),(x+1,y+1/2+epsi),(x+1,y+1/2-epsi),(x+1/2-epsi,y+1/2-epsi)],fill=True,color = couleur[-y][x])),
            (lambda x,y :pylab.Polygon([(x+1/2-epsi,y),(x+1/2+epsi,y),(x+1/2+epsi,y+1/2-epsi),(x+1,y+1/2-epsi),(x+1,y+1/2+epsi),(x+1/2+3*epsi,y+1/2+epsi),(x+1/2+3*epsi,y+1/2-epsi),(x+1/2+epsi,y+1/2-epsi),(x+1/2+epsi,y+1),(x+1/2-epsi,y+1),(x+1/2-epsi,y+1/2-epsi),(x,y+1/2-epsi),(x,y+1/2+epsi),(x+1/2-3*epsi,y+1/2+epsi),(x+1/2-3*epsi,y+1/2-epsi),(x+1/2-epsi,y+1/2-epsi)],fill=True,color = couleur[-y][x])),
            (lambda x,y :pylab.Polygon([(x,y+1/2-epsi),(x,y+1/2+epsi),(x+1/2-epsi,y+1),(x+1/2+epsi,y+1)],fill=True,color = couleur[-y][x][1])),
            (lambda x,y :pylab.Polygon([(x+1/2-epsi,y),(x,y+1/2-epsi),(x,y+1/2+epsi),(x+1/2+epsi,y)],fill=True,color = couleur[-y][x][0]))]
    t0 = clock()
    n,p = len(m),len(m[0])
    for i in range(n):
        for j in range(p) :
            if boule :
                F.add_patch(pylab.Rectangle((j,-i),1,1,fill = False))
            if m[i][j] != 0 :
                F.add_patch(aff[m[i][j]-1](j,-i))
            if m[i][j] == 9:
                F.add_patch(pylab.Polygon([(j+1/2+epsi,-i),(j+1,-i+1/2-epsi),(j+1,-i+1/2+epsi),(j+1/2-epsi,-i)],fill = True,color = couleur[i][j][0]))
            if m[i][j] == 10 :
                F.add_patch(pylab.Polygon([(j+1/2-epsi,-i+1),(j+1/2 + epsi, -i+1),(j+1, -i+1/2+epsi),(j+1, -i+1/2-epsi)],fill = True,color = couleur[i][j][1]))
    if temps:
        print("matrice",clock()-t0)
    pylab.axis('scaled')
    if not(multi) :
        pylab.show()
