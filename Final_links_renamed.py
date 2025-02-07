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

######### Focntions finales 

#Les fonctions alea_entrelacs et alea_extraction génèrent aléatoirement des noeuds alternés
#equiv teste l'appartenance de deux noeuds a la même classe d'équivalence pour la relation du cardinal de décomposition
#colore affiche le noeud simplifié visuellement dont les composantes premières sont coloriées
#decompose affiche la coloration de m, la decomposition de son diagramme de corde et m

#Ce code est issu d'un TIPE de prépa et l'affiche au moins, n'est probablement pas du tout optimisé.


def alea_liaison(n,p):

	"""Noeud alterné de taille (n,p) avec méthode d'hyper-connexité"""
	
	return alea_noeud_liaison_alterne(n,p)
	
	
def alea_extraction(n,p):

	"""Noeud alterné de taille (n,p) avec méthode d'extraction"""
	
	return alea_noeud_extraction_alterne(n,p)

	
def equiv(a,b):

	"""True si a et b equivalents au sens du cardinal de decomposition"""
	
	return equiv_card(a,b)

	
def colore(m):

	"""colorie les composantes premieres de m"""
	
	visuel_premier(m)
	

def decompose(m):

	"""affiche le noeud original m, m coloré, son diagramme de corde coloré et décomposé"""
	
	aff_decomposition(m)

########################################################################

def t(f,x):

    """temps d'execution de f(x)"""

    t_0 = clock()
    f(x)
    return clock()-t_0

global temps
temps = False

####### Génération de noeuds

def entoure(M) :
    
    """entoure la matrice de 0"""

    m = [[0 for i in range(len(M[0])+2)]]
    for i in M : 
        m += [[0]+i+[0]]
    m += [[0 for i in range(len(M[0])+2)]]
    return m


def est_entrelacs(m) :

    """vérifie si m est un entrelacs"""

    n,p = len(m),len(m[0])
    M = entoure(m)
    b = False
    for i in range(1,n) :
        for j in range(1,p) :
            k = M[i][j]
            if not(( M[i][j-1] in correct[k][0]) and (M[i][j+1] in correct[k][1]) and (M[i-1][j] in correct[k][2]) and (M[i+1][j] in correct[k][3])):
                return False
            else :
                b = True
    return b


def inter(a,b,c=[],d=[],boule = False):

    """réalise l'intersection de a et b et c et d si non vides
	Si boule intersection de listes triées"""
    
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


def inter_sous_listes(l) :

    """intersection des listes de l"""
    
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


def alea_entrelacs(n,p=0) : 

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


def recompose(m,l):

    """crèe une composante suivant l un parcours"""
    
    for i in range(1,len(l)-1):
        a,b = l[i]
        L = prive([(a+1,b),(a-1,b),(a,b+1),(a,b-1)],[l[i+1]])
        (a1,b1),(a2,b2),(a3,b3) = L[0],L[1],L[2]
        if m[a][b] == 0:
            m[a][b] = inter(correct[m[a1][b1]][direction(m,a,b,a1,b1)],correct[m[a2][b2]][direction(m,a,b,a2,b2)],correct[m[a3][b3]][direction(m,a,b,a3,b3)])[0]
        elif m[a][b] in [1,2] :
            m[a][b] = randint(7,8)
        elif m[a][b] in [3,6]:
            m[a][b] = 9
        elif m[a][b] in [4,5]:
            m[a][b] = 10
    return m


def alea_noeud_extraction(n,p=0):

    """noeud dans une matrice de taille n,p """

    m = alea_entrelacs(n,p)
    t = [[0 for i in range(len(m[0])+2)] for i in range(len(m)+2)]
    l = max_liste([i[0] for i in parcours_interne(m)])
    l = l + [l[0]]
    for i in range(len(l)):
        a,b = l[i]
        l[i] = (a+1,b+1)
    a,b=l[0]
    t[a][b] = 3
    t = recompose(t,l)
    return [[t[i][j] for j in range(1,len(t[0])-1)] for i in range(1,len(t)-1)] 


def alea_noeud_extraction_alterne(n,p=0) :

    """noeud  alterné de taille n,p"""

    return alterne(alea_noeud_extraction(n,p))


def reliable(a,b):

    """True si a et b ont un élément en commun"""

    for i in a : 
        for j in b : 
            if i ==j :
                return True
    return False


def reliable_triee(a,b):

    """True si a et b triées ont un élément en commun"""
    
    if a == [] or b == []:
        return False
    i,j = 0,0
    while i < len(a) and j < len(b) :
        if a[i] == b[j]:
            return True
        if ordre_lexico(a[i],b[j]) :
            j += 1 
        else :
            i += 1
    return False


def graphe_hyper(k,L) :

    """graphe d'hyper-connexite et marquage de la composante connexe de la kieme liste de L"""
    
    v = [False for i in range(len(L))]
    v[k] = True
    g = [[] for i in range(len(L))]
    for i in range(len(L)):
        for j in range(len(L)) :
            if i != j:
                if not(i in g[j]):
                    if reliable_triee(L[i],L[j]):
                        g[i] += [j]
                        g[j] += [i]
    def parcours_interne(i):
        for j in range(len(g[i])):
            if not(v[g[i][j]]):
                v[g[i][j]]=True
                parcours_interne(g[i][j])
    parcours_interne(k)
    return v,g


def alea_noeud_liaison(n,p=0):

    """noeud aléatoire de taille n,p"""
    
    return lie(alea_entrelacs(n,p)) 


def lie(m) :

    """relie correctement les composantes connexes reliables de m"""
   
    t0 = clock()
    l = parcours_hyper_connexite(m) 
    if len(l) != 1:
        ind,lenm = 0,len(l[0][0])
        for i in range(1,len(l)) :
            if len(l[i][0]) > lenm:
                ind = i
                lenm = len(l[i][0])
        supp = []
        sec = [trif(i) for i in second_liste(l)]
        v,g = graphe_hyper(ind,sec)
        N = len(g)
        compo = [i for i in range(N)]
        for i in range(N):
            if v[i] and g[i] != []:
                rel = [compo[i]]
                for j in range(len(g[i])):
                    k = g[i][j]
                    pos = inter_couple(sec[i],sec[k])
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
                        g[h] = prive(g[h],connecte)
                    else : 
                        if v[h] :
                           g[h] = une_occurence(g[h],connecte)
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


def alea_noeud_liaison_alterne(n,p=0):

    """noeud alterné a un seul brin dans une matrice de taille n,p ou n,n si p = 0"""

    return alterne(alea_noeud_liaison(n,p))


def alea_noeud_premier(n,p=0):

    """noeud premier de taille n"""

    global m
    m = reduit(aualean(n,p))
    l = decomposition_premier(m)
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
        l = decomposition_premier(m)
    if len(l) in[0,1] :
        return m
    parc = parcours_simple(m)[0]
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
                m = supprime_brin(m,a,b,direction(m,u,v,a,b))
            if not(boule) and marq[a][b]:
                m = relie(m,a,b,e,f)
            boule = marq[a][b]
            if marq[a][b] :
                e,f = a,b
        if not(boule) :
            if m[a][b] in [9,10] : 
                c,d = parc[i-1]
                h = direction(m,a,b,c,d)
                if m[a][b] == 9 :
                    m[a][b] = 6-3*(h%2)
                else : 
                    m[a][b] = 4 + int(((h%3)+1)/2)
            else : 
                m[a][b] = 0
    return reduit(m)
        

########### Fonctions de parcours

def parcours_simple(m) :

    """liste des parcours de chacun des noeuds de m"""

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


def parcours_interne(m) :

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


def parcours_case(m):

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


def parcours_hyper_connexite(m):

    """Renvoie la liste des parcours_interne de chacun des noeuds de m ainsi  que la liste des contacts entre chaque composante"""
    
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

def max_liste(l):

    """liste de plus grande taille de l"""

    L = []
    for i in l :
        if i != None :
            L += [len(i)]
        else : 
            L += [0]
    return l[premant(max(L),L)]


def premier_liste(l) :

    """liste des premiers éléments de chaque couple de l"""

    L = []
    for i in l :
        L += [i[0]]
    return L


def second_liste(l) :

    """liste des second éléments de chaue couple de l"""

    L = []
    for i in l :
        L += [i[1]]
    return L


def ordre_lexico(x,y):

    """ x si plus grand que y pour ordre lexicographique sur les couples"""

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
        if ordre_lexico(a[0],b[0]):
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


def inter_couple(a,b):

    """intersection de a et b listes de couples triées"""

    i,j = 0,0
    L = []
    while i < len(a) and j < len(b) :
        if a[i] == b[j] :
            L += [a[i]]
            i += 1
            j += 1
        elif ordre_lexico(a[i],b[j]):
            j += 1
        else :
            i += 1
    return L


def rev(l):

    """pareil que l.reverse() sans toucher la liste"""

    return [l[i] for i in range(len(l)-1,-1,-1)]


def prive(a,b):

    """liste a sans les éléments de b"""

    if a == [] :
        return a 
    else :
        return [i for i in a if not(i in b)]


def une_occurence(a,b):

    """liste a avec un seul des éléments de b quand c'est possible, a sinon"""

    pos = inter(a,b)
    if len(pos) <= 1 :
        return a
    else : 
        c = pos[randint(0,len(pos)-1)]
        return prive(a,sppr_de(c,b))


def sppr_de(x,l):

    """liste l sans l'élément x"""

    return [i for i in l if i != x]

########## Outils sur les noeuds 

def mot_gauss(m,boule = False,decomposition_premier = False):

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
        print("mot_gauss",clock()-t0)
    if decomposition_premier :
        return L,[[corr[i][j],(i,j)] for i in range(n) for j in range(p)]
    return L


def reduction_taille(m):

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


def card_decomp(m):

    """liste de 2-listes contenant le nombre de croisements d'une d'une composante première et le nombre de composantes de ce nombre de croisements"""

    l = decomposition_premier(detorsade(mot_gauss(m)[0]))
    ma = max([len(i) for i in l])
    L = [[i,0] for i in range(ma+1)]
    for i in l : 
        if len(i) >1 : 
            L[len(i)][1] += 1 
    return [i for i in L if i[1] >0]



def equiv_card(m,M):

    """False si m et M noeuds alternés sont non équivalents pour le cardinal de decomposition 
	True sinon"""

    return card_decomp(m) == card_decomp(M)    

######## Réduction

def encercle(t):
    
    """entoure t d'un noeud trivial"""

    m = deepcopy(t)
    for i in range(len(m)):
        m[i] = [2]+m[i]+[2]
    return [[3]+[1 for i in range(len(m[0])-2)]+[4]]+m+[[5]+[1 for i in range(len(m[0])-2)]+[6]]


def ouverture_hauteur(m,x,y):

    """-1 si ouvert en haut, 1 en bas, 0 sinon"""

    if x > 0 and not(m[x][y] in correct[m[x-1][y]][3] and m[x-1][y] in correct[m[x][y]][2]):
        return -1
    if x < len(m) and not(m[x][y] in correct[m[x+1][y]][2] and m[x+1][y] in correct[m[x][y]][3]):
        return 1
    return 0


def ouverture_cote(m,x,y) :

    """-1 si ouvert à gauche, 1 à droite, 0 sinon"""

    if y > 0 and not(m[x][y] in correct[m[x][y-1]][1] and m[x][y-1] in correct[m[x][y]][0]):
        return -1
    if y < len(m[0]) and not(m[x][y] in correct[m[x][y+1]][0] and m[x][y+1] in correct[m[x][y]][1]):
        return 1
    return 0


def ouverture(m,x,y) : 

    """couple de 1,0,-1 indiquant une ouverture de m[x][y]"""
    
    return (ouverture_hauteur(m,x,y),ouverture_cote(m,x,y))


def direction(m,a,b,x,y):
    
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
    di,dj,ai,aj = di+ouverture_hauteur(m,di,dj),dj+ouverture_cote(m,di,dj),ai+ouverture_hauteur(m,ai,aj),aj+ouverture_cote(m,ai,aj)
    if boule :
        return join_brin(m,di,dj,ai,aj)
    else : 
        return join_rec(m,di,dj,ai,aj)


def join_brin(m,di,dj,ai,aj):
    
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
    m = recompose(m,chem)
    return m


def supprime_brin(m,di,dj,direc,parc = [],num = 0):

    """supprime les éléments de m après (di,dj) dans la direction direc jusqu'au croisemet suivant 0 gauche, 1 droite, 2 dessus et 3 dessous"""
    
    if parc == [] :
        parc = parcours_interne(m)[num][0]
    parc = parc*2 + rev(parc)*2
    boule = False
    for i in range(len(parc)):
        a,b = parc[i]
        if boule :
            if m[a][b] in [9,10] : 
                c,d = parc[i-1]
                h = direction(m,a,b,c,d)
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
            if direction(m,c,d,a,b) == direc:
                boule = True
    return m


def detorsade(m,num= 0,l = []):

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
            l = parcours_interne(m)[num]
        mot = mot_gauss(m,True)[0]
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
                dep = direction(m,a,b,e,f)%2
                arr = direction(m,a,b,g,h)%2
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
                dep = direction(m,a,b,e,f)%2
                arr = direction(m,a,b,g,h)%2
                if (dep,arr) == (0,0) or (dep,arr) == (1,1):
                    m[a][b] = 10
                else :
                    m[a][b] = 9
                mot.remove(mot[len(mot)-1])
                mot.remove(mot[len(mot)-1])
        return m


def spprime_brin_indice(m,i,parc):

    """supprime les éléments de m après (di,dj) dans la direction direc jusqu'au croisemet et retourne ses indices suivant 0 gauche, 1 droite, 2 dessus et 3 dessous"""
    
    boule = True
    while boule:
        a,b = parc[i]
        if m[a][b] in [9,10] : 
            c,d = parc[i-1]
            h = direction(m,a,b,c,d)
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
    m = encercle(detorsade(deepcopy(m)))
    if temps :
        print("detorsade",clock() -t)
    if l == []:
        l = parcours_interne(m)[num+1]
    if l[1] == [] : 
        return [[3,4],[5,6]]
    else : 
        mot = mot_gauss(m,True)[num+1]
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
                supm[a][b][direction(m,c,d,a,b)]=i
                c,d = parc[i-1]
                supm[a][b][direction(m,c,d,a,b)]=3*len(parc)-1-i
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
                    m,e,f = spprime_brin_indice(m,(supm[a][b][direction(m,c,d,a,b)])+1,parc)
                    t00 += clock()-t
                    t =clock()
                    m = relie(m,a,b,e,f,True)
                    t11 += clock()-t
        if temps :
            print("reducted",clock()-t2)
            print("   supp",t00)
            print("   relie",t11)
        return reduction_taille([[m[i][j] for j in range(1,len(m[0])-1)] for i in range(1,len(m)-1)])

########### Décomposition

def emboitement(m):
    
    """emboitement des sommet du mot de gauss de m ou du mot de gauss m"""

    if type(m[0]) == list :
        m = mot_gauss(m)[0]
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


def graphe_emboitement_o(m):
    
    """graphe d'emboitement du mot de gauss (de) m"""

    if type(m[0]) == list :
        m = mot_gauss(m)[0]
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
    return g


def graphe_emboitement_no(m):
    
    """graphe d'emboitement non orienté du mot de gauss (de) m"""

    if type(m[0]) == list :
        m = mot_gauss(m)[0]
    n = int(len(m)/2)
    g = [[] for i in range(n)]
    t = [(-1) for i in range(n)]
    for i in range(len(m)):
        if t[m[i]] == -1 :
            t[m[i]] = i
        else : 
            dej = [False for j in range(n)]
            acc = []
            pos = t[m[i]]
            t[m[i]] = i-t[m[i]]-1
            if len(m)-t[m[i]]-2 < t[m[i]]:
                for j in range(pos):
                    if not(dej[m[j]]) : 
                        dej[m[j]] = True
                        acc += [m[j]]
                    else :
                        dej[m[j]] = False
                for j in range(i+1,len(m)):
                    if not(dej[m[j]]) : 
                        dej[m[j]] = True
                        acc += [m[j]]
                    else :
                        dej[m[j]] = False
            else : 
                for j in range(pos+1,i):
                    if not(dej[m[j]]) : 
                        dej[m[j]] = True
                        acc += [m[j]]
                    else :
                        dej[m[j]] = False
            for j in acc : 
                if dej[j] : 
                    g[m[i]] += [j]
    return g


def co_graphe(g):
    
    """co-graphe de g"""

    t = [[] for i in range(len(g))]
    for i in range(len(g)):
        for j in g[i] : 
            t[j] += [i] 
    return t


def extrait_connexe_no(g):

    """composantes connexes de g non orienté"""

    l = []
    v = [False for i in range(len(g))]
    def parcours_interne(i):
        l[0] += [i]
        v[i] = True
        for j in g[i] : 
            if not(v[j]) :
                parcours_interne(j)
    for i in range(len(v)):
        if not(v[i]):
            l = [[]]+l
            parcours_interne(i)
    return l


def extrait_connexe_o(g):
    
    """composantes fortement connexes de g non orienté"""

    v = [0 for i in range(len(g))]
    dec = [[]]
    def parcours_interne(i):
        v[i] = -1
        for j in range(len(g[i])):
            if v[g[i][j]] == 0 :
                parcours_interne(g[i][j])
        dec[0] = [i]+dec[0]
    for i in range(len(g)):
        if v[i] == 0 :
            parcours_interne(i) 
    t = co_graphe(g)
    c = [0]
    def parcoursr(i) :
        v[i] = c[0]
        for j in range(len(t[i])):
            if v[t[i][j]] == -1 :
                parcoursr(t[i][j])
    i = 0
    while i < len(dec[0]) :
        if v[dec[0][i]] == -1:
            parcoursr(dec[0][i])
            c[0] += 1    
        i += 1
    l = [[] for i in range(max(v)+1)]
    for i in range(len(v)):
        l[v[i]] += [i]
    return l


def decomposition_premier(m):

    """si m mot de gauss, sous-mots de gauss, si m noeuds, listes des sous noeuds données par intersections"""

    if type(m[0]) == list :
        t0 = clock() 
        l,corr = mot_gauss(m,False,True)
        l = l[0]
        d = dict(corr)
        if l != [] :
            L = extrait_connexe_no(graphe_emboitement_no(l))
        else : 
            L = []
        for i in range(len(L)):
            for j in range(len(L[i])):
                L[i][j] = d[L[i][j]]
        if temps :
            print("decomposition_premier",clock()-t0)
        return L
    else : 
        return extrait_connexe_no(graphe_emboitement_no(m))


def visuel_premier(m,boule = True,multi = False,fg = False):

    """affiche la coloration des composantes premières de m réduit ou sous diagramme si m mot, si boule reduit m, si multi calcule la figure sans l'afficher, fg pour aff_decomposition"""

    if type(m[0]) == list :
        if boule :
            m = reduit(m)
        if len(m) ==2 :
            affiche(m)
            return
        parc,comp = parcours_simple(m)[0],decomposition_premier(m)
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
                                k[0] = (k[0]+1)%7
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
                    if direction(m,a,b,x1,y1) == 2 or direction(m,a,b,x2,y2) == 2:
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
        aff_premier(m,couleur,True,0.075,multi)
    else :
        m = detorsade(m)
        if m != [] :
            l = decomposition_premier(m)
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
                return L,ism
            corde(L,False,multi)
        else : 
            corde([[3,4],[5,6]])

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


def noeud_corde(m):

    """affiche le noeud et le digramme de corde"""

    affiche(m,True,0.075,True)
    corde(m,True,True)
    pylab.show()


def noeud_corde_decompose(m):

    """affiche diagramme de corde, noeud decomposé et mot de gauss décomposé"""

    visuel_premier(m,True,True)
    corde(m,True,True)
    visuel_premier(mot_gauss(m)[0],True,True)
    pylab.show()


def aff_decomposition(m):

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
    gauss,ism = visuel_premier(mot_gauss(m)[0],False,True,True)
    comp,couleur,vectcouleur = visuel_premier(m,False,True,True)
    F = pylab.figure().gca()
    G = pylab.figure().gca()
    j = -1
    g = mot_gauss(m)[0]
    N = len(g)
    G.add_patch(pylab.Circle([0,0],radius = 1,fill = False))
    for i in range(len(comp)) :
        if len(comp[i]) > 1:
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
    for i in ism:
        c_1,c_2 = suivant(g,i,True)
        G.add_patch(pylab.Polygon([(cos(c_1*2*pi/N),sin(c_1*2*pi/N)),(cos(c_2*2*pi/N),sin(c_2*2*pi/N))],color = 'black' ))
    F.axis('scaled')
    G.axis('scaled')
    visuel_premier(m,True,True)
    pylab.show()


def aff_isole(m):

    """donne tout et plus encore"""

    m = reduit(m)
    comp,couleur,vectcouleur = visuel_premier(m,False,True,True)
    for i in range(len(comp)):
        if len(comp)>1 : 
            t = isole(deepcopy(m),i,comp)
            col = [[ vectcouleur[couleur[i]] for k in range(len(t[0]))] for k in range(len(t))]
            for k in range(len(t)):
                for j in range(len(t[0])):
                    if t[k][j] in [9,10]:
                        col[k][j] = [vectcouleur[couleur[i]],vectcouleur[couleur[i]]]
            aff_premier(t,col,True,0.075,True)
    visuel_premier(m,True,True)
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
            gauss = mot_gauss(m)
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


def affiche(m,boule = True, epsi = 0.075, multi = False) :

    """affiche un noeud sur pylab, si non boule pas de cadrillage, epsi taille du trait, si multi calcule la figure sans l'afficher"""

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


def aff_premier(m,couleur,boule = True, epsi = 0.075, multi = False) :

    """affiche un noeud selon les colorations de la fonction couleur, si non boule pas de cadrillage, epsi taille du trait, si multi calcule la figure sans l'afficher"""

    
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
