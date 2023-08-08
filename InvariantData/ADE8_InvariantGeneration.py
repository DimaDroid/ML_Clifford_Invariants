'''Script to compute varying coxeter elements, using the basis of simple roots'''
#Import libraries
import sys
from itertools import permutations, combinations
from multiprocessing import Pool
from sympy import symbols, sqrt
from sympy.matrices import zeros
from galgebra.ga import Ga

#Set the algebra to compute: [A8, D8, E8] --> [1,2,3]
alg_choice = int(sys.argv[1]) #...generalise this to bash system input later 

#Define parallised function for {W permutation} --> {invariants}
def InvComputation(inputs):
    perm = inputs

    #Setup CA basis
    scoords = (x,y,z,u,a,b,c,d) = symbols('1 2 3 4 5 6 7 8', real=True)
    s = Ga('e',g=[1,1,1,1,1,1,1,1],coords=scoords)
    M = s.mv('M','mv',f = True)
    e1, e2, e3, e4, e5, e6, e7, e8 = s.mv()
    
    def CM(simple_roots_4D):
        A = zeros(len(simple_roots_4D), len(simple_roots_4D))
        for i in range(len(simple_roots_4D)):
            for j in range(len(simple_roots_4D)):
                tmp = (simple_roots_4D[i]*simple_roots_4D[j])
                A[i,j]=2*tmp.blade_coefs()[0] 
        return A
    
    #Select here A8, D8, E8, from parameter: alg_choice
    if alg_choice == 1:
        a1, a2, a3, a4 =  1/sqrt(2)*(-e7+e8), 1/sqrt(2)*(e7-e6), 1/sqrt(2)*(e6-e5),  1/sqrt(2)*(e5-e4) 
        a5, a6, a7, a8 = 1/sqrt(2)*(e4-e3), 1/sqrt(2)*(e3-e2), 1/sqrt(2)*(e2-e1), 1/sqrt(8)*(e1-e2-e3-e4-e5-e6-e7-e8)
    elif alg_choice == 2:
        a1, a2, a3, a4 = 1/sqrt(2)*(e1-e2), 1/sqrt(2)*(e2-e3),  1/sqrt(2)*(e3-e4), 1/sqrt(2)*(e4-e5) 
        a5, a6, a7, a8 = 1/sqrt(2)*(e5-e6), 1/sqrt(2)*(e6-e7),  1/sqrt(2)*(e7-e8), 1/sqrt(2)*(e7+e8)
    elif alg_choice == 3:
        a1, a2, a3, a4 = 1/sqrt(2)*(e7-e6), 1/sqrt(2)*(e6-e5),  1/sqrt(2)*(e5-e4), 1/sqrt(2)*(e4-e3) 
        a5, a6, a7, a8 = 1/sqrt(2)*(e3-e2), 1/sqrt(2)*(e2-e1),  1/sqrt(8)*(e1-e2-e3-e4-e5-e6-e7+e8), 1/sqrt(2)*(e1+e2) 
        
    simple_roots = [a1, a2, a3, a4, a5, a6, a7, a8]
    A = CM(simple_roots)
    Ainv = A**(-1)

    reciprocal_roots = []
    for i in range(len(simple_roots)):
        tmp = 0*e1
        for j in range(len(simple_roots)):
            tmp += 2*Ainv[i,j]*simple_roots[j]
        reciprocal_roots.append(tmp)

    MV_basis_list = [e1*e1]
    MV_rec_basis  = [e1*e1]
    for inv_n in range(1, 9):
        for comb in combinations(range(8), inv_n):
            tmp_a = e1*e1
            tmp_b = e1*e1
            for i in range(len(comb)):
                tmp_b = tmp_b ^ (reciprocal_roots[comb[i]])
                tmp_a =  (simple_roots[comb[i]]) ^ tmp_a
            MV_basis_list.append(tmp_a)
            MV_rec_basis.append(tmp_b)
    
    W = simple_roots[perm[0]]*simple_roots[perm[1]]*simple_roots[perm[2]]*simple_roots[perm[3]]*simple_roots[perm[4]]*simple_roots[perm[5]]*simple_roots[perm[6]]*simple_roots[perm[7]]

    Invariants_list = [W]
     
    simpleroot_invariants = []
    for inv in Invariants_list:
        sr_blade_coefs = [inv.blade_coefs()[0] ]
        for J in range(1,len(MV_basis_list)):
            sr_blade_coefs.append((inv|MV_rec_basis[J]).blade_coefs()[0])
        simpleroot_invariants.append(sr_blade_coefs)

    #Output the Coxeter element permutation with the respective simple root invariants
    return [perm, simpleroot_invariants]

########################################################
if __name__ == '__main__':  
    #Initialise the output file
    alg_names = ['A8','D8','E8']
    #with open('./'+alg_names[alg_choice-1]+'W_SimpleRootData.txt','w') as file:
    #    file.write(alg_names[alg_choice-1]+' Clifford Algebra: (Coxeter W in simple root basis)\n')
        
    #Run pooling (save as soon as a worker finishes)
    with Pool(12) as p:
        for idx, output in enumerate(p.imap_unordered(InvComputation,list(permutations(range(8)))[37143:])):
            with open('./'+alg_names[alg_choice-1]+'W_SimpleRootData.txt','a') as file:
                file.write(str(output)+'\n')
