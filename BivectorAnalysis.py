'''Script to analyse bivector subinvariants, and the graphs constructed from them'''
'''To run:
    ~ Ensure the filepaths are correct for data import.
    ~ Select the algebra to consider using the 'data_label' variable in the second cell. 
    ~ Run cells sequentially (demarked by '#%%') to perform the respective analysis. 
    ~ Below the ### lines is further code for analysis of overlaps beween datasets and random matrices as quoted in the paper.
'''
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy

#Define a function to return the adjacency matrix from the bivector invariant
def matrix(inv,symmetric=True):
    mat = np.zeros((8,8),dtype=int) #...initialise the matrix
    #Select whether to construct the symmetric graph from existence of a bivector, or the antisymmetric graph from the explicit bivector multiplicities (&sign) 
    if symmetric:
        inv2 = [1 if num else 0 for num in inv] #...convert all nonzero entries to 1
        mat[0,1:] = inv2[:7]
        mat[1:,0] = inv2[:7]
        mat[1,2:] = inv2[7:13]
        mat[2:,1] = inv2[7:13]
        mat[2,3:] = inv2[13:18]
        mat[3:,2] = inv2[13:18]
        mat[3,4:] = inv2[18:22]
        mat[4:,3] = inv2[18:22]
        mat[4,5:] = inv2[22:25]
        mat[5:,4] = inv2[22:25]
        mat[5,6:] = inv2[25:27]
        mat[6:,5] = inv2[25:27]
        mat[6,7]  = inv2[27]
        mat[7,6]  = inv2[27]
    else:
        mat[0,1:] =  inv[:7]
        mat[1:,0] = -inv[:7]
        mat[1,2:] =  inv[7:13]
        mat[2:,1] = -inv[7:13]
        mat[2,3:] =  inv[13:18]
        mat[3:,2] = -inv[13:18]
        mat[3,4:] =  inv[18:22]
        mat[4:,3] = -inv[18:22]
        mat[4,5:] =  inv[22:25]
        mat[5:,4] = -inv[22:25]
        mat[5,6:] =  inv[25:27]
        mat[6:,5] = -inv[25:27]
        mat[6,7]  =  inv[27]
        mat[7,6]  = -inv[27]
    return mat

'''
#Define the ADE8 graphs
A8 = nx.Graph([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)])
D8 = nx.Graph([(0,2),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)])
E8 = nx.Graph([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(2,7)])
#Compute their max eigenvalues
ADE_eigvals = [max(np.linalg.eigvals(nx.adjacency_matrix(A8).todense())),max(np.linalg.eigvals(nx.adjacency_matrix(D8).todense())),max(np.linalg.eigvals(nx.adjacency_matrix(E8).todense()))]
'''

#%% #Import data
data_label = 'A' #...choose: ['A','D','E']
with open('./InvariantData/'+data_label+'8inv_SimpleRootData.txt','r') as file:
    data = []
    for line in file.readlines()[1:]:
        data.append(eval(line))
del(line,file)

#Extract bivector data
invariants = [[] for i in range(4)] #...sublist for each of the 4 unique invariants orders (1,2,3,4) ignoring the trivial zero-invaraints for 0/8 ((5,6,7,8) match (3,2,1,0) repsectively)
for perm in data:
    for inv_order, inv in enumerate(perm[1][1:5]):
        invariants[inv_order].append(inv[9:37])
invariants = np.array(invariants)   
del(perm,inv,inv_order)  
print('Bivectors imported...')

#Extract unique bivectors
invariants_unique, invariants_uniquecounts = np.unique(np.concatenate(invariants,axis=0),axis=0,return_counts=True)
invariants_ordersplit_unique = np.array([np.unique(order,axis=0) for order in invariants])

#Extract unique adjacency matrices
adjacencies_unique, adjacencies_uniquecounts = np.unique(np.array([matrix(inv) for inv in invariants_unique]),axis=0,return_counts=True)
adjacencies_ordersplit_unique = [np.unique(np.array([matrix(inv) for inv in order]),axis=0) for order in invariants_ordersplit_unique]

#Extract non-isomorphic graphs
graphs_unique, graphs_uniquecounts = [], []
for adjacency in adjacencies_unique:
    new_check = True
    for graph_idx in range(len(graphs_unique)):
        if nx.is_isomorphic(nx.from_numpy_matrix(adjacency),nx.from_numpy_matrix(graphs_unique[graph_idx])):
            graphs_uniquecounts[graph_idx] += 1
            new_check = False
            break
    if new_check:
        graphs_unique.append(adjacency)
        graphs_uniquecounts.append(1)
graphs_unique = np.array(graphs_unique)
graphs_uniquecounts = np.array(graphs_uniquecounts)

graphs_ordersplit_unique = []
for order in adjacencies_ordersplit_unique:
    graphs_ordersplit_unique.append([])
    for adjacency in order:
        new_check = True
        for graph in graphs_ordersplit_unique[-1]:
            if nx.is_isomorphic(nx.from_numpy_matrix(adjacency),nx.from_numpy_matrix(graph)): 
                new_check = False
                break
        if new_check:
            graphs_ordersplit_unique[-1].append(adjacency)
    graphs_ordersplit_unique[-1] = np.array(graphs_ordersplit_unique[-1])
    
del(adjacency,graph_idx,new_check,order,graph)

#Print numbers of unique objects in each reduction stage
print(f'Number of unique (invariants, adjacencies, graphs): ({len(invariants_unique)},{len(adjacencies_unique)},{len(graphs_unique)})\n...note: zero invaraint omitted from counts!')

#%% #Compute eigenvalues
symmetric = True #...can change this to False to compute the eigenspectrums for the directed weighted graph representation of the bivectors (note eigenvalues are complex)
eigvals_invariants = [[] for i in range(4)]
eigvals_graphs     = [[] for i in range(4)]
for inv_idx in range(4):
    for inv in invariants[inv_idx]: 
        eigvals_invariants[inv_idx].append(np.linalg.eigvals(matrix(inv,symmetric)))
    for graph in graphs_ordersplit_unique[inv_idx]:
        eigvals_graphs[inv_idx].append(np.linalg.eigvals(graph))
del(inv_idx,inv,graph)

eigvals_invariants = [np.array(i) for i in eigvals_invariants] 
eigvals_graphs     = [np.array(i) for i in eigvals_graphs] 
if symmetric: 
    #Change type to be real (since symmetric matrices) and round off the precision error
    eigvals_invariants = [np.round(np.real(i),8) for i in eigvals_invariants]
    eigvals_graphs     = [np.round(np.real(i),8) for i in eigvals_graphs]
    #Extract the maximum eigenvalue in each case
    maxev_invariants = [np.array(list(map(max,i))) for i in eigvals_invariants]
    maxev_graphs     = [np.array(list(map(max,i))) for i in eigvals_graphs]
    #Compute unique eigenvalues
    ev_uniq_info_invariants = [np.unique(i,return_counts=True) for i in maxev_invariants]
    ev_uniq_info_graphs     = [np.unique(i,return_counts=True) for i in maxev_graphs] #...this is computed to verify that all the frequencies are 1 when reducing to unique graphs (as can be seen from np.unique(np.concatenate([ev_info[1] for ev_info in ev_uniq_info_graphs])))
    print(f'# unique eigenvalues: {[len(i[0]) for i in ev_uniq_info_invariants]}')

#%% #Histogram of max-eigenvalues
plt.figure('Max Eigenvalue Histogram')
for inv_idx in range(4):
    plt.scatter(ev_uniq_info_invariants[inv_idx][0],ev_uniq_info_invariants[inv_idx][1],alpha=0.5,label='Order '+str(inv_idx+1))
plt.xlabel('Max Eigenvalue')
plt.ylabel('Multiplicity')
plt.ylim(0)
#plt.yticks(range(0,14000,2000))
plt.xlim(1,7.2)
leg=plt.legend(loc='upper left')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.grid()
plt.tight_layout()
#plt.savefig(data_label+'8_MaxEigenvalueScatter_Invariants.pdf')

#%% #Compute the dominant eigenvector
largest_eigvec = [[] for i in range(4)]
for inv_idx in range(4): 
    for inv in invariants[inv_idx]: 
        evec = np.linalg.eigh(matrix(inv,True))[1][:,-1]
        if evec[0] < 0: #...PF says all eigenvector entries the same sign, but may need to flip the sign to positive
            evec = - evec
        largest_eigvec[inv_idx].append(evec)
largest_eigvec = np.array(largest_eigvec)
del(inv_idx,inv,evec)

#Compute the centrlity variance and most central node for each dominant eigenvector
largest_eigvec_variances = np.var(largest_eigvec,axis=2)
largest_eigvec_centralistnode = np.argmax(largest_eigvec,axis=2)
largest_evvar_uniq_info = [np.unique(i,return_counts=True) for i in largest_eigvec_variances]
largest_evidx_uniq_info = [list(np.unique(i,return_counts=True)) for i in largest_eigvec_centralistnode]
#Add frequencies of 0 to the counts for indices that don't occur as the most central for any invariants 
for inv_idx in range(4):
    for node in range(8):
        if node not in largest_evidx_uniq_info[inv_idx][0]:
            largest_evidx_uniq_info[inv_idx][0] = np.append(largest_evidx_uniq_info[inv_idx][0],[node])
            largest_evidx_uniq_info[inv_idx][1] = np.append(largest_evidx_uniq_info[inv_idx][1],[0])

#%% #Histogram of eigenvector-centrality variances
plt.figure('Eigenvector-Centrality Analysis')
for inv_idx in range(4):
    plt.scatter(largest_evvar_uniq_info[inv_idx][0],largest_evvar_uniq_info[inv_idx][1],alpha=0.5,label='Order '+str(inv_idx+1))
    #plt.scatter(np.array(largest_evidx_uniq_info[inv_idx][0])+1,largest_evidx_uniq_info[inv_idx][1],label='Order '+str(inv_idx+1))
plt.xlabel('Eigenvector Centrality Variance')
#plt.xlabel('Most Central Node Index')
plt.ylabel('Multiplicity')
#plt.ylim(-100)
leg=plt.legend(loc='upper right')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.grid()
plt.tight_layout()
plt.savefig(data_label+'8_MaxEigenvectorVariances.pdf')
#plt.savefig(data_label+'8_MostCentralNode.pdf')

####################################################################################
#%% #Compute overlaps between datasets
print('Length changes from unique reduction between orders:')
print('Invariants:',sum(map(len,invariants_ordersplit_unique))-len(invariants_unique))
print('Adjacencies:',sum(map(len,adjacencies_ordersplit_unique))-len(adjacencies_unique))
print('Graphs:',sum(map(len,graphs_ordersplit_unique))-len(graphs_unique))

#To check the overlap between algebras (if there is no overlap the lengths of the unique reductions of each algebra individually will sum to the length of the unique reduction of all algebras together):
#...run the data import cell for each data_label
#...rename each dataset with 'A8 = deepcopy(invariants)', etc
#...run 'print(f'Overlap between algebras: {sum(map(len,[A8,D8,E8])) - len(np.unique(np.concatenate((A8,D8,E8),axis=0),axis=0))}')'

####################################################################################
#%% #Random graph max eigenvalue distribution
number = 282240 #...number of non-empty matrices in each algebra (so frequency scales the same!)
random_graphs, random_maxevs = [], []
disconnect_count = 0
while len(random_graphs) < number:
    random_inv = np.random.choice([0,1],28) 
    mat = matrix(random_inv)
    if not nx.is_connected(nx.from_numpy_matrix(mat)): 
        disconnect_count+=1
        continue
    random_graphs.append(mat)
    random_maxevs.append(max(np.round(np.real(np.linalg.eigvals(mat)),8)))
    #if idx%10000 == 0: print(idx) #...progress report
#Compute unique graphs
random_graphs = np.array(random_graphs)
randomgraph_uniqueinfo = np.unique(random_graphs,axis=0,return_counts=True)
#Compute unique eigenvalues
random_maxevs = np.array(random_maxevs)
ev_uniq_info = np.unique(random_maxevs,return_counts=True)
print(f'Number of unique (graphs, eigenvalues): ({len(randomgraph_uniqueinfo[0])},{len(ev_uniq_info[0])})')
del(random_inv,mat)

#%% #Histogram of random max-eigenvalues
plt.figure('Max Eigenvalue Histogram')
plt.scatter(ev_uniq_info[0],ev_uniq_info[1],alpha=0.2)
plt.xlabel('Max Eigenvalue')
plt.ylabel('Multiplicity')
plt.ylim(0)
plt.xlim(1,7.2)
plt.grid()
plt.tight_layout()
#plt.savefig('Random8_MaxEigenvalueScatter.png')

#%% #Extract graphs with eigenvalues < 2 --> only ADE!
l2, l2idx = [], [] #...the eigenvalues < 2, and the indices in the list which correspond to that invariant
for idx, ev in enumerate(random_maxevs):
    if ev < 2:
        l2.append(ev)
        l2idx.append(idx)
del(idx,ev)
#Identify the unique eigenvalues that occur
l2uniq, l2uniq_counts = np.unique(l2,return_counts=True)
print(f'Unique Eigenvalues < 2:\t{l2uniq}\n...with frequencies:\t\t{l2uniq_counts}')
