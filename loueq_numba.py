from copy import deepcopy
from random import shuffle

from NEMtropy import UndirectedGraph as UG
from tqdm import trange

import numpy as np

from numba import njit
from numba.core import types
from numba.typed import Dict

int_array = types.int8[:]

class loueq_loop():
    # probably, a class is an overshooting.
    # nevertheless, it permits to store all the relevant information
    def __init__(self, links, run=100):
        self.links=links
        
        self.nodes, self.fitnesses=MaxEntCM(self.links)
        
        #adjacency list
        self.adl=edgelist_to_adjacencylist(links)

        
        self.Q=0
        self.communities=None
        self.membership=None
        self.loueq_loop(run)
        
        
    def loueq_loop(self, run):
        for i in trange(run):
            if i==0:
                nro=False
            else:
                nro=True
                
            _run=loueq(self.links, self.nodes, self.fitness_dict, self.adl, randomize=nro)
            if _run.Q>self.Q:
                self.Q=_run.Q
                self.communities=_run.communities
                self.membership=_run.membership
                
                
    def modularity_exact(self, membership=None):
        
        # if membership is None, calculate the modularity and put it in self.Q
        if membership==None:
            self.Q=0
            for key in self.communities.keys():
                guys=self.communities[key]
                for i_g in range(len(guys)-1):
                    for j_g in range(i_g+1, len(guys)):
                        x_i_g=self.fitness_dict[guys[i_g]]
                        x_j_g=self.fitness_dict[guys[j_g]]
                        self.Q-=p_func(x_i_g, x_j_g)
                        if j_g in self.adl[i_g]:
                            self.Q+=1
            self.Q/=len(self.links)
            return self.Q
        # otherwise calculate it, but do not store it in the internal variable                     
        else:
            # create the communities variable
            communities={}
            # create the Q variable
            Q=0
            for key in membership.keys():
                comm=membership[key]
                if comm in communities.keys():
                    communities[comm].append(key)
                else:
                    communities[comm]=[key]
                    
            for key in communities.keys():
                guys=communities[key]
                for i_g in range(len(guys)-1):
                    for j_g in range(i_g+1, len(guys)):
                        x_i_g=self.fitness_dict[guys[i_g]]
                        x_j_g=self.fitness_dict[guys[j_g]]
                        Q-=p_func(x_i_g, x_j_g)
                        if j_g in self.adl[i_g]:
                            Q+=1
            Q/=len(self.links)
        return Q
         
        

class loueq():
    
    def __init__(self, links, nodes=None, fitness_dict=None, adl=None, randomize=False):
        
        self.links=links
        
        if nodes==None:
            self.nodes, self.fitnesses=MaxEntCM(self.links)
            
        else:
            self.nodes=nodes
            self.fitness_dict=fitness_dict

        if adl==None:
            #adjacency list
            self.adl=edgelist_to_adjacencylist(links)
        else:
            self.adl=adl
        
        #if randomize:
        #    shuffle(self.nodes)
            
        # node membership
        self._membership=[i for i in range(len(self.nodes))]
        # communities
        self._communities=[[node] for node in self.nodes]
        
        
        self.loueq()
        self.community_name_update()
        self.modularity_exact()
        
    
    def loueq(self):
        nodes=[]
        
        while nodes!=self._communities:
            # since [] is different from the community structure initialization,
            # it starts by entering the while loop
            # it stops when the following loops does not change the 
            # community structure
            nodes=deepcopy(self._communities)
            
            # PAY ATTENTION! nodes here are the aggregated nodes,
            # i.e. nodes are the community at the previous steps 
            # and so different from self.nodes
            # at the very beggining they are lists made by a single node,
            # then they turn to their aggregated versions

            # This is why deepcopy is needed: for a copy of a list,
            # .copy() is enough. For a copy of a list of something
            # (in our case, lists), deepcopy is needed.
            
            comms_0=deepcopy(self._communities)
            for node in nodes:
                self.step_0(node)
                
                
            # if the communities found at the previous round
            # are different than the one at the very beginning,
            # try again and  
            # update the community composition
            # until you end up with something stable
            while list(self._communities.values())!=comms_0:
                comms_0=list(self._communities.values())
                for node in nodes:
                    self.step_0(node)        
                

    @njit
    def step_0(self, nodes):
        nn=[]
        nn_memb=[]
        for node in nodes:
            for a_node in self.adl[node]:
                # check for the nearest neighbors and their membership
                if a_node not in nn:
                    nn.append(a_node)
                _m_nn=self._membership[a_node]
                if _m_nn not in nn_memb:
                    nn_memb.append(_m_nn)
        

        my_memb=self._membership[node]
        # it is the membership of the last node in the previous loop:
        # since by construction they are all in the same
        # community, it is ok
        my_comm=self._communities[my_memb].copy()
        
        old_comm=my_comm.copy()
        for node in nodes:
            old_comm.remove(node)
        
        deltaQ=0
        new_memb=my_memb
        # the fitnesses and the adjacency list of all the nodes in nodes
                
        for _new_memb in nn_memb:
            if _new_memb!=my_memb:
                # all the elements in the new community
                new_comm=self._communities[_new_memb]
                _deltaQ=0
                # in the case in which the (super)node constitues a community on its own, 
                # deltaQ is zero.
                # In this sense, any change should at least produce an increase in Q, 
                # i.e. a positive _deltaQ.
                # Consider that I am taking the membership of the connected nodes, 
                # so I am considering even that community I am in.
                # In this sense, I am checking if it a good choice.
                for i_node in range(len(nodes)):
                    # the fitnesses and the adjacency list of each node in nodes
                    x_i=self.fitnesses[i_node]
                    a_i=self.adl[i_node]
                    _deltaQ+=self.delta_Q(old_comm, new_comm, a_i, x_i)
                if _deltaQ>deltaQ:
                    deltaQ=_deltaQ
                    new_memb=_new_memb
        if new_memb!=my_memb:
            # I choose a different community
            for node in nodes:
                self._membership[node]=new_memb
                self._communities[new_memb].append(node)
                # the community I belong to is great enough to survive to my desertion
                self._communities[my_memb].remove(node)
                
            if len(self._communities[my_memb])==0:
                # I was the last guy in my community
                del self._communities[my_memb]
        
    @njit
    def delta_Q_0(self, c, a_i, x_i):
        # c is the new community 
        # a_i is the entry of the adjacency list
        # x_i is the fitness of i
        if len(c)==0:
            return 0
        else:
            out=0
            for j in c:
                if j in a_i:
                    out+=1
                x_j=self.fitness_dict[j]
                out-=p_func(x_i, x_j)
            return out
    

    def delta_Q(self, c_0, c_1, a_i, x_i):
        # c_0 is the old community
        # c_1 is the new community
        return self.delta_Q_0(c_1, a_i, x_i)-self.delta_Q_0(c_0, a_i, x_i)
    
    @njit
    def community_name_update(self):
        self.communities={}
        converter={}
        for i_key, key in enumerate(self._communities.keys()):
            self.communities[i_key]=self._communities[key]
            self.communities[i_key].sort()
            converter[key]=i_key

        self.membership={}
        for key in self._membership.keys():
            self.membership[key]=converter[self._membership[key]]
            
    @njit
    def modularity_exact(self, membership=None):
        
        # if membership is None, calculate the modularity and put it in self.Q
        if membership==None:
            self.Q=0
            for key in self.communities.keys():
                guys=self.communities[key]
                for i_g in range(len(guys)-1):
                    for j_g in range(i_g+1, len(guys)):
                        x_i_g=self.fitness_dict[guys[i_g]]
                        x_j_g=self.fitness_dict[guys[j_g]]
                        self.Q-=p_func(x_i_g, x_j_g)
                        if j_g in self.adl[i_g]:
                            self.Q+=1
            self.Q/=len(self.links)
            return self.Q
        # otherwise calculate it, but do not store it in the internal variable                     
        else:
            # create the communities variable
            communities={}
            # create the Q variable
            Q=0
            for key in membership.keys():
                comm=membership[key]
                if comm in communities.keys():
                    communities[comm].append(key)
                else:
                    communities[comm]=[key]
                    
            for key in communities.keys():
                guys=communities[key]
                for i_g in range(len(guys)-1):
                    for j_g in range(i_g+1, len(guys)):
                        x_i_g=self.fitness_dict[guys[i_g]]
                        x_j_g=self.fitness_dict[guys[j_g]]
                        Q-=p_func(x_i_g, x_j_g)
                        if j_g in self.adl[i_g]:
                            Q+=1
            Q/=len(self.links)
        return Q
                                    
def MaxEntCM(links):
    myGraph=UG()
    myGraph.set_edgelist(links)
    myGraph.solve_tool('cm', 'fixed-point', initial_guess='chung_lu', adjacency='cm_exp', method_adjacency='fixed-point', initial_guess_adjacency='chung_lu', max_steps=500, full_return=False, verbose=False, linsearch=True, tol=1e-08, eps=1e-08)
        
    # PAY ATTENTION! nodes are indeed the nodes of the graph
    nodes=np.array(myGraph.nodes_dict.values())
    fitnesses=myGraph.x
    
    return nodes, fitnesses


def p_func(x_i, x_j):
    return x_i*x_j/(1+x_i*x_j)

# NUMBA WARNING: this part has not been numbized since I cannot see a simple way to do it,
# without losing the benefit of having a dictionary
def edgelist_to_adjacencylist(el):
    adl={}
    for edge in el:
        if edge[0] in adl.keys():
            adl[edge[0]].append(edge[1])
        else:
            adl[edge[0]]=[edge[1]]
        if edge[1] in adl.keys():
            adl[edge[1]].append(edge[0])
        else:
            adl[edge[1]]=[edge[0]]
    return adl
        