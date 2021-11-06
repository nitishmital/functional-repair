######## Functional Regenerating codes ###############
## Author : Nitish Mital

## Implementation of paper "Functional Broadcast Repair of Multiple Partial Failures in Wireless Distributed Storage Systems", submitted to JSAIT 7th issue. This code implements the functional repair of r full node failures.

######## Generates a random message, encodes and stores it in n nodes using the coding scheme proposed in the paper.
######## Contains a function repair() that takes as arguments lists of helper node indices and newcomer node indices, and implements the repair procedure proposed in the paper.

################################################################################################

import random
from collections import deque
import numpy as np
############################

'''
Function to repair a set of r nodes with the help of transmissions from d helper nodes
Inputs:
    helpers: list of indices of helper nodes
    newcmrs: list of indices of newcmr nodes
'''

def repair2(helpers, newcmrs,x_pkt):
	
	eval_pts = random.sample(base_field_elements, n-1)
	C_storage = codes.GeneralizedReedSolomonCode(eval_pts,d-(jbar-1)*r)
	C_gen_storage = C_storage.systematic_generator_matrix()


	## helper node forms r random linear combinations of r+x_pkt packets, and sends them to the r newcomers ##
	ind_tx_pkts = [t for t in range(d-(jbar-1)*r)]  # indices of the packets stored in each helper node => 1 to d-(jbar-1)*r. Therefore, the last n-1-(d-(jbar-1)r) packets do not play a role in the repair.
	rnd_nodes = random.sample(ind_tx_pkts,r+x_pkt) # r+xpkt indices of packets to be randomly sampled 
	read_pkt = [None]*len(helpers)
	for i in range(len(helpers)):
		read_pkt[i] = zero_matrix(k,r+x_pkt,2)

	## randomly sample r+xpkt packets from each helper node ##
	for i in range(len(helpers)):
		for j in range(r+x_pkt):
			read_pkt[i][j,:] = nodes[helpers[i]][rnd_nodes[j],:]
	
	
	rand_matrix = matrix([random.sample(base_field_elements,r+x_pkt) for i in range(r)])  ## r*(r+x_pkt)
	tx_pkt = [None]*len(helpers)
	for i in range(len(helpers)):
		tx_pkt[i] = zero_matrix(k,r,2)
	for i in range(len(helpers)):
		tx_pkt[i][:,:] = rand_matrix * read_pkt[i][:,:]

	for ncmr in range(len(newcmrs)):
		cyclic_list_tmp = [[[j,i] for j in range(g*r,d-(jbar-1-g)*r)] for g in range(jbar) for i in range(r)]
		cyclic_list_tmp = cyclic_list_tmp[:jbar*r]
		cyclic_list = cyclic_list_tmp
		rotations = [i for i in range(r)]
		for i in range(jbar*r):
			c_list = deque(cyclic_list_tmp[i])
			c_list.rotate(rotations[mod(i,r)])
			c_list = list(c_list)
			cyclic_list[i] = c_list
		cyclic_list = [[cyclic_list[i][j] for i in range(len(cyclic_list))] for j in range(len(cyclic_list[0]))]  ## creating the transpose of the list, so cyclic_list becomes (d-(j-1)r)*r
		

		labels = [0 for j in range(n-1)]
		for i in range(d-(jbar-1)*r):
			labels[i] = helpers[cyclic_list[i][ncmr][0]]
		
		node_lst = [i for i in range(n)]
		node_lst.remove(newcmrs[ncmr])
		
		for i in range(d-(jbar-1)*r):
			node_lst.remove(helpers[i])
			
		labels[d-(jbar-1)*r:n-1] = node_lst 

		Yx = matrix([[ tx_pkt[h[0]][h[1],0] for h in cyclic_list[i]] for i in range(len(cyclic_list))])  ## repair matrix for the evaluation points
		Yy = matrix([[ tx_pkt[h[0]][h[1],1] for h in cyclic_list[i]] for i in range(len(cyclic_list))])  ## repair matrix for the evaluation points		

		Yx_repair = Yx[:,0]
		Yy_repair = Yy[:,0]
		
		for i in range(d-(jbar-1)*r):
			rand_pts = matrix(random.sample(base_field_elements, jbar*r))		
			Yx_repair[i] = Yx[i,:]*rand_pts.transpose()  # Matrix of repair packets encoded with the MDS code
			Yy_repair[i] = Yy[i,:]*rand_pts.transpose()
		
		## generating n-1 packets for each node
		Yx_ordered = matrix([[k(0)] for i in range(n-1)])
		Yy_ordered = matrix([[k(0)] for i in range(n-1)])
		
		Yx_ordered[:d-(jbar-1)*r,0] = Yx_repair[:,0]
		Yy_ordered[:d-(jbar-1)*r,0] = Yy_repair[:,0]
		
		for i in range(n-1):
			Yx_ordered[i,0] = (C_gen_storage[:,i].transpose() * Yx_ordered[:d-(jbar-1)*r,0])[0,0]
			Yy_ordered[i,0] = (C_gen_storage[:,i].transpose() * Yy_ordered[:d-(jbar-1)*r,0])[0,0]
			
		
		ind = [t for t in range(n) if t not in [newcmrs[ncmr]]]
		for i in range(n-1):
			nodes[newcmrs[ncmr]][i,0] = Yx_ordered[i,0]
			nodes[newcmrs[ncmr]][i,1] = Yy_ordered[i,0]
		
################################################


######## Function to verify the linear independence of a list of points in extension field ##########################
######## Returns True is they are linearly independent, False if not #########################
## Inputs:
###    vpoints: list of points whose linear independence has to be checked
def verify_independence(vpoints):
	lnth=len(vpoints)
	v=[[base_k(0)]*l for i in range(lnth)]
	for i in range(lnth):
		v[i] = vector(vpoints[i])
	return((base_k^l).linear_dependence([v[j] for j in range(lnth)]) == [])
################################################

## Function to estimate the dimension of the intersection of two subspaces
## Inputs: 
##    v1: subspace 1
##    v2: subspace 2

def subspace_dimension(v):
	V = VectorSpace(GF(q),l)
	p_lst = [V(list(V(v[i]))) for i in range(len(v))]
	S = V.subspace(p_lst)
	return S.dimension()


if __name__ == '__main__':
	n=input('number of nodes: ')
	k1=input('recovery threshold: ')
	d=input('number of helpers: ')
	r=input('number of nodes repaired: ')
	jbar = input('jbar parameter for interior point: ')
	x_pkt = input('x_pkt parameter: ') ## input the parameter e from the paper
	q=input('size of base field: ')
	base_k=GF(q)
	l=(d-(jbar-1)*r)*(n-r)  # size of extension chosen so that there are enough linearly independent points in the finite extension field
	k.<x>=GF(q^l)
	subpacketization = int(k1/2*(2*(d-(jbar-1)*r) - k1 + r)  + r*((jbar-1)*k1 - jbar*(jbar-1)*r/2))
	Frob=k.frobenius_endomorphism();
	S.<y>=k['y',Frob]
	msg=[0]*subpacketization
	for i in range(subpacketization):
	   msg[i] = k.random_element()   # list of randomly picked points which act as the message symbols

	####### Construct linearized polynomial ################
	fy=0
	for i in range(subpacketization):
	    fy=fy+msg[i]*y^i

	####### choose l linearly independent evaluation points ##########
	points=[[0,0] for i in range(l)]
	v=[[base_k(0)]*l for i in range(l)]

	while(True):
	    for i in range(l):
		points[i][0] = k.random_element()   # list of randomly picked points which act as the evaluation points
		points[i][1] = fy(points[i][0])
		v[i] = vector(points[i][0])
	    print("initial evaluation points are linearly independent - ", (base_k^l).linear_dependence([v[j] for j in range(l)]) == [])
	    if ((base_k^l).linear_dependence([v[j] for j in range(l)]) == []):
		break
	##############################################


	base_field_elements = [i for i in base_k if i not in [0]]
	
	eval_pts = random.sample(base_field_elements, n-1)
	C_storage = codes.GeneralizedReedSolomonCode(eval_pts,d-(jbar-1)*r)
	C_gen_storage = C_storage.systematic_generator_matrix()

	eval_pts = random.sample(base_field_elements, (jbar+1)*r)
	C_repair = codes.GeneralizedReedSolomonCode(eval_pts,jbar*r)
	C_gen_repair = C_repair.systematic_generator_matrix()
	
	##############################################


	########## Store coded packets/symbols in the first d nodes ############
	nodes = [None]*n
	for i in range(n):
	    ####### initializing n-1 placeholders in each node. Each row is of type (x,y), where x is the evaluation point, and y is the coded packet (evaluation of fy on x)
	    nodes[i] = zero_matrix(k,n-1,2) 
	
	for i in range(n-r):
	   nodes[i][:d-(jbar-1)*r,:] = matrix(points[i*(d-(jbar-1)*r):(i+1)*(d-(jbar-1)*r)])   # d - (jbar-1)*r packets in node i
	   nodes[i][d-(jbar-1)*r:,:] = C_gen_storage[:,d-(jbar-1)*r:].transpose() * nodes[i][:d-(jbar-1)*r,:]  # storing the parity bits

	##############################################
	
	########## Fill in the remaining n-r nodes by using the repair scheme #############
	
	newcmrs = range(n-r,n)  # node indices of r newcomers
	helpers = range(d)  # nodes indices of d helper nodes
	repair2(helpers,newcmrs,x_pkt)
	
	# multiple repair rounds
	print('Performing 100 repair rounds of random failures and randomly chosen helper nodes..')
	for loop in range(100):
		newcmrs_helpers = random.sample(range(n),r+d)
		newcmrs2 = newcmrs_helpers[0:r]
		helpers2 = newcmrs_helpers[r:]
		#helpers = [0,2,4,5,6,7,8,9,10,12,13,15]
		repair2(helpers2,newcmrs2,x_pkt)
	
	
	## Check the dimension of 50 random sets of k1 nodes to estimate the probability of dimension being greater than subpacketization
	print('Randomly checking the dimension of various k-subspaces..')
	avg_dim = 0
	space_dim = [0 for i in range(50)]
	min_dim = l
	for loop in range(50):
		node_list = [i for i in range(n)]
		sample1 = random.sample(node_list,k1)
		vpoints1 = [nodes[j][i,ind] for j in sample1 for i in range(n-1) for ind in [0]]
		space_dim[loop] = subspace_dimension(vpoints1)
		if space_dim[loop] < min_dim:
			min_dim = space_dim[loop]
		avg_dim += space_dim[loop]/50
	
	print('average dimension of 50 k-subspaces: ',float(avg_dim))
	print('minimum dimension of k-subspace: ', float(min_dim))
	print('Optimal desired dimension (P*): ',subpacketization)
	
	
	


