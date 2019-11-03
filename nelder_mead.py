# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 18:32:25 2019
******************************************************************************
language		: python
requirement		: --
author			: Lu Wang, Chemical Engineering, University of Southern California
function		: Carry out Nelder-Mead Simplex algorithm in a separate file, for easier implement
******************************************************************************
"""

import numpy as np
import copy


# ==================== define dimension and initial conditions ================
dimension = 3 # number of dimensions, n

# define boundary conditions
dimension1_lim = [100,1000]
dimension2_lim = [100,1000]
dimension3_lim = [100,1000]
x_lim = [dimension1_lim, dimension2_lim, dimension3_lim]


# define initial simplex
x0 = np.array([400,400,100])
x1 = np.array([600,400,200])
x2 = np.array([400,200,200])
x3 = np.array([400,400,300])


# initialize lists and variables
x_list = [x0, x1, x2, x3] # it needs to be a list, for summation
f_list = []
method = None
[fh, i_h, xh, fs, i_s, xs, fl, i_l, xl, c] = [None, None, None, None, None, None, None, None, None, None]
[fr, xr, fe, xe, fc, xc] = [None, None, None, None, None, None]





# ensure x meets boundary conditions
def meet_lim(x,x_lim): # x is an array of n-dimension, x_lim is a n-length list of limits
	for i in range(len(x)):
		if x[i] < x_lim[i][0]:
			x[i] = x_lim[i][0]
		elif x[i] > x_lim[i][1]:
			x[i] = x_lim[i][1]


	# apply following extra restrictions when running expriments
	if x[-1] > 1.5*x[0]: # ratio p(Br)/p(Cs-Pb) should not be larger than 1.5 (for reasonable stoichiometry)
		x[-1] = x[0]*1.5
	if x[1] > 1.5*x[0]: # ratio p(gas)/p(Cs-Pb) should not be larger than 1.5 (prevent backflowing)
		x[1] = x[0]*1.5
	elif x[1] < 0.2*x[0]: # ratio p(gas)/p(Cs-Pb) should not be smaller than 0.2 (prevent uneven droplets across channels)
		x[1] = x[0]*0.2

	return x


# used in the main script when re-starting a new round of optimization
def new_x_list(p1_set, p3_set, p4_set, f_new):
	global x_list, f_list
	x_list = []
	f_list = []
	x_list.append(np.array([p1_set, p3_set, p4_set]))
	for i in range(len(x_list[0])):
		x_temp = copy.deepcopy(x_list[0])
		if x_temp[i] < 1000*0.5:
			x_temp[i] += 300
		else: 
			x_temp[i] -= 300
		x_temp = meet_lim(x_temp, x_lim)
		x_temp = x_temp.astype(int)
		x_list.append(x_temp)


# this is used in main script for increasing throughput
def expand_pressure(p1_set, p3_set, p4_set, threshold):
	if p1_set > p4_set:
		expand_ratio = threshold/p4_set
		p1_set = int(p1_set*expand_ratio)
		p3_set = int(p3_set*expand_ratio)
		p4_set = int(p4_set*expand_ratio)
	else:
		expand_ratio = threshold/p1_set
		p1_set = int(p1_set*expand_ratio)
		p3_set = int(p3_set*expand_ratio)
		p4_set = int(p4_set*expand_ratio)

	x_temp = meet_lim(np.array([p1_set,p3_set,p4_set]),x_lim)
	x_new = x_temp.astype(int)
	return x_new


def ordering(f_list, x_list):
	# ordering: h = maximum (high), l = minimum (low), s = second maximum
	global fh, i_h, xh, fs, i_s, xs, fl, i_l, xl, c
	fh = max(f_list)
	print('fh ', fh)
	i_h = f_list.index(fh)
	print('i_h', i_h)
	xh = x_list[i_h]
	print('xh', xh)
	fl = min(f_list)
	print('fl', fl)
	i_l = f_list.index(fl)
	print('i_l', i_l)
	xl = x_list[i_l]
	print('xl', xl)
	f_list_sort = sorted(f_list)
	fs = f_list_sort[-2]
	print('fs ', fs)
	i_s = f_list.index(fs)
	print('i_s', i_s)
	xs = x_list[i_s]
	print('xs', xs)
	# centroid (an array)
	c = (sum(x_list)-x_list[i_h])/dimension



def simplex(method,f_new):
	# standard method, alpha = 1, beta = 0.5, gamma = 2, sigma = 0.5
	global fh, i_h, xh, fs, fl, i_l, xl, c, fr, xr, fe, xe, fc, xc
	global f_list, x_list
	
	if method == None:					# when not having enough vertices, just append and continue
		f_list.append(f_new)
		if len(f_list) <= dimension: 
			x_new = x_list[len(f_list)]
		elif len(f_list) > dimension:	# get n+1 vertices now, should use reflect method now
			ordering(f_list,x_list)
			# get new x value by reflect
			x_new = c + 1*(c-xh)
			# exam the limit
			x_new = meet_lim(x_new, x_lim)
			x_new = x_new.astype(int)
			print('start reflect ',x_new)
			xr = x_new
			x_list[i_h] = xr 	# update x_list
			method = 'reflect'

	elif method == 'shrink':
		f_list.append(f_new)
		# get new x value
		if len(f_list) <= dimension: 	# when not having n+1 dimensions, should run more times
			x_new = xl + 0.5*(x_list[len(f_list)] - xl)
			x_new = x_new.astype(int)
			x_list[len(f_list)] = x_new
		elif len(f_list) > dimension:	# get n+1 vertices now, should use reflect method now
			ordering(f_list, x_list)
			# get new x value by reflect
			x_new = c + 1*(c-xh)
			# exam the limit
			x_new = meet_lim(x_new, x_lim)
			x_new = x_new.astype(int)
			print('start reflect ',x_new)
			xr = x_new
			x_list[i_h] = xr # update x_list
			method = 'reflect'

	elif method == 'reflect':
		fr = f_new
		if fr>=fl and fr<fs:			 # accept xr
			f_list[i_h] = fr
			x_list[i_h] = xr
			ordering(f_list, x_list)
			# get new x value by reflect
			x_new = c + 1*(c-xh)
			# exam the limit
			x_new = meet_lim(x_new, x_lim)
			x_new = x_new.astype(int)
			print('reflect again ',x_new)
			xr = x_new
			x_list[i_h] = xr # update x_list
			method = 'reflect'
		elif fr<fl: 					# EXPAND method
			x_new = c + 2*(xr-c)
			# exam the limits
			x_new = meet_lim(x_new, x_lim)
			x_new = x_new.astype(int)
			print('expand ',x_new)
			xe = x_new
			x_list[i_h] = xe # update x_list
			method = 'expand'
		elif fr>=fs: 					# CONTRACT
			if fr<fh: 					# outside contract
				x_new = c + 0.5*(xr-c)
				x_new = x_new.astype(int)
				print('outside contract ',x_new)
				xc = x_new
				x_list[i_h] = xc # update x_list
				method = 'out_contract'
			elif fr>=fh: 				# inside contract
				x_new = c + 0.5*(xh-c)
				x_new = x_new.astype(int)
				print('inside contract ',x_new)
				xc = x_new
				x_list[i_h] = xc # update x_list
				method = 'in_contract'

	elif method == 'expand':
		fe = f_new
		if fe<fr: 						# accept xe
			f_list[i_h] = fe
			x_list[i_h] = xe
			ordering(f_list, x_list)
		elif fe>=fr: 					# accept xr
			f_list[i_h] = fr
			x_list[i_h] = xr
			ordering(f_list, x_list)
		# get new x value by reflect
		x_new = c + 1*(c-xh)
		# exam the limit
		x_new = meet_lim(x_new, x_lim)
		x_new = x_new.astype(int)
		print('reflect after expand ',x_new)
		xr = x_new
		x_list[i_h] = xr # update x_list
		method = 'reflect'

	elif method == 'out_contract':
		fc = f_new
		#xc = x_used
		if fc <= fr: 					# accept xc
			f_list[i_h] = fc
			x_list[i_h] = xc
			ordering(f_list, x_list)
			# get new x value by reflect
			x_new = c + 1*(c-xh)
			# exam the limit
			x_new = meet_lim(x_new, x_lim)
			x_new = x_new.astype(int)
			print('reflect after outside contract ',x_new)
			xr = x_new
			x_list[i_h] = xr # update x_list
			method = 'reflect'
		else: 							# SHRINK
			x_list[i_l], x_list[0] = x_list[0], x_list[i_l] # switch xl to the first
			f_list = [fl]
			# get new x value
			x_new = xl + 0.5*(x_list[1]-xl)
			x_new = x_new.astype(int)
			x_list[1] = x_new # update x_list
			print('shrink after outside contract ',x_new)
			method = 'shrink'

	elif method == 'in_contract':
		fc = f_new
		#print(x_list)
		if fc < fh: 					# accept xc
			f_list[i_h] = fc
			x_list[i_h] = xc
			ordering(f_list, x_list)
			# get new x value by reflect
			x_new = c + 1*(c-xh)
			# exam the limit
			x_new = meet_lim(x_new, x_lim)
			x_new = x_new.astype(int)
			print('reflect after inside contract ',x_new)
			xr = x_new
			x_list[i_h] = xr # update x_list
			method = 'reflect'
		else: 							# SHRINK
			x_list[i_l], x_list[0] = x_list[0], x_list[i_l] # switch xl to the first
			f_list = [fl]
			# get new x value by shrink
			x_new = xl + 0.5*(x_list[1]-xl)
			x_new = x_new.astype(int)
			x_list[1] = x_new # update x_list
			print('shrink after inside contract ',x_new)
			method = 'shrink'
			
	# print('return method and new x')
	return method, x_new




# an example of using this algorithm
if __name__ == "__main__":
	'''
	# define a problem
	def fx(x,y,z):
		...
		return ...

	# start optimizing	
	count = 1
	target = ...
	while fx(x,y,z) > target:
		print('iteration ', str(count))
		print(x)
		print(y)
		print(z)
		print(fx(x,y,z))
		method, [x,y,z] = nm.simplex(method,fx(x,y,z))
		count += 1
	'''