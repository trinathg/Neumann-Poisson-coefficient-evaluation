import scipy.sparse.linalg as splinal
import math
from scipy.sparse import hstack
import scipy.sparse as sp
import numpy as np
import time

"""
Code to compute the null space coefficients for various Multigrid levels using LSQR method: a least squares based iterative method. 
"""

tic = time.time(); 

nx = 128; # No. of spacings in the X-direction 
ny = 128; # No. of spacings in the Y-direction

Nx = nx+1; # No.of points in X-direction
Ny = ny+1; # No.of points in Y-direction

lx = 32.0; # Length of the domain in X-direction
ly = 16.0; # Length of the domain in Y-direction

hx = lx/nx; 
hy = ly/ny; 

###########################################################
#Constant coefficients required for all the levels of multigrid

#Coefficients at the boundary 
alpha_b = 11.0;

#The coefficients at the near boundary points 
anb = 12./10.;
alphanb = 1./10.; 

#Coeficients in the interior 
a = 12./11.;
b = (1./4.)*(3./11.); 
alpha = 2./11.; 

#One sided coefficients normalized with B1
B1 = -11./6.;           
B2 = 3.;
B3 = (-3./2.); 
B4 = (1./3.); 

############################################################
#Figuring out the no. of levels in the Multi-grid 

levels_x = math.log(nx,2) -2; #The minimum the grid coarsening can go to is 8X8 grid 
levels_y = math.log(ny,2) -2; 

levels = int(min(levels_x,levels_y)); 

dx = [0.]*levels; 
dy = [0.]*levels; 

hxx = [0.]*levels; 
hyy = [0.]*levels; 

X1 = [0]*levels; 
X2 = [0]*levels; 
X3 = [0]*levels; 
X4 = [0]*levels; 
X5 = [0]*levels; 
X6 = [0]*levels; 

Y1 = [0]*levels; 
Y2 = [0]*levels; 
Y3 = [0]*levels; 
Y4 = [0]*levels; 
Y5 = [0]*levels; 
Y6 = [0]*levels; 
Y7 = [0]*levels; 

XA = [0]*levels; 
YA = [0]*levels; 
XH = [0]*levels; 
YH = [0]*levels; 
YB = [0]*levels; 

Z1 = [0]*levels; 
Z2 = [0]*levels; 

#Running loops through the levels to compute dx,dy and their squares

for i in range(levels):
    
    dx[i] = math.pow(2,i)*hx; 
    dy[i] = math.pow(2,i)*hy;   
     
    hxx[i] = dx[i]*dx[i]; 
    hyy[i] = dy[i]*dy[i]; 
    
    #The intermediate coefficients 
    
    X1[i] = (1./hxx[i])*(1. - (B2/B1)*alphanb)*hxx[i]; 
    X2[i] = -(anb/hyy[i])*(2. + (B2/B1))*hxx[i]; 
    X3[i] = (1. - (B3/B1))*(alphanb/hxx[i])*hxx[i]; 
    X4[i] = (1. - (B3/B1))*(anb/hyy[i])*hxx[i]; 
    X5[i] = -(B4/B1)*(alphanb/hxx[i])*hxx[i]; 
    X6[i] = -(B4/B1)*(anb/hyy[i])*hxx[i]; 

    Y1[i] = (alpha/hxx[i])*hxx[i];
    Y2[i] = (1./hyy[i])*(a - (B2/B1)*b)*hxx[i]; 
    Y3[i] = (1./hxx[i])*hxx[i]; 
    Y4[i] = -(2.*(a+b) + (B3/B1)*b)*hxx[i]/hyy[i]; 
    Y5[i] = alpha*hxx[i]/hxx[i]; 
    Y6[i] = (a - (B4/B1)*b)*hxx[i]/hyy[i]; 
    Y7[i] = b*hxx[i]/hyy[i]; 

    XA[i] = alpha*hxx[i]/hxx[i];
    YA[i] = a*hxx[i]/hyy[i]; 
    XH[i] = 1.*hxx[i]/hxx[i];
    YH[i] = -2.*(a+b)*hxx[i]/hyy[i];
    YB[i] = b*hxx[i]/hyy[i]; 

    Z1[i] = alphanb*hxx[i]/hxx[i]; 
    Z2[i] = anb*hxx[i]/hyy[i];     
   

################ Matrix Definitions for LHS ####################

#Defining the reduced R matrix (ReR of size (Nx-2)X(Nx-2) 

for i in range(levels): 
	print "---------"   
	print "Level=",i	
	pnx = math.pow(2,i)
	pny = math.pow(2,i) 
	    
	nx_level = int(math.ceil(Nx/pnx))
	ny_level = int(math.ceil(Ny/pny)) 

	tot_nump = (nx_level-2)*(ny_level-2)
	    
	ReR = sp.lil_matrix((nx_level-2,nx_level-2))
	
	print "nx_level=",nx_level
	print "ny_level=",ny_level
	print "Size of ReR = ", ReR.get_shape()
	
	for j in range(nx_level-2):    	        
		#Boundary		
		
		if j==0:  
			ReR[j,j] = -2.*anb                  
			ReR[j,j+1] = anb                    		      
		
		#Near boundary 
		if j==1:
			ReR[j,j-1] = a               
			ReR[j,j] = -2.*(a+b)
			ReR[j,j+1] = a                         
			ReR[j,j+2] = b                        
		 
		
		#Interior
		if( (j>=2) and (j<=nx_level-5) ): 
			ReR[j,j-2] = b 
			ReR[j,j-1] = a
			ReR[j,j] = -2.*(a+b)
			ReR[j,j+1] = a
			ReR[j,j+2] = b    		
		
		#Near boundary                    
		if j==nx_level-4:
			ReR[j,j+1] = a
			ReR[j,j] = -2.*(a+b) 
			ReR[j,j-1] = a 
			ReR[j,j-2] = b                      
		        
		#Boundary
		if j==nx_level-3:
			ReR[j,j] = -2.*anb                
			ReR[j,j-1] = anb                                  
	                                	

	#Defining the reduced L matrix 

	ReL = sp.lil_matrix( (nx_level-2,nx_level-2))

	for j in range(nx_level-2):
	
		#Boundary 
		if j==0:
			ReL[j,j] = 1.0                             
			ReL[j,j+1] = alphanb
		        
		#Interior
		if( j>=1 and j<=nx_level-4):
			ReL[j,j-1] = alpha
			ReL[j,j] = 1
			ReL[j,j+1] = alpha
		             
		#Boundary
		if(j==nx_level-3):           
			ReL[j,j] = 1.0                                   
			ReL[j,j-1] = alphanb
		                   

	
	#Defining Term 1 to be added to ReR 

	Term1 = sp.lil_matrix((nx_level-2,nx_level-2))

	for j in range(nx_level-2):
		if j==0:
			Term1[j,j] = (-B2/B1)*anb 
			Term1[j,j+1] = (-B3/B1)*anb
			Term1[j,j+2] = (-B4/B1)*anb 
		   
		if j==1:
			Term1[j,j-1] = (-B2/B1)*b 
			Term1[j,j] = (-B3/B1)*b 
			Term1[j,j+1] = (-B4/B1)*b                 
		    
		if j==nx_level-4:
			Term1[j,j+1] = (-B2/B1)*b 
			Term1[j,j] = (-B3/B1)*b 
			Term1[j,j-1] = (-B4/B1)*b            
	    
		if j==nx_level-3:
			Term1[j,j] = (-B2/B1)*anb 
			Term1[j,j-1] = (-B3/B1)*anb 
			Term1[j,j-2] = (-B4/B1)*anb              
	     
	
	#Defining Term 2 to be added to ReL 

	Term2 = sp.lil_matrix((nx_level-2,nx_level-2))

	for j in range(nx_level-2):
		if j==0:
			Term2[j,j] = (-B2/B1)*alphanb
			Term2[j,j+1] = (-B3/B1)*alphanb 
			Term2[j,j+2] = (-B4/B1)*alphanb 
		   
		if j==nx_level-3:
			Term2[j,j] = (-B2/B1)*alphanb
			Term2[j,j-1] = (-B3/B1)*alphanb 
			Term2[j,j-2] = (-B4/B1)*alphanb         
		     
	    
	
	############# Computing individual blocks ###############
	 
	#At the boundary 
	J11 = (X1[i]*(ReR + Term1) + X2[i]*(ReL+ Term2))     
	J12 = (X3[i]*(ReR + Term1) + X4[i]*(ReL + Term2)) 
	J13 = (X5[i]*(ReR + Term1) + X6[i]*(ReL + Term2)) 

	#Near boundary 
	J21 = (Y1[i]*(ReR+Term1) + Y2[i]*(ReL+Term2)) 
	J22 = (Y3[i]*(ReR+Term1) + Y4[i]*(ReL+Term2)) 
	J23 = (Y5[i]*(ReR+Term1) + Y6[i]*(ReL+Term2)) 
	J24 = (Y7[i]*(ReL+Term2)) 

	#Interior points 
	JI2_ = (YB[i]*(ReL+Term2)) 
	JI1_= (XA[i]*(ReR+Term1) + YA[i]*(ReL+Term2)) 
	JI = (XH[i]*(ReR+Term1) + YH[i]*(ReL+Term2)) 
	JI1 = (XA[i]*(ReR+Term1) + YA[i]*(ReL+Term2)) 
	JI2 = (YB[i]*(ReL+Term2)) 

	
	############ Assemble the Super matrix ##################
	#Defining the super matrix in a block form

	#Dimensions of the square supermatrix 

	dim_sup = (nx_level-2)*(ny_level-2) 

	dim_x = nx_level-2 
	dim_y = ny_level-2

	zero = sp.lil_matrix((nx_level-2,nx_level-2)) 
	    
	row_mat = sp.lil_matrix((dim_x, dim_sup))
			    
	for rblk in range(dim_y):
		if(rblk==0):
			S = J11              
			#size(S)
			for col in range(dim_y):
			        
		        	if col==1:                                         
		            		S = sp.hstack([S,J12])
		    
		        	if col==2:
		            		S = sp.hstack([S,J13])
		    
		        	if col>2:                    
		            		S = sp.hstack([S,zero])                   
		                
		    	row_mat = S;                 
			
	       		print row_mat.get_shape()
		################################        
		
		if(rblk==1):            
			S = J21
		    
			for col in range(dim_y):
		        
		        	if(col==1):                     		                                  	            
		            		S = sp.hstack([S,J22])
		        	if(col==2):
		            		S = sp.hstack([S,J23])                                
		        	if(col==3):
		            		S = sp.hstack([S,J24])                   
		    
		        	if(col>3): 
		            		S = sp.hstack([S,zero])                                  
		                 
		    	row_mat = sp.vstack([row_mat,S])
		        
		        print row_mat.get_shape()
		################################
		
		if(rblk==2):            
			S = JI2_
			             
			for col in range(dim_y):
				if(col==1):
		            		S = sp.hstack([S,JI1_])                               
		        	if(col==2):
		            		S = sp.hstack([S,JI])                              
		        	if(col==3):
		            		S = sp.hstack([S,JI1])                                
		        	if(col==4):
		            		S = sp.hstack([S,JI2])                                
		        	if(col>4):
		            		S = sp.hstack([S,zero]) 

		    	row_mat = sp.vstack([row_mat,S])         
		    	
		    	print row_mat.get_shape()
		################################
		                        
		if(rblk>2 and rblk<dim_y-3):            
			S = zero 
		    
			for col in range(dim_y):                
		                    
				if(col>=1 and col<rblk-2):
					S = sp.hstack([S,zero])                
				
				if(col>=rblk-2 and col<=rblk+2):                       
					if(col==rblk-2):
				       		S = sp.hstack([S,JI2_])                                       
				   	if(col==rblk-1):
				       		S = sp.hstack([S,JI1_])                                      
				   	if(col==rblk):
				       		S = sp.hstack([S,JI])                                     
				   	if(col==rblk+1):
				       		S = sp.hstack([S,JI1])                                       
				   	if(col==rblk+2):
				       		S = sp.hstack([S,JI2])                                  
				if(col>rblk+2):
			       		S = sp.hstack([S,zero])                           
		        row_mat = sp.vstack([row_mat,S]) 
		         
	  		print row_mat.get_shape()      
		
		################################
		      
		if(rblk==dim_y-3):
		    
			S = JI2_ 
		    
			for col in range(dim_y-2,-1,-1):
		        	
			    	if(col==dim_y-2):
					S = sp.hstack([JI1_,S])                  
				
				if(col==dim_y-3):
					S = sp.hstack([JI,S])                
				
				if(col==dim_y-4):
					S = sp.hstack([JI1,S])
				               
				if(col==dim_y-5):
					S = sp.hstack([JI2,S]) 
				                
				if(col<dim_y-5):
					S = sp.hstack([zero,S])                                
		                               
		    	row_mat = sp.vstack([row_mat,S]) 
		    	
		    	print row_mat.get_shape()
		                        
		################################
		        
		if(rblk==dim_y-2):            
			S = J21 
		    
			for col in range(dim_y-2,-1,-1):
		        
				if(col==dim_y-2):
					S = sp.hstack([J22,S])                  
			    
				if(col==dim_y-3):
					S = sp.hstack([J23,S]) 		         
				
				if(col==dim_y-4):
					S = sp.hstack([J24,S])  		         	
							    
				if(col<(dim_y-4)): 
					S = sp.hstack([zero,S]) 		       		                                                       				
			row_mat = sp.vstack([row_mat,S])                            
		
			print row_mat.get_shape()
		
		################################
		
		if(rblk==dim_y-1):
			S = J11 
			for col in range(dim_y-2,-1,-1):
		    		if(col==dim_y-2): 
		            		S = sp.hstack([J12,S])                                              
				if(col==dim_y-3):
					S = sp.hstack([J13,S]) 		        		    
				if(col<dim_y-3): 
					S = sp.hstack([zero,S]) 		                                                  
		    	row_mat = sp.vstack([row_mat,S]) 
		    	
		    	print row_mat.get_shape()

	
	A = row_mat.tocsc()
	rhsb = np.ones(tot_nump)

	
	sol = splinal.lsqr(A, rhsb, damp=0.0, atol=1e-12, btol=1e-12, conlim=1e18, iter_lim=100000, show=True, calc_var=False)

	x= sol[0] 

	z = A.transpose()*(rhsb-A*x)

	print 'Norm of the dot product with nullspace vector is ', np.linalg.norm(z)

	V = rhsb - A*x 

	filename = 'coeff_from_lab_' + str(i+1) + '.dat'

	np.savetxt(filename,V/V[0])
	np.savetxt('z.txt',z)
	
	print "Size of ReR at the end = ", ReR.get_shape()
	
toc = time.time()

print 'Time elapsed is ', toc - tic, 'secs'
