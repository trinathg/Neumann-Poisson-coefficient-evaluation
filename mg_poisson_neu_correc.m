%Code to solve the non-periodic Poisson problem using direct methods and
%computing the null space coefficients for various Multigrid levels

clear all; 

nx = 128; % No. of spacings in the X-direction 
ny = 128; % No. of spacings in the Y-direction

Nx = nx+1; %No.of points in X-direction
Ny = ny+1; %No.of points in Y-direction

lx = 1.0; %Length of the domain in X-direction
ly = 1.0; %Length of the domain in Y-direction

hx = lx/nx; 
hy = ly/ny; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Constant coefficients required for all the levels of multigrid

%Coefficients at the boundary 
alpha_b = 11.0;

%The coefficients at the near boundary points 
anb = 12./10.;
alphanb = 1./10.; 

% Coeficients in the interior 
a = 12./11.;
b = (1./4.)*(3./11.); 
alpha = 2./11.; 

%One sided coefficients normalized with B1
B1 = -11./6.;           
B2 = 3.;
B3 = (-3./2.); 
B4 = (1./3.); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Figuring out the no. of levels in the Multi-grid 

levels_x = log2(nx) -2; %The minimum the grid coarsening can go to is 8X8 grid 
levels_y = log2(ny) -2; 

levels = min(levels_x,levels_y); 

%Running loops through the levels to compute dx,dy and their squares

for i=1:levels
    dx(i) = (2^(i-1))*hx; 
    dy(i) = (2^(i-1))*hy; 
    
    hxx(i) = dx(i)*dx(i); 
    hyy(i) = dy(i)*dy(i); 
    
    %The intermediate coefficients 
    
    X1(i) = (1./hxx(i))*(1. - (B2/B1)*alphanb)*hxx(i); 
    X2(i) = -(anb/hyy(i))*(2. + (B2/B1))*hxx(i); 
    X3(i) = (1. - (B3/B1))*(alphanb/hxx(i))*hxx(i); 
    X4(i) = (1. - (B3/B1))*(anb/hyy(i))*hxx(i); 
    X5(i) = -(B4/B1)*(alphanb/hxx(i))*hxx(i); 
    X6(i) = -(B4/B1)*(anb/hyy(i))*hxx(i); 

    Y1(i) = (alpha/hxx(i))*hxx(i);
    Y2(i) = (1./hyy(i))*(a - (B2/B1)*b)*hxx(i); 
    Y3(i) = (1./hxx(i))*hxx(i); 
    Y4(i) = -(2.*(a+b) + (B3/B1)*b)*hxx(i)/hyy(i); 
    Y5(i) = alpha*hxx(i)/hxx(i); 
    Y6(i) = (a - (B4/B1)*b)*hxx(i)/hyy(i); 
    Y7(i) = b*hxx(i)/hyy(i); 

    XA(i) = alpha*hxx(i)/hxx(i);
    YA(i) = a*hxx(i)/hyy(i); 
    XH(i) = 1.*hxx(i)/hxx(i);
    YH(i) = -2.*(a+b)*hxx(i)/hyy(i);
    YB(i) = b*hxx(i)/hyy(i); 

    Z1(i) = alphanb*hxx(i)/hxx(i); 
    Z2(i) = anb*hxx(i)/hyy(i);     
end 


%%%%%%%%%%%  Matrix Definitions for LHS %%%%%%%%%%%%%%%%%

%Defining the reduced R matrix (ReR of size (Nx-2)X(Nx-2) 

i=1; 

%for i = 1:levels 
    
    nx_level = ceil(Nx/(2^(i-1))); 
    ny_level = ceil(Ny/(2^(i-1))); 
    
    ReR = sparse(nx_level-2); 

    for j=1:nx_level-2 
        
        %Boundary 
        
        if(j==1)  
            ReR(j,j) = -2.*anb;                  
            ReR(j,j+1) = anb;                    
        end
        
        %Near boundary 
        if(j==2)
            ReR(j,j-1) = a;               
            ReR(j,j) = -2.*(a+b);               
            ReR(j,j+1) = a;                         
            ReR(j,j+2) = b;                        
        end 
        
        %Interior
        if( (j>=3) && (j<=nx_level-4) )
            ReR(j,j-2) = b; 
            ReR(j,j-1) = a;
            ReR(j,j) = -2.*(a+b); 
            ReR(j,j+1) = a;
            ReR(j,j+2) = b;             
        end
        
        %Near boundary                    
        if(j==nx_level-3)
           ReR(j,j+1) = a;
           ReR(j,j) = -2.*(a+b); 
           ReR(j,j-1) = a; 
           ReR(j,j-2) = b;                      
        end
        
        %Boundary
         if(j==nx_level-2)
           ReR(j,j) = -2.*anb;                
           ReR(j,j-1) = anb;                                  
         end                           
    end 


%spy(ReR)

%Defining the reduced L matrix 

    ReL = sparse(nx_level-2);

    for j=1:nx_level-2        
        %Boundary 
        if(j==1)
            ReL(j,j) = 1.0;                             
            ReL(j,j+1) = alphanb;             
        end            
        
        %Interior
        if( j>=2 && j<=nx_level-3)
            ReL(j,j-1) = alpha; 
            ReL(j,j) = 1;
            ReL(j,j+1) = alpha;              
        end
             
        %Boundary
         if(j==nx_level-2)           
            ReL(j,j) = 1.0;                                   
            ReL(j,j-1) = alphanb;             
         end                  
    end

%spy(ReL)

%Defining Term 1 to be added to ReR 

    Term1 = sparse(nx_level-2); 

    for j=1:nx_level-2
        
        if(j==1)
            Term1(j,j) = (-B2/B1)*anb; 
            Term1(j,j+1) = (-B3/B1)*anb; 
            Term1(j,j+2) = (-B4/B1)*anb; 
        end 
    
        if(j==2)
            Term1(j,j-1) = (-B2/B1)*b; 
            Term1(j,j) = (-B3/B1)*b; 
            Term1(j,j+1) = (-B4/B1)*b;                 
        end    
    
        if(j==nx_level-3)
            Term1(j,j+1) = (-B2/B1)*b; 
            Term1(j,j) = (-B3/B1)*b; 
            Term1(j,j-1) = (-B4/B1)*b; 
        end     
    
        if(j==nx_level-2)
            Term1(j,j) = (-B2/B1)*anb; 
            Term1(j,j-1) = (-B3/B1)*anb; 
            Term1(j,j-2) = (-B4/B1)*anb;         
        end     
    end 

%spy(Term1) 

%Defining Term 2 to be added to ReL 

    Term2 = sparse(nx_level-2); 
    for j=1:nx_level-2
        if(j==1)
            Term2(j,j) = (-B2/B1)*alphanb; 
            Term2(j,j+1) = (-B3/B1)*alphanb; 
            Term2(j,j+2) = (-B4/B1)*alphanb; 
        end     
    
        if(j==nx_level-2)
            Term2(j,j) = (-B2/B1)*alphanb; 
            Term2(j,j-1) = (-B3/B1)*alphanb; 
            Term2(j,j-2) = (-B4/B1)*alphanb;         
        end     
    end 


    %spy(Term2)
 
% %%%%%%%%%%%  Computing individual blocks %%%%%%%%%%%%%%%%%
 
    %At the boundary 
    J11 = (X1(i)*(ReR + Term1) + X2(i)*(ReL+ Term2));     
    J12 = (X3(i)*(ReR + Term1) + X4(i)*(ReL + Term2)); 
    J13 = (X5(i)*(ReR + Term1) + X6(i)*(ReL + Term2)); 

    %Near boundary 
    J21 = (Y1(i)*(ReR+Term1) + Y2(i)*(ReL+Term2)); 
    J22 = (Y3(i)*(ReR+Term1) + Y4(i)*(ReL+Term2)); 
    J23 = (Y5(i)*(ReR+Term1) + Y6(i)*(ReL+Term2)); 
    J24 = (Y7(i)*(ReL+Term2)); 

    %Interior points 
    JI2_ = (YB(i)*(ReL+Term2)); 
    JI1_= (XA(i)*(ReR+Term1) + YA(i)*(ReL+Term2)); 
    JI = (XH(i)*(ReR+Term1) + YH(i)*(ReL+Term2)); 
    JI1 = (XA(i)*(ReR+Term1) + YA(i)*(ReL+Term2)); 
    JI2 = (YB(i)*(ReL+Term2)); 


    %size(J11) 

%%%%%%%%%%%  Assemble the Super matrix %%%%%%%%%%%%%%%%%
%Defining the super matrix in a block form

%Dimensions of the square supermatrix 

    dim_sup = (nx_level-2)*(ny_level-2); 
    dim_x = nx_level-2; 
    dim_y = ny_level-2;
    zero = sparse(nx_level-2,nx_level-2); 
    
    row_mat = sparse(dim_x, dim_sup); 
    row_A = sparse([]); %The RHS multiplier matrix   
    
    for rblk=1:dim_y    
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if(rblk==1)
            S = sparse(J11);              
            %size(S)
            for col=2:dim_y
                
                if(col==2) 
                    S = [S J12];                  
                end 
            
                if(col==3)
                    S = [S J13]; 
                end 
            
                if(col>3) 
                    S = [S zero]; 
                end 
                
            end             
            row_mat = S; 
        end                
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        if(rblk==2)            
            S = J21; 
            
            for col=2:dim_y
                
                if(col==2) 
                    S = [S J22];                  
                end 
            
                if(col==3)
                    S = [S J23]; 
                end 
                
                if(col==4)
                    S = [S J24];  
                end 
            
                if(col>4) 
                    S = [S zero]; 
                end 
                
            end             
            row_mat = [row_mat; S];                      
        end         
               
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if(rblk==3)
            
            S = JI2_; 
            
            for col=2:dim_y
                if(col==2)
                    S = [S JI1_]; 
                end 
                
                if(col==3)
                    S = [S JI];
                end 
                
                if(col==4)
                    S = [S JI1];
                end 
                
                if(col==5)
                    S = [S JI2]; 
                end
                
                if(col>5)
                    S = [S zero]; 
                end
            end
            row_mat = [row_mat; S]; 
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                
        if(rblk>3 && rblk<dim_y-2)
            
            S = zero; 
            
            for col=2:dim_y                
                
                if(col>=2 && col<rblk-2)
                    S = [S zero];
                end 
                
                if(col>=rblk-2 && col<=rblk+2)
                   
                   if(col==rblk-2)
                       S = [S JI2_];
                   end 
                   
                   if(col==rblk-1)
                       S = [S JI1_];
                   end 
                   
                   if(col==rblk)
                       S = [S JI];
                   end 
                   
                   if(col==rblk+1)
                       S = [S JI1];
                   end 
                   
                   if(col==rblk+2)
                       S = [S JI2]; 
                   end                             
                end
                
                if(col>rblk+2)
                    S = [S zero]; 
                end 
            end
            
            row_mat = [row_mat; S]; 
        end         
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              
        if(rblk==dim_y-2)
            
            S = JI2_; 
            
            for col=dim_y-1:-1:1
                
                if(col==dim_y-1)
                    S = [JI1_ S]; 
                end 
                
                if(col==dim_y-2)
                    S = [JI S];
                end 
                
                if(col==dim_y-3)
                    S = [JI1 S];
                end 
                
                if(col==dim_y-4)
                    S = [JI2 S]; 
                end
                
                if(col<dim_y-4)
                    S = [zero S]; 
                end
            end
            
            row_mat = [row_mat; S]; 
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
        if(rblk==dim_y-1)
            
            S = J21; 
            
            for col=dim_y-1:-1:1
                
                if(col==dim_y-1) 
                    S = [J22 S];                  
                end 
            
                if(col==dim_y-2)
                    S = [J23 S]; 
                end 
                
                if(col==dim_y-3)
                    S = [J24 S];  
                end 
            
                if(col<(dim_y-3)) 
                    S = [zero S]; 
                end 
                
            end             
            
            row_mat = [row_mat; S];           
        end         
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        S = zero; 
        
        if(rblk==dim_y)
            S = J11; 
            for col=dim_y-1:-1:1
                
                if(col==dim_y-1) 
                    S = [J12 S];                  
                end 
            
                if(col==dim_y-2)
                    S = [J13 S]; 
                end 
            
                if(col<dim_y-2) 
                    S = [zero S]; 
                end                 
            end             
            row_mat = [row_mat; S]; 
        end               
    
    end   
    
    [V,D,flag] = eigs(row_mat',2,0.0001);     
                   
    
    filename = ['coeff_from_lab_' num2str(i) '.dat'];
    
    nspace = fopen(filename,'w'); 
        
    if(D(1)<1e-6)
        fprintf(nspace,'%f \n',V(:,1)/ V(1,1)); 
        %save(filename,V(:,1)/ V(1,1)); 
    else
        fprintf(nspace,'%f \n',V(:,2)/ V(1,2)); 
        %save(filename,V(:,2)/ V(1,2)); 
    end 
    
%end 