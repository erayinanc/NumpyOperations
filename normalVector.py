# efficient computation of normal vector
# mathematical operation \vec(n_i) = -grad(\phi_i) / |grad(\phi_i)| 
# libs: numpy as np
# usage: nVector = normalVec(Field(n^3))
# return: 4D field of normal vector with each axis on the last element 
# i.e. nVector[:,:,:,1] is the normal vector for axis=1

# gradient of 3D field
def Grad3D(Field):
    # parameters
    Ima,Jma,Kma = Field.shape[:]
   
    # allocate
    Grad = np.zeros((Ima,Jma,Kma,3))

    # compute 
    Grad[1:Ima-1,1:Jma-1,1:Kma-1,0] = (0.5 / delta) * \
        ((Field[2:Ima,1:Jma-1,1:Kma-1] - Field[0:Ima-2,1:Jma-1,1:Kma-1]))
    Grad[1:Ima-1,1:Jma-1,1:Kma-1,1] = (0.5 / delta) * \
        ((Field[1:Ima-1,2:Jma,1:Kma-1] - Field[1:Ima-1,0:Jma-2,1:Kma-1]))
    Grad[1:Ima-1,1:Jma-1,1:Kma-1,2] = (0.5 / delta) * \
        ((Field[1:Ima-1,1:Jma-1,2:Kma] - Field[1:Ima-1,1:Jma-1,0:Kma-2])) 
    
    return Grad

# abs gradient of a 3D field
def AGrad3D(Field):      
    # compute 
    GradF = Grad3D(Field,delta)
    ABSGrad = np.sqrt(GradF[:,:,:,0]**2.0 + GradF[:,:,:,1]**2.0 + GradF[:,:,:,2]**2.0)
    return ABSGrad

# Compute normal vector N = -grad(Yp)/absgrad(Yp)
def normalVec(Field):
    # parameters
    Ima,Jma,Kma = Field.shape[:]
    
    # allocate
    N = np.zeros((Ima,Jma,Kma,3))
    
    # compute 
    GradF = Grad3D(Field,IB)
    ABSGradF = AGrad3D(Field,IB)
    for i in range(3):
        N[:,:,:,i] = -GradF[:,:,:,i] / np.maximum(1.0e-9, ABSGradF)
    
    return N

#EOF
