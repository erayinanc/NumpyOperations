# efficient computation of dot product of two 3D fields (vectors)
# mathematical operation \phi_i\dot\phi_i
# libs: numpy as np
# usage: dotProd = DotP(Field1(n^3), Field2(n^3))
# return: 3D field of dot products of the given fields

# dot product of two vectors
def DotP(Field1,Field2):
    # parameters
    Ima,Jma,Kma,Sma = Field1.shape[:]
    
    # allocate
    Dot = np.zeros((Ima,Jma,Kma))
    
    # compute 
    Dot += Field1[:,:,:,0] * Field2[:,:,:,0]
    Dot += Field1[:,:,:,1] * Field2[:,:,:,1]
    Dot += Field1[:,:,:,2] * Field2[:,:,:,2]
    
    return Dot  
    
#EOF
