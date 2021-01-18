# efficient computation of dot product of two 4D fields (tensors)
# mathematical operation \phi_i\dot\phi_i
# libs: numpy as np
# usage: dotProd = DotP(Field1(n^4), Field2(n^4))
# return: 3D field of dot products of the given fields

# dot product of two vectors
def DotP(Field1,Field2):
    # parameters
    Ima,Jma,Kma,Sma = Field1.shape[:]
    
    # allocate
    Dot = np.zeros((Ima,Jma,Kma))
    
    # compute
    for i in range(Sma):
        Dot += Field1[:,:,:,i] * Field2[:,:,:,i]

    return Dot  
    
#EOF
