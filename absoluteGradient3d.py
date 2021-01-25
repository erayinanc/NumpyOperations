# efficient computation of absolute gradient of 3D field
# mathematical operation |grad(\phi_i)|
# libs: numpy as np
# usage: absoluteGradient = Grad3D(Field(n^3), delta=<float>)
# note: delta is cell size in meters, sets delta to 1 m if not given
# return: 3D field with absolute gradient

# gradient of 3D field
def Grad3D(Field,delta):
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
def AGrad3D(Field,**kwargs):
    # parameters
    delta = kwargs.get('delta', None)
    if not isinstance(delta, float):
        delta = 1.0
        print('delta is set to 1 m')
        
    # compute 
    GradF = Grad3D(Field,delta)
    ABSGrad = np.sqrt(GradF[:,:,:,0]**2.0 + GradF[:,:,:,1]**2.0 + GradF[:,:,:,2]**2.0)
    return ABSGrad 

#EOF
