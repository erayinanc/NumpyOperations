# efficient computation of absolute gradient of 2D field
# mathematical operation |grad(\phi_i)|
# libs: numpy as np
# usage: absoluteGradient = Grad2D(Field(n^2), delta=<float>)
# note: delta is cell size in meters, sets delta to 1 m if not given
# return: 2D field with absolute gradient

# gradient of 2D field
def Grad2D(Field,delta):
    # parameters
    Ima,Kma = Field.shape[:]

    # allocate
    Grad = np.zeros((Ima+2,Kma+2,2))

    # compute 
    Grad[1:Ima-1,1:Kma-1,0] = (0.5 / delta) * \
        ((Field[2:Ima,1:Kma-1] - Field[0:Ima-2,1:Kma-1]))
    Grad[1:Ima-1,1:Kma-1,1] = (0.5 / delta) * \
        ((Field[1:Ima-1,2:Kma] - Field[1:Ima-1,0:Kma-2]))
    
    return Grad 

# abs gradient of a 2D field
def AGrad2D(Field,**kwargs):
    # parameters
    delta = kwargs.get('delta', None)
    if not isinstance(delta, float):
        delta = 1.0
        print('delta is set to 1 m')
        
    # compute 
    GradF = Grad2D(Field,delta)
    ABSGrad = np.sqrt(GradF[:,:,0]**2.0 + GradF[:,:,1]**2.0)
    return ABSGrad 



#EOF
