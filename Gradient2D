# Efficient gradient of 2D field by Eray Inanc (eray.inanc@uni-due.de)
# Note: Compute 2D gradient once, replaces numpy.gradient with two axes
# Libs: import numpy as np
# Usage: gradients = Grad2D(Field(n^2), delta=<float>
# return is a 3D field with gradients[:,:,0] is the axis=0 and gradients[:,:,1] is the axis=1
# delta is cell size in meters

def Grad2D(Field,**kwargs):
    # parameters
    Ima,Kma = Field.shape[:]
    delta = kwargs.get('delta', None)
    if not isinstance(delta, float):
        delta = 1.0
        print('delta is set to 1 m')

    # allocate
    Grad = np.zeros((Ima+2,Kma+2,2))

    # compute 
    Grad[1:Ima-1,1:Kma-1,0] = (0.5 / delta) * \
        ((Field[2:Ima,1:Kma-1] - Field[0:Ima-2,1:Kma-1]))
    Grad[1:Ima-1,1:Kma-1,1] = (0.5 / delta) * \
        ((Field[1:Ima-1,2:Kma] - Field[1:Ima-1,0:Kma-2]))
    
    return Grad 

#EOF
