# efficient computation of divergence of 4D field (i.e velocity tensor)
# mathematical operation div(\phi_i)
# libs: numpy as np
# usage: divergence = Div(Field(n^4), delta=<float>)
# note: delta is cell size in meters, sets delta to 1 m if not given
# return: 3D field with divergence field

# divergence of 4D field 
def Div(Field,**kwargs):
    # parameters
    Ima,Jma,Kma,Sma = Field.shape[:]
    delta = kwargs.get('delta', None)
    if not isinstance(delta, float):
        delta = 1.0
        print('delta is set to 1 m')

    # allocate
    Div = np.zeros((Ima,Jma,Kma))

    # compute 
    Div[1:Ima-1,1:Jma-1,1:Kma-1] += \
        (0.5 / delta) * ((Field[2:Ima,1:Jma-1,1:Kma-1,0] - Field[0:Ima-2,1:Jma-1,1:Kma-1,0]))
    Div[1:Ima-1,1:Jma-1,1:Kma-1] += \
        (0.5 / delta) * ((Field[1:Ima-1,2:Jma,1:Kma-1,1] - Field[1:Ima-1,0:Jma-2,1:Kma-1,1]))
    Div[1:Ima-1,1:Jma-1,1:Kma-1] += \
        (0.5 / delta) * ((Field[1:Ima-1,1:Jma-1,2:Kma,2] - Field[1:Ima-1,1:Jma-1,0:Kma-2,2]))

    return Div 

# EOF
