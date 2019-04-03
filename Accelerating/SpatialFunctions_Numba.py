from numba import njit, prange

Forward  = 1
Backward = 2
Central  = 3
DifferenceScheme = Backward

def SpatialFunctions_Setup(GW=100, GH=100, dx=1.0, dy=1.0): 
    global dX,dY,Grid_Width,Grid_Height
    dX = dx
    dY = dy
    Grid_Width = GW
    Grid_Height = GH
    
@njit(parallel=True)
def d_dx(z,c,r): 
    global DifferenceScheme
    if (DifferenceScheme == Forward):
        dz_dx = ( ( z[c+1,r] - z[c,r] )/dX )
    elif (DifferenceScheme == Backward):
        dz_dx = ( ( z[c,r] - z[c-1,r] )/dX )
    elif (DifferenceScheme == Central):
        dz_dx = ( ( z[c+1,r] - z[c-1,r] )/2.0/dX )        
    return dz_dx

@njit(parallel=True)
def d_dy(z,c,r): 
    global DifferenceScheme
    if (DifferenceScheme == Forward):
        dz_dy = ( ( z[c,r+1] - z[c,r] )/dY )
    elif (DifferenceScheme == Backward):
        dz_dy = ( ( z[c,r] - z[c,r-1] )/dY )
    elif (DifferenceScheme == Central):
        dz_dy = ( ( z[c,r+1] - z[c,r-1] )/2.0/dY )        
    return dz_dy

@njit(parallel=True)
def d2_dxy2(z,c,r):   # (Array z, Column c, Row r)
    return  (z[c-1,r] + z[c+1,r] - 2.0*z[c,r])/dX/dY + \
            (z[c,r-1] + z[c,r+1] - 2.0*z[c,r])/dY/dY 

# Periodic Boundary conditions function

@njit(parallel=True)
def PeriodicBoundaries(z,c,r):
    if  (r==0):             # Lower boundary
        z[c,r] = z[c,Grid_Height-2]
    elif(r==Grid_Height-1): # Upper boundary
        z[c,r] = z[c,1]
    elif(c==0):             # Left boundary
        z[c,r] = z[Grid_Width-2,r]
    elif(c==Grid_Width-1):  # Right boundary
        z[c,r] = z[1,r]

# Neumann Boundary conditions function, having zero flux on the edge

@njit(parallel=True)
def NeumanBoundaries(z,c,r):
    if  (r==0):             # Lower boundary
        z[c,r]=z[c,1]
    elif(r==Grid_Height-1): # Upper boundary
        z[c,r]=z[c,Grid_Height-2];
    elif(c==0):             # Left boundary
        z[c,r]=z[1,r]
    elif(c==Grid_Width-1):  # Right boundary
        z[c,r]=z[Grid_Width-2,r]

# Reflecting Boundary conditions function, for shallow water equations

@njit(parallel=True)
def ReflectingBoundaries(u,v,c,r): 
    if  (r==0):             # Lower boundary
        u[c,r] = u[c,1]
        v[c,r] =-v[c,1]
    elif(r==Grid_Height-1): # Upper boundary
        u[c,r] = u[c,Grid_Height-2]
        v[c,r] =-v[c,Grid_Height-2]
    elif(c==0):             # Left boundary
        u[c,r] =-u[1,r]
        v[c,r] = v[1,r]
    elif(c==Grid_Width-1):  # Right boundaries
        u[c,r] =-u[Grid_Width-2,r]
        v[c,r] = v[Grid_Width-2,r]

# Persistent Flux Boundary condition function, extrapolating over de boundaries

@njit(parallel=True)
def PersistentFluxBoundaries(z,c,r):
    if  (r==0):             # Lower boundary
        z[c,r] = 2*z[c,1] - z[c,2]
    elif(r==Grid_Height-1): # Upper boundary
        z[c,r] = 2*z[c,Grid_Height-2] - z[c,Grid_Height-3]
    elif(c==0):             # Left boundary
        z[c,r] = 2*z[1,r] - z[2,r]
    elif(c==Grid_Width-1):  # Right boundary
        z[c,r] = 2*z[Grid_Width-2,r] - z[Grid_Width-3,r]

# Dirichlet Boundary condition function, having fixed values on the edge

@njit(parallel=True)
def DirichletBoundaries(z,c,r):
    if  (r==0):             # Lower boundary
        z[c,r]=Value
    elif(r==Grid_Height-1): # Upper boundary
        z[c,r]=Value
    elif(c==0):             # Left boundary
        z[c,r]=Value
    elif(c==Grid_Width-1):  # Right boundary
        z[c,r]=Value