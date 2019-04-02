//
//  SpatialFunctions_iPy.cl
//
//  Created by Johan Van de Koppel on 8/13/17.
//  Copyright © 2017 Johan Van de Koppel. All rights reserved.
//

#ifndef DifferenceScheme
    #define Forward  1
    #define Backward 2
    #define Central  3
    #define DifferenceScheme Backward
#endif

#ifndef SPATIALFUNCTIONS_CL
#define SPATIALFUNCTIONS_CL

////////////////////////////////////////////////////////////////////////////////
// Apriori prototyping
////////////////////////////////////////////////////////////////////////////////

float d2_dxy2( __global float* ); // A prototype definition for d2_dxy2
float d_dx(__global float* );
float d_dy( __global float* );
void PeriodicBoundaries( __global float* );
void NeumannBoundaries( __global float* );
void DirichletBoundaries( __global float*, float );

////////////////////////////////////////////////////////////////////////////////
// Gradient operator definitions, to calculate advective fluxes
////////////////////////////////////////////////////////////////////////////////

float d_dx(__global float* z)
{
    // Getting thread coordinates on the grid
    const size_t current = get_global_id(0);
    const size_t row     = (size_t)current/Grid_Width;
    const size_t column  = current%Grid_Width;

    #if DifferenceScheme == Forward
        const size_t right = row * Grid_Width + column+1;
        const size_t left = row * Grid_Width + column;
        const float dx = dX;
    #elif DifferenceScheme == Backward
        const size_t right = row * Grid_Width + column;
        const size_t left = row * Grid_Width + column-1;
        const float dx = dX;
    #elif DifferenceScheme == Central
        const size_t right = row * Grid_Width + column+1;
        const size_t left = row * Grid_Width + column-1;
        const float dx = dX*2;
    #endif

    return ( ( z[right] - z[left] )/dx );
}

float d_dy(__global float* z)
{
    // Getting thread coordinates on the grid
    const size_t current = get_global_id(0);
    const size_t row	 = (size_t)current/Grid_Width;
    const size_t column	 = current%Grid_Width;

    #if DifferenceScheme == Forward
        const size_t top = (row+1) * Grid_Width + column;
        const size_t bottom = (row) * Grid_Width + column;
        const float dy = dY;
    #elif DifferenceScheme == Backward
        const size_t top = (row) * Grid_Width + column;
        const size_t bottom = (row-1) * Grid_Width + column;
        const float dy = dY;
    #elif DifferenceScheme == Central
        const size_t top = (row+1) * Grid_Width + column;
        const size_t bottom = (row-1) * Grid_Width + column;
        const float dy = dY*2;
    #endif

    return ( ( z[top] - z[bottom] )/dy );
}


////////////////////////////////////////////////////////////////////////////////
// Laplacation operator definition, to calculate diffusive fluxes
////////////////////////////////////////////////////////////////////////////////

float d2_dxy2(__global float* z)
{
    const float dx = dX;  // Forcing dX to become a float
    const float dy = dY;  // Forcing dY to become a float

    // Getting thread coordinates on the grid
    const size_t current = get_global_id(0);
    const size_t row	 = (size_t)current/Grid_Width;
    const size_t column	 = current%Grid_Width;

    const size_t left = row * Grid_Width + column-1;
    const size_t right = row * Grid_Width + column+1;
    const size_t top = (row-1) * Grid_Width + column;
    const size_t bottom = (row+1) * Grid_Width + column;

    return  (z[left] + z[right ] - 2.0*z[current])/dx/dx +
            (z[top ] + z[bottom] - 2.0*z[current])/dy/dy ;
}

////////////////////////////////////////////////////////////////////////////////
// Periodic Boundary condition function, copying values from the other edge
////////////////////////////////////////////////////////////////////////////////

void PeriodicBoundaries(__global float* z)
{
    // Getting thread coordinates on the grid
    const size_t current = get_global_id(0);
    const size_t row	 = (size_t)current/Grid_Width;
    const size_t column	 = current%Grid_Width;

    // current = row * Grid_Width + column;

    if(row==0) // Lower boundary
    {
        z[row * Grid_Width + column] = z[(Grid_Height-2) * Grid_Width+column];
    }
    else if(row==Grid_Height-1) // Upper boundary
    {
        z[row * Grid_Width + column] = z[1*Grid_Width+column];
    }
    else if(column==0) // Left boundary
    {
        z[row * Grid_Width + column] = z[row * Grid_Width + Grid_Width-2];
    }
    else if(column==Grid_Width-1) // Right boundary
    {
        z[row * Grid_Width + column] = z[row * Grid_Width + 1];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Neumann Boundary condition function, having zero flux on the edge
////////////////////////////////////////////////////////////////////////////////

void NeumannBoundaries(__global float* z)
{
    // Getting thread coordinates on the grid
    const size_t current = get_global_id(0);
    const size_t row	 = (size_t)current/Grid_Width;
    const size_t column	 = current%Grid_Width;

    // current = row * Grid_Width + column;

    if(row==0) // Lower boundary
    {
        z[current]=z[1*Grid_Width + column];
    }
    else if(row==Grid_Height-1) // Upper boundary
    {
        z[current]=z[(Grid_Height-2) * Grid_Width + column];
    }
    else if(column==0) // Left boundary
    {
        z[current]=z[row * Grid_Width + 1];
    }
    else if(column==Grid_Width-1) // Right boundary
    {
        z[current]=z[row * Grid_Width + Grid_Width-2];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Dirichlet Boundary condition function, having fixed values on the edge
////////////////////////////////////////////////////////////////////////////////

void DirichletBoundaries(__global float* z, float Value)
{
    // Getting thread coordinates on the grid
    const size_t current = get_global_id(0);
    const size_t row	 = (size_t)current/Grid_Width;
    const size_t column	 = current%Grid_Width;

    // current = row * Grid_Width + column;

    if(row==0) // Lower boundary
    {
        z[current]=Value;
    }
    else if(row==Grid_Height-1) // Upper boundary
    {
        z[current]=Value;
    }
    else if(column==0) // Left boundary
    {
        z[current]=Value;
    }
    else if(column==Grid_Width-1) // Right boundary
    {
        z[current]=Value;
    }
}

#endif
