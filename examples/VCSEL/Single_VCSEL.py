import emopt
from emopt.misc import NOT_PARALLEL

import numpy as np
import matplotlib.pyplot as plt
from math import pi

####################################################################################
# Set up the size of the problem
####################################################################################
unit_cell_size = 3.0 # microns
nominal_width = 1.243 # microns
dx = 0.02 # microns
dy = 0.02 # microns
wavelength = 0.85 # microns
simulation_height = 4*unit_cell_size # microns
simulation_width = 4*unit_cell_size # microns
N = int(np.ceil((simulation_height)/dx)+1)
M = int(np.ceil((simulation_width)/dy)+1)

GaAs_Index = 3.4
AlGaAsO_Index = 3.3

GaAs_cavity_centered = emopt.grid.Polygon(
    xs=[-unit_cell_size/2, -unit_cell_size/2,
        -nominal_width/2,nominal_width/2,
        unit_cell_size/2, unit_cell_size/2,
        nominal_width/2, -nominal_width/2],
    ys=[-nominal_width/2, nominal_width/2,
        unit_cell_size/2, unit_cell_size/2,
        nominal_width/2, -nominal_width/2,
        -unit_cell_size/2, -unit_cell_size/2]
)

GaAs_cavity_1 = GaAs_cavity_centered.copy().translate(unit_cell_size/2,0)
GaAs_cavity_1.layer = 1
GaAs_cavity_1.material_value = GaAs_Index**2  # GaAs permittivity

GaAs_cavity_2 = GaAs_cavity_centered.copy().translate(-unit_cell_size/2,0)
GaAs_cavity_2.layer = 1
GaAs_cavity_2.material_value = GaAs_Index**2 #GaAs permittivity

GaAs_substrate = emopt.grid.Rectangle(
    x0=0, 
    y0=0,
    xspan=simulation_width,
    yspan=simulation_height,
)

GaAs_substrate.layer = 2
GaAs_substrate.material_value = AlGaAsO_Index**2

## Combine the primitives into a epsilon structure
eps = emopt.grid.StructuredMaterial2D(simulation_height,
                                      simulation_width,
                                      dx, 
                                      dy)
eps.add_primitive(GaAs_cavity_1)
eps.add_primitive(GaAs_cavity_2)
eps.add_primitive(GaAs_substrate)

# Define the permeability structure,
# which is constant in this case
mu = emopt.grid.ConstantMaterial2D(1.0)

domain = emopt.misc.DomainCoordinates(
    -simulation_width/2,simulation_width/2,
    -simulation_height/2,simulation_height/2,
    0, 0, # z coordinates are not used in 2D,
    dx, dy, 1.0 # grid spacing in x, y, and z directions
    )

#display epsilon values to confirm array is correct before running simulation
eps_array = eps.get_values_in(domain)
plt.figure()
plt.imshow(np.real(eps_array),origin='lower')
plt.show()


## setup the mode mode solver
neigs = 4
modes = emopt.modes.ModeFullVector(
    wavelength,
    eps,
    mu,
    domain,
    n0=np.sqrt(GaAs_Index**2),  # initial guess for the effective index,
    neigs=neigs,  # number of modes to compute
)
modes.bc = ['0','0']
modes.build()
modes.solve()

#%% Plotting
mode_index = 0
Ex = modes.get_field_interp(mode_index, 'Ex', squeeze=True)
Ey = modes.get_field_interp(mode_index, 'Ey', squeeze=True)
if(NOT_PARALLEL):
    print('Effective index = {:.4}'.format(modes.neff[0].real))

    eps_arr = eps.get_values_in(domain)

    vmin = np.min(np.abs(Ex))
    vmax = np.max(np.abs(Ex))
    f, axs = plt.subplots(1,2)
    f.suptitle(f'Mode Index: {mode_index}')
    im1 = axs[0].imshow(np.abs(Ex),
                   extent=[-simulation_width/2,simulation_width/2,
                           -simulation_height/2,simulation_height/2],
                   vmin=vmin,
                   vmax=vmax, cmap='hot', origin='lower')
    axs[0].set_title('Ex Field Component')
    
    vmin = np.min(np.abs(Ey))
    vmax = np.max(np.abs(Ey))
    im2 = axs[1].imshow(np.abs(Ey),
                        extent=[-simulation_width/2,simulation_width/2,
                                -simulation_height/2,simulation_height/2],
                        vmin=vmin,
                        vmax=vmax, cmap='hot', origin='lower')
    axs[1].set_title('Ey Field Component')
    
    #plot cavity wireframe for both components
    for ax in axs:
        ax.plot(GaAs_cavity_1.wireframe_xs,
                GaAs_cavity_1.wireframe_ys,
                'w-', linewidth=1.0)
    
        ax.plot(GaAs_cavity_2.wireframe_xs,
                GaAs_cavity_2.wireframe_ys,
                'w-', linewidth=1.0)

    f.colorbar(im1)
    f.colorbar(im2)
    plt.show()