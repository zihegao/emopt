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
separable_array_size = [5,5]
dx = 0.75 # microns
dy = 0.75 # microns
wavelength = 0.85 # microns
simulation_height = (2+separable_array_size[0])*unit_cell_size # microns
simulation_width = (2+separable_array_size[1])*unit_cell_size # microns
N = int(np.ceil((simulation_height)/dx)+1)
M = int(np.ceil((simulation_width)/dy)+1)

GaAs_Index = 3.4
AlGaAsO_Index = 3.3

ixs, iys = np.meshgrid(np.arange(separable_array_size[0]),
                       np.arange(separable_array_size[1]),
                       indexing='ij')
indices = np.stack([ixs,iys],axis=-1).reshape(-1,2)

## Combine the primitives into a epsilon structure
eps = emopt.grid.StructuredMaterial2D(simulation_height,
                                      simulation_width,
                                      dx, 
                                      dy)

cavity_temp = emopt.grid.Polygon(
    xs=[-unit_cell_size/2, -unit_cell_size/2,
        -nominal_width/2,nominal_width/2,
        unit_cell_size/2, unit_cell_size/2,
        nominal_width/2, -nominal_width/2],
    ys=[-nominal_width/2, nominal_width/2,
        unit_cell_size/2, unit_cell_size/2,
        nominal_width/2, -nominal_width/2,
        -unit_cell_size/2, -unit_cell_size/2]
)
#place temp cavity in bottom left position of array
cavity_temp.translate(int(-separable_array_size[0]/2)*unit_cell_size,
                      int(-separable_array_size[1]/2)*unit_cell_size)

#iterate through array indices and add primitives into the epsilon matrix object
for index in indices:
    cavity_ij = cavity_temp.copy().translate(index[0]*unit_cell_size,index[1]*unit_cell_size)
    cavity_ij.layer = 1
    cavity_ij.material_value = GaAs_Index**2  # GaAs permittivity
    eps.add_primitive(cavity_ij)
    

GaAs_substrate = emopt.grid.Rectangle(
    x0=0, 
    y0=0,
    xspan=simulation_width,
    yspan=simulation_height,
)

GaAs_substrate.layer = 2
GaAs_substrate.material_value = AlGaAsO_Index**2
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
neigs = 50
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
plotting = True
if plotting:
    Ex0 = modes.get_field_interp(0, 'Ex', squeeze=True)
    Ey0 = modes.get_field_interp(0, 'Ey', squeeze=True)
    #vmin = np.min(np.append(np.abs(Ex0),np.abs(Ey0)))
    #vmax = np.max(np.append(np.abs(Ex0),np.abs(Ey0)))
    for mode_index in range(0,neigs):
        Ex = modes.get_field_interp(mode_index, 'Ex', squeeze=True)
        Ey = modes.get_field_interp(mode_index, 'Ey', squeeze=True)
        print('Effective index = {:.4}'.format(modes.neff[mode_index].real))

        eps_arr = eps.get_values_in(domain)
        
        f, axs = plt.subplots(1,2)
        f.suptitle(f'Mode Index: {mode_index}')
        vmin = np.min(np.abs(Ex))
        vmax = np.max(np.abs(Ex))
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
    
        # #plot cavity wireframe for both components
        # for ax in axs:
        #     ax.plot(GaAs_cavity_1.wireframe_xs,
        #             GaAs_cavity_1.wireframe_ys,
        #             'w-', linewidth=1.0)
    
        #     ax.plot(GaAs_cavity_2.wireframe_xs,
        #             GaAs_cavity_2.wireframe_ys,
        #             'w-', linewidth=1.0)

        f.colorbar(im1)
        f.colorbar(im2)
        plt.show()