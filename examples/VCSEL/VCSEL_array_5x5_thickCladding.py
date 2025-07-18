import emopt
from emopt.misc import NOT_PARALLEL

import numpy as np
import matplotlib.pyplot as plt
from math import pi


####################################################################################
# Set up the size of the problem
####################################################################################
plt.close('all')
unit_cell_size = 3.0 # microns
nominal_width = 1.2 # microns
separable_array_size = [5,5]
dx = 0.1 # microns
dy = 0.1 # microns
wavelength = 0.85 # microns
simulation_height = (separable_array_size[1])*unit_cell_size+15 # microns
simulation_width = (separable_array_size[0])*unit_cell_size+15 # microns
N = int(np.ceil((simulation_width)/dx)+1)
M = int(np.ceil((simulation_height)/dy)+1)
simulation_width =  (N-1)*dx # microns
simulation_height = (M-1)*dy # microns


GaAs_Index = 3.4
AlGaAsO_Index = 3.4-0.1 # large for now, will change to 0.02

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
cavity_temp.translate(
                        (-(separable_array_size[0]-1)/2)*unit_cell_size,
                        (-(separable_array_size[1]-1)/2)*unit_cell_size
                      )

#iterate through array indices and add primitives into the epsilon matrix object
cavities = []
for index in indices:
    xs = cavity_temp.xs + index[0]*unit_cell_size
    ys = cavity_temp.ys + index[1]*unit_cell_size
    cavity_ij = emopt.grid.Polygon(
        xs = xs,
        ys = ys
    )
    cavity_ij.layer = 1
    cavity_ij.material_value = GaAs_Index**2  # GaAs permittivity
    cavities.append(cavity_ij)
    
for cavity in cavities:
    eps.add_primitive(cavity)

cladding = emopt.grid.Rectangle(
    x0=0, 
    y0=0,
    xspan=simulation_width,
    yspan=simulation_height,
)
cladding.layer = 2
cladding.material_value = AlGaAsO_Index**2

core = emopt.grid.Polygon(
    xs=[
        -unit_cell_size/2, 
        -unit_cell_size/2,
        -nominal_width/2, nominal_width/2,
        unit_cell_size/2, unit_cell_size/2,
        nominal_width/2, -nominal_width/2
        ],
    ys=[
        -nominal_width/2, 
        nominal_width/2,
        unit_cell_size/2, unit_cell_size/2,
        nominal_width/2, -nominal_width/2,
        -unit_cell_size/2, -unit_cell_size/2
        ]
)
# core = emopt.grid.Rectangle(x0=0,y0=0,xspan=1,yspan=1)
core.layer = 1
core.material_value = GaAs_Index**2  
eps.add_primitive(cladding)
# eps.add_primitive(core)

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

#%%
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

DoF = N*M*6 # 6 field vectors on each grid
dense_matrix_ram_GB = (N*M*6)**2 * 4 /1024**3 # int32 takes 4 bytes
nnz = DoF * 13 #high-end approximation of the number of non-zero numbers 
aprox_sparse_matrix_ram_GB = (8 * nnz + 4 * (DoF + 1)) / 1024**3
print(f'{N=},  {M=},  {DoF=}')
print(f'{dense_matrix_ram_GB=:.2f}')
print(f'{aprox_sparse_matrix_ram_GB=:.2f}')

#%% Plotting
plt.close('all')
plotting = True

mode_plot = np.arange(0,50)
# mode_plot = [0,1]
if plotting:
    
    #display epsilon profile
    eps_array = eps.get_values_in(domain)
    plt.figure()
    plt.imshow(np.real(eps_array),origin='lower')
    
    Ex0 = modes.get_field_interp(0, 'Ex', squeeze=True)
    Ey0 = modes.get_field_interp(0, 'Ey', squeeze=True)
    #vmin = np.min(np.append(np.abs(Ex0),np.abs(Ey0)))
    #vmax = np.max(np.append(np.abs(Ex0),np.abs(Ey0)))
    
    for mode_index in mode_plot:
        Ex = modes.get_field_interp(mode_index, 'Ex', squeeze=True)
        Ey = modes.get_field_interp(mode_index, 'Ey', squeeze=True)
        Ez = modes.get_field_interp(mode_index, 'Ez', squeeze=True)
        Hx = modes.get_field_interp(mode_index, 'Hx', squeeze=True)
        Hy = modes.get_field_interp(mode_index, 'Hy', squeeze=True)
        Hz = modes.get_field_interp(mode_index, 'Hz', squeeze=True)
        print('Effective index = {:.4}'.format(modes.neff[mode_index].real))

        eps_arr = eps.get_values_in(domain)
        
        f, axs = plt.subplots(3,2)
        f.suptitle(f'Mode Index: {mode_index}\n n_eff = {modes.neff[mode_index].real:.5f}')
        # First column: Ex, Ey, Ez
        vmin = np.min(np.abs(Ex))
        vmax = np.max(np.abs(Ex))
        im1 = axs[0,0].imshow(np.abs(Ex),
                            extent=[-simulation_width/2,simulation_width/2,
                                    -simulation_height/2,simulation_height/2],
                            vmin=vmin,
                            vmax=vmax, cmap='hot', origin='lower')
        axs[0,0].set_title('Ex Field Component')
        f.colorbar(im1, ax=axs[0,0])
        vmin = np.min(np.abs(Ey))
        vmax = np.max(np.abs(Ey))
        im2 = axs[1,0].imshow(np.abs(Ey),
                            extent=[-simulation_width/2,simulation_width/2,
                                    -simulation_height/2,simulation_height/2],
                            vmin=vmin,
                            vmax=vmax, cmap='hot', origin='lower')
        axs[1,0].set_title('Ey Field Component')
        f.colorbar(im2, ax=axs[1,0])
        vmin = np.min(np.abs(Ez))
        vmax = np.max(np.abs(Ez))
        im3 = axs[2,0].imshow(np.abs(Ez),
                            extent=[-simulation_width/2,simulation_width/2,
                                    -simulation_height/2,simulation_height/2],
                            vmin=vmin,
                            vmax=vmax, cmap='hot', origin='lower')
        axs[2,0].set_title('Ez Field Component')
        f.colorbar(im3, ax=axs[2,0])
        # Second column: Hx, Hy, Hz
        vmin = np.min(np.abs(Hx))
        vmax = np.max(np.abs(Hx))
        im4 = axs[0,1].imshow(np.abs(Hx),
                            extent=[-simulation_width/2,simulation_width/2,
                                    -simulation_height/2,simulation_height/2],
                            vmin=vmin,
                            vmax=vmax, cmap='hot', origin='lower')
        axs[0,1].set_title('Hx Field Component')
        f.colorbar(im4, ax=axs[0,1])
        vmin = np.min(np.abs(Hy))
        vmax = np.max(np.abs(Hy))
        im5 = axs[1,1].imshow(np.abs(Hy),
                            extent=[-simulation_width/2,simulation_width/2,
                                    -simulation_height/2,simulation_height/2],
                            vmin=vmin,
                            vmax=vmax, cmap='hot', origin='lower')
        axs[1,1].set_title('Hy Field Component')
        f.colorbar(im5, ax=axs[1,1])
        vmin = np.min(np.abs(Hz))
        vmax = np.max(np.abs(Hz))
        im6 = axs[2,1].imshow(np.abs(Hz),
                            extent=[-simulation_width/2,simulation_width/2,
                                    -simulation_height/2,simulation_height/2],
                            vmin=vmin,
                            vmax=vmax, cmap='hot', origin='lower')
        axs[2,1].set_title('Hz Field Component')
        f.colorbar(im6, ax=axs[2,1])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()