import taichi as ti
import numpy as np
import argparse
import os
import sys



# parameters
parser = argparse.ArgumentParser()
parser.add_argument("--object", default="dot_in")
parser.add_argument("--particle", default="1", choices=["1","10","100"])
parser.add_argument("--x", type=int, default=0)
parser.add_argument("--y", type=int, default=0)

args = parser.parse_args()

if not os.path.exists("surface"):
    os.makedirs("surface")

# taichi initialization
ti.init(arch=ti.gpu) 

# world initialization
t_ti = ti.field(dtype=ti.f32, shape=())
t_ti[None] = 0
dt = 1e-4

# gel and indenter initialization
num_l = num_w = 100+1
num_h = 20+1

l = 20
w = 20
h = 4

dis = l/(num_l-1)

obj_name = "obj_" + args.particle + "/" + args.object + ".npy"
data_ = np.load(obj_name)
data = data_.copy()
if args.object == "cross_lines" or args.object == "cylinder_side" or args.object == "hexagon" or args.object == "line":
  data[:,0]=data_[:,1]
  data[:,1]=data_[:,0]
elif args.object == "dots":
  data[:,1]=-data_[:,1]
elif args.object == "wave1":
  data[:,0]=-data_[:,0]
elif args.object == "moon" or args.object == "pacman":
  data[:,0]=data_[:,1]
  data[:,1]=-data_[:,0]
elif args.object == "triangle":
  data[:,0]=-data_[:,1]
  data[:,1]=data_[:,0]
print(np.shape(data)[0])


# grid initialization
n_grid = 256
dx = 33 / n_grid
inv_dx = 1 / dx
grid_v = ti.Vector.field(3, dtype=float, shape=(n_grid, n_grid, n_grid)) # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid)) # grid node mass

# particle initialization
n_particles = num_l*num_w*num_h+np.shape(data)[0]
x = ti.Vector.field(3, dtype=float, shape=n_particles) # position
x_2d = ti.Vector.field(2, dtype=float, shape=n_particles) # 2d positions - this is necessary for circle visualization
v = ti.Vector.field(3, dtype=float, shape=n_particles) # velocity
C = ti.Matrix.field(3, 3, dtype=float, shape=n_particles) # affine velocity field
F = ti.Matrix.field(3, 3, dtype=float, shape=n_particles) # deformation gradient
material = ti.field(dtype=int, shape=n_particles) # material id


p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 1.45e5, 0.45 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters - may change these later to model other materials

# np particle position 
p_xpos_list = np.empty([0,num_w * num_l])
p_ypos_list = np.empty([0,num_w * num_l])
p_zpos_list = np.empty([0,num_w * num_l])

@ti.kernel
def substep():
  for i, j, k in grid_m:
    grid_v[i, j, k] = [0, 0, 0]
    grid_m[i, j, k] = 0

  # particle to grid
  for p in x: 

    # first for particle p, compute base index
    base = (x[p] * inv_dx - 0.5).cast(int)
    
    # quadratic kernels 
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    mu, la = mu_0, lambda_0 
    U, sig, V = ti.svd(F[p])
    J = 1.0
    for d in ti.static(range(3)):
      J *= sig[d, d]

    stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 3) * la * J * (J - 1)
    stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
    affine = stress + p_mass * C[p]

    #P2G for velocity and mass 
    for i, j, k in ti.static(ti.ndrange(3, 3, 3)): # Loop over 3x3x3 grid node neighborhood
      offset = ti.Vector([i, j, k])
      dpos = (offset.cast(float) - fx) * dx
      weight = w[i][0] * w[j][1] * w[k][2]
      grid_m[base + offset] += weight * p_mass #mass transfer
      grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
  
  # grid operation
  for i, j, k in grid_m:
    if grid_m[i, j, k] > 0: 
      grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k] # momentum to velocity
      
      #wall collisions - handle all 3 dimensions
      if i < 3 and grid_v[i, j, k][0] < 0:          grid_v[i, j, k][0] = 0 # Boundary conditions
      if i > n_grid - 3 and grid_v[i, j, k][0] > 0: grid_v[i, j, k][0] = 0
      if j < 3 and grid_v[i, j, k][1] < 0:          grid_v[i, j, k][1] = 0
      if j > n_grid - 3 and grid_v[i, j, k][1] > 0: grid_v[i, j, k][1] = 0
      if k < 3 and grid_v[i, j, k][2] < 0:          grid_v[i, j, k][2] = 0
      if k > n_grid - 3 and grid_v[i, j, k][2] > 0: grid_v[i, j, k][2] = 0
  
  # grid to particle
  for p in x: 

    # compute base index
    base = (x[p] * inv_dx - 0.5).cast(int)

    # quadratic kernels
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]

    new_v = ti.Vector.zero(float, 3)
    new_C = ti.Matrix.zero(float, 3, 3)
    new_F = ti.Matrix.zero(float, 3, 3)

    for i, j, k in ti.static(ti.ndrange(3, 3, 3)): # loop over 3x3x3 grid node neighborhood
      dpos = ti.Vector([i, j, k]).cast(float) - fx
      g_v = grid_v[base + ti.Vector([i, j, k])]
      weight = w[i][0] * w[j][1] * w[k][2]
      new_v += weight * g_v
      new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

    # particle operation
    if p < num_l*num_w*3:
      new_v = ti.Vector([0,0,0])
    if material[p] == 1:
      new_C = ti.Matrix.zero(float, 3, 3)
      new_v = ti.Vector([0,0,-200])

    v[p], C[p] = new_v, new_C

    # move the particles
    x[p] += dt * v[p] 
    x_2d[p] = [x[p][1]*3-10, x[p][2]*3-10] # update 2d positions
    F[p] = (ti.Matrix.identity(float, 3) + (dt * new_C)) @ F[p] #update F (explicitMPM way)


def save(p_xpos_list, p_ypos_list, p_zpos_list):
    x_ = x.to_numpy()

    p_xpos = x_[num_l*num_w*(num_h-1):num_l*num_w*num_h,0]
    p_ypos = x_[num_l*num_w*(num_h-1):num_l*num_w*num_h,1]
    p_zpos = x_[num_l*num_w*(num_h-1):num_l*num_w*num_h,2]

    p_xpos = np.reshape(p_xpos,(1,num_l*num_w))
    p_ypos = np.reshape(p_ypos,(1,num_l*num_w))
    p_zpos = np.reshape(p_zpos,(1,num_l*num_w))

    p_xpos_list = np.concatenate((p_xpos_list,p_xpos))
    p_ypos_list = np.concatenate((p_ypos_list,p_ypos))
    p_zpos_list = np.concatenate((p_zpos_list,p_zpos))

    return p_xpos_list, p_ypos_list, p_zpos_list

@ti.kernel
def initialize(data: ti.types.ndarray(),data_len: ti.f32, ind_x: ti.f32, ind_y: ti.f32):
  for i,j,k in ti.ndrange(num_l,num_w,num_h):
    m = i+j*num_l+k*num_l*num_w
    offest = ti.Vector([20-l/2,20-w/2,10-h/2])
    x[m] = ti.Vector([i,j,k])*dis+offest
    x_2d[m] = [x[m][0], x[m][1]]
    v[m] = [0, 0, 0]
    material[m] = 0
    F[m] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    C[m] = ti.Matrix.zero(float, 3, 3)


  
  for i in ti.ndrange(data_len):
    m = i+num_l*num_w*num_h
    offest = ti.Vector([20-ind_y,20-ind_x,33])
    x[m] = ti.Vector([-data[i,0],-data[i,1],-data[i,2]])+offest
    x_2d[m] = [x[m][0], x[m][1]]
    v[m] = [0, 0, 0]
    material[m] = 1
    F[m] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    C[m] = ti.Matrix.zero(float, 3, 3)

# taichi gui
gui = ti.GUI("Explicit MPM rotate", res=400, background_color=0x112F41)
colors = np.array([0x808080,0x00ff00,0xEEEEF0], dtype=np.uint32)

# initialization
initialize(data, np.shape(data)[0], args.x, args.y)

# step sim
while True:
  t_ti[None] += 1
  p_xpos_list, p_ypos_list, p_zpos_list = save(p_xpos_list, p_ypos_list, p_zpos_list)
  depth = np.min(p_zpos_list[-1,:])


  substep()
  if t_ti[None]>=30:
    if (depth-11)>-1e-2:
      if t_ti[None]%2==0:
        p_xpos_list, p_ypos_list, p_zpos_list = save(p_xpos_list, p_ypos_list, p_zpos_list)
    # print(np.min(p_zpos_list[-1,:]))

    else:
      np.savez('surface/' + args.particle + '_' +args.object,p_xpos_list=p_xpos_list, p_ypos_list=p_ypos_list, p_zpos_list=p_zpos_list)
      print("press saved")
      sys.exit()

  # taichi gui
  # gui.circles(x_2d.to_numpy()/100, radius=1, color=colors[material.to_numpy()])
  # gui.show() 

  print(t_ti[None])