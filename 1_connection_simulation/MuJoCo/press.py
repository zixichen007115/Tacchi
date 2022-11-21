from mujoco_py import load_model_from_path, MjSim, MjViewer
from scipy import interpolate
import taichi as ti
import numpy as np
from depth_generation import generate
import cv2
import argparse
import os

if not os.path.exists("sim"):
    os.makedirs("sim")

if not os.path.exists("depth"):
    os.makedirs("depth")

# parameters
parser = argparse.ArgumentParser()
parser.add_argument("--object", default="dot_in")
parser.add_argument("--particle", default="1", choices=["1","10","100"])
parser.add_argument("--x", type=int, default=0)
parser.add_argument("--y", type=int, default=0)

args = parser.parse_args()

# taichi initialization
arch =  ti.vulkan if ti._lib.core.with_vulkan() else  ti.cuda
ti.init(arch=arch)

# mujoco initialization
xml_file = "xmls/mujoco_" + args.object + ".xml"
model = load_model_from_path(xml_file)
sim = MjSim(model)
viewer = MjViewer(sim)

# world initialization
t_ti = ti.field(dtype=ti.f32, shape=())
t_ti[None] = 0
dt = 1e-4

# taichi gel and indenter initialization
num_l = num_w = 100+1
num_h = 20+1

l = 20
w = 20
h = 4

dis = l/(num_l-1)

obj_name = "obj_" + args.particle + "/" + args.object + ".npy"

data_ = np.load(obj_name)
obj = data_.copy()
if args.object == "cross_lines" or args.object == "cylinder_side" or args.object == "hexagon" or args.object == "line":
  obj[:,0]=data_[:,1]
  obj[:,1]=data_[:,0]
elif args.object == "dots":
  obj[:,1]=-data_[:,1]
elif args.object == "wave1":
  obj[:,0]=-data_[:,0]
elif args.object == "moon" or args.object == "pacman":
  obj[:,0]=data_[:,1]
  obj[:,1]=-data_[:,0]
elif args.object == "triangle":
  obj[:,0]=-data_[:,1]
  obj[:,1]=data_[:,0]
num_obj = np.shape(obj)[0]

# grid initialization
n_grid = 256
dx = 33 / n_grid
inv_dx = 1 / dx
grid_v = ti.Vector.field(3, dtype=float, shape=(n_grid, n_grid, n_grid)) # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid)) # grid node mass

# particle initialization
n_particles = num_l*num_w*num_h+num_obj
x = ti.Vector.field(3, dtype=float, shape=n_particles) # position
x_2d = ti.Vector.field(2, dtype=float, shape=n_particles) # 2d positions - this is necessary for circle visualization
x_surface = ti.Vector.field(3, dtype=float, shape=num_l*num_w)
v = ti.Vector.field(3, dtype=float, shape=n_particles) # velocity
C = ti.Matrix.field(3, 3, dtype=float, shape=n_particles) # affine velocity field
F = ti.Matrix.field(3, 3, dtype=float, shape=n_particles) # deformation gradient
material = ti.field(dtype=int, shape=n_particles) # material id
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 1.45e5, 0.45 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters - may change these later to model other materials

# image parameters
img_length = 640
img_width = 480
xi = np.linspace(12.5, 27.5, num=img_length)
yi = np.linspace(14.375, 25.625, num=img_width)

def simulate():
    sim.step()
    viewer.render()

@ti.kernel
def initialize_taichi(obj: ti.types.ndarray(),num_obj: ti.f32, ind_x: ti.f32, ind_y: ti.f32):
  for i,j,k in ti.ndrange(num_l,num_w,num_h):
    m = i+j*num_l+k*num_l*num_w
    offest = ti.Vector([20-l/2,20-w/2,10-h/2])
    x[m] = ti.Vector([i,j,k])*dis+offest
    x_2d[m] = [x[m][0], x[m][1]]
    v[m] = [0, 0, 0]
    material[m] = 0
    F[m] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    C[m] = ti.Matrix.zero(float, 3, 3)

  for i in ti.ndrange(num_obj):
    m = i+num_l*num_w*num_h
    offest = ti.Vector([20-ind_y,20-ind_x,33])
    x[m] = ti.Vector([-obj[i,0],-obj[i,1],-obj[i,2]])+offest
    x_2d[m] = [x[m][0], x[m][1]]
    v[m] = [0, 0, 0]
    material[m] = 1
    F[m] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    C[m] = ti.Matrix.zero(float, 3, 3)

@ti.kernel
def substep(vel: ti.f32):
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
      new_v = ti.Vector([0,0,-vel])

    v[p], C[p] = new_v, new_C
    
    # move the particles
    x[p] += dt * v[p] 
    x_2d[p] = [x[p][1]*3-10, x[p][2]*3-10] # update 2d positions
    F[p] = (ti.Matrix.identity(float, 3) + (dt * new_C)) @ F[p] #update F (explicitMPM way)

    # collect surface particle positions
    if p < num_l*num_w*num_h and p >= num_l*num_w*(num_h-1):
    	p_surf = p-num_l*num_w*(num_h-1)
    	x_surface[p_surf] = x[p]

def timg_generation(ind_z, img_name):

  x_surf = x_surface.to_numpy()

  xp = x_surf[:,0]
  yp = x_surf[:,1]
  p_depth = (x_surf[:,2]-12)/1000+0.03

  i_depth = interpolate.griddata((xp, yp), p_depth, (xi[None,:],yi[:,None]), method='cubic', fill_value=0.03)
  i_depth = i_depth.astype(np.float32)

  npy_name = "depth/"+ img_name +".npy"
  np.save(npy_name,i_depth)

  img = generate(i_depth)
  img_name = "sim/"  + img_name +".png"
  cv2.imwrite(img_name,img)

  cv2.imshow("tactile_image_MuJoCo",img)
  cv2.waitKey(5)

# taichi gui
gui = ti.GUI("Tacchi", res=400, background_color=0x112F41)
colors = np.array([0x808080,0x008000,0xEEEEF0], dtype=np.uint32)

# initialization
initialize_taichi(obj, num_obj, args.x, args.y)
z_ref = 12
sim.data.qpos[1]=args.x*1e-3
sim.data.qpos[2]=args.y*1e-3

# step sim
while z_ref>=11:

  # MuJoCo sim
  sim.data.ctrl[0] = 0.2
  simulate()
  vel = sim.data.qvel[0]*1e3

  # Tacchi sim
  substep(vel)

  # rendering
  x_surf = x_surface.to_numpy()
  z = np.min(x_surf[:,2])

  if np.abs(z-z_ref)<1e-2:
  	ind = int((12-z_ref)*10+0.1)
  	print(ind)
  	img_name = args.object+"__"+str((args.x+1)*33+(args.y+1)*11+ind+1)+"__%d_%d_%d"%(args.x,args.y,ind)
  	timg_generation(z_ref, img_name)
  	z_ref-=0.1

  # taichi gui
  gui.circles((x_2d.to_numpy())/100, radius=1, color=colors[material.to_numpy()])
  gui.show()

