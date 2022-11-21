#!/usr/bin/env python

import rospy
import time
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
import cv2
import math
import numpy as np

from scipy import interpolate
import taichi as ti
from depth_generation import generate
import argparse

# parameters
parser = argparse.ArgumentParser()
parser.add_argument("--object", default="dot_in")
parser.add_argument("--particle", default="1", choices=["1","10","100"])
args = parser.parse_args()

# taichi initialization
arch =  ti.vulkan if ti._lib.core.with_vulkan() else  ti.cuda
ti.init(arch=arch)

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
obj = np.load(obj_name)
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

def timg_generation(img_name):

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
    cv2.imshow("tactile_img",img)
    cv2.waitKey(5)

# taichi gui
gui = ti.GUI("Tacchi", res=400, background_color=0x112F41)
colors = np.array([0x808080,0x008000,0xEEEEF0], dtype=np.uint32)

pub = None
rate = None
previous_position = (0, 0, 0)
# printer_speed = 0.004
printer_speed = 0.001

WS_MAX = (0.32, 0.32, 0.42)
# WS_MIN = (0, 0, 0.06)
WS_MIN = (0, 0, 0.01)
gelsight_img = None
gelsight_depth = None


def show_normalized_img(name, img):
    draw = img.copy()
    draw -= np.min(draw)
    draw = draw / np.max(draw)
    cv2.imshow(name, draw)
    return draw


def euclidean_dist(t1, t2):
    return math.sqrt(math.pow(t1[0] - t2[0], 2) + math.pow(t1[1] - t2[1], 2) + math.pow(t1[2] - t2[2], 2))


def move(x, y, z, force=False, wait=None):
    if rospy.is_shutdown():
        exit(2)

    if not force and (
            x > WS_MAX[0] or y > WS_MAX[1] or z > WS_MAX[2] or x < WS_MIN[0] or y < WS_MIN[1] or z < WS_MIN[2]):
        print('ERROR. Attempted to move to invalid position.', (x, y, z))
        exit(2)

    global previous_position
    print('Move to:', (x, y, z))
    pos = Float64MultiArray()
    pos.data = [x, y, z]
    pub.publish(pos)

    if wait is None:
        s = 3 + euclidean_dist(previous_position, (x, y, z)) / printer_speed
    else:
        s = wait
    print('Waiting seconds: ', s)
    time.sleep(s)
    previous_position = (x, y, z)


def collect_data():
    global previous_position

    x_steps = 3
    y_steps = 3
    z_steps = 10
    h_step_size = 0.001
    z_step_size = 0.0001

    # SIM
    start_x = 0.185 - 0.045
    start_y = 0.165 + 0.007
    start_z = 0.022 + 0.027

    previous_position = (0.0, 0.0, 0.0)


    starting_position = previous_position

    print('START.')
    move(0, 0, start_z, force=True, wait=20)

    # TEST
    move(start_x, start_y, start_z + 3 * z_step_size, wait=20)  # 20 for sim, 60 for real


    k = 0
    solid = args.object
    z_ref = 12


    for x_ in range(-(x_steps // 2), (x_steps // 2) + 1):
        for y in range(-(y_steps // 2), (y_steps // 2) + 1):

            initialize_taichi(obj, num_obj, x_, y)
            substep(200)
            x_surf = x_surface.to_numpy()
            z_surf = np.min(x_surf[:,2])

            p = start_x + x_ * h_step_size, start_y + y * h_step_size
            move(*(p + (start_z + z_step_size,)))
            print('========================================>')
            for z in range(z_steps + 1):
                print('--START ---->')
                print('POINT:', (x_, y, z))

                pp = p + (start_z - z * z_step_size,)
                move(*pp)
                k += 1

                while (z_surf+z/10-z_ref)>1e-3:

                    substep(100)
                    x_surf = x_surface.to_numpy()
                    z_surf = np.min(x_surf[:,2])
                    gui.circles([1,1]-x_2d.to_numpy()/100, radius=1, color=colors[material.to_numpy()])
                    gui.show()

                name = solid + '__' + str(k) + '__' + str(x_) + '_' + str(y) + '_' + str(z) + '.png'
                timg_generation(name)



                print('--END   ---->')
                time.sleep(1)

            print('========================================>')
            move(*(p + (start_z + z_step_size,)))

    # move(*starting_position)


if __name__ == '__main__':
    rospy.init_node('gelsight_simulation_dc')

    pub = rospy.Publisher('/fdm_printer/xyz_controller/command', Float64MultiArray, queue_size=10)

    rate = rospy.Rate(1)
    rate.sleep()
    print('--------------------_>>> Data Collection start.')
    collect_data()
