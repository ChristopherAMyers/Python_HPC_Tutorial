import multiprocessing as mp
import time
import numpy as np
import Cube # custom gaussian cube file reader
from os.path import *
import numba
from math import sqrt

AU_2_EV = 27.211396 # convert from atomic units to electron volts

#   notice the parallel=True argument
@numba.jit(nopython=True, fastmath=True, parallel=True)
def _calc_coulomb_numba_parallel(pts_1, rho_1, pts_2, rho_2, dV):
    total = 0.0
    n_pts_1 = len(pts_1)
    n_pts_2 = len(pts_2)
    for i in numba.prange(n_pts_1): # prange tells numba where to parallelize the loop
        for j in range(n_pts_2):

            x1, y1, z1 = pts_1[i]
            x2, y2, z2 = pts_2[j]
            dx = x1 - x2
            dy = y1 - y2
            dz = z1 - z2
            r = sqrt(dx**2 + dy**2 + dz**2)
            total += rho_1[i]*rho_2[j]/r
            
    return total*dV

def calc_coulomb_MP_numba_external(n_proc, pts_1, rho_1, pts_2, rho_2, dV):
    #   outer loop will be split by each process
    pts_1_split = np.array_split(pts_1, n_proc)
    rho_1_split = np.array_split(rho_1, n_proc)

    #   inner loop will remain the same, so we simply copy the data
    pts_2_copies = [pts_2]*n_proc
    rho_2_copies = [rho_2]*n_proc

    #   a copy also needs to be supplied to each process
    dV_list = [dV]*n_proc

    with mp.Pool(n_proc) as pool:
        func_params = zip(pts_1_split, rho_1_split, pts_2_copies, rho_2_copies, dV_list)
        results = pool.starmap(_calc_coulomb_numba_parallel, func_params) #   call the numba version with multiple threads

    return np.sum(results)


if __name__ == '__main__':

    data_1 = Cube.CubeData(join('CV_data', 'transdens_1_low.cub'))
    data_2 = Cube.CubeData(join('CV_data', 'transdens_2_low.cub'))
    dV_12 = data_1.dV * data_2.dV
    data_1_L = Cube.CubeData(join('CV_data', 'transdens_1_extra_low.cub'))
    data_2_L = Cube.CubeData(join('CV_data', 'transdens_2_extra_low.cub'))
    dV_12_L = data_1_L.dV * data_2_L.dV

    print("Number of points in regular cubes: ", data_1.coords.shape, data_2.coords.shape)
    print("                                   ", data_1.cube_data.shape, data_2.cube_data.shape)

    print("Number of points in reduced cubes: ", data_1_L.coords.shape, data_2_L.coords.shape)
    print("                                   ", data_1_L.cube_data.shape, data_2_L.cube_data.shape)

    point_ratio = data_1.n_points * data_2.n_points/(data_1_L.n_points * data_2_L.n_points)
    print("Point ratio: ", point_ratio)

    #   run once so that numba can compile the function before we benchmark
    N_PROCESS = 2
    N_THREADS = 4
    numba.set_num_threads(N_THREADS)
    total = calc_coulomb_MP_numba_external(N_PROCESS, data_1.coords, data_1.cube_data, data_2.coords, data_2.cube_data, dV_12)

    start = time.time()
    total = calc_coulomb_MP_numba_external(N_PROCESS, data_1.coords, data_1.cube_data, data_2.coords, data_2.cube_data, dV_12)
    total_time = (time.time() - start)

    print(f'multiprocess_numpy: {total_time:.2f} s ({total*AU_2_EV} a.u.)')
    np.savetxt('_combo_external.txt', [total_time, total])
    print("Exting external program\n\n")