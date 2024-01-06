#!/usr/bin/python

# (c) Aleksey A. Kocherzhenko, January 7, 2016

#   Imports
import sys
import math
import numpy as np
import numba

class CubeData():
    #   original cube file reader from 
    #   Aleksey A. Kocherzhenko, January 7, 2016
    def __init__(self, infile, dtype=np.float64) -> None:
        pass

        linenum = 0
        xvector = []
        yvector = []
        zvector = []
        atomnum = 1000000

        xnum = 1
        ynum = 1
        znum = 0

        cube = {}

        numpoints = 0

        # READ IN CUBE FILE GIVEN AS THE ARGUMENT IN THE COMMAND LINE
        print("Reading cube file")
        for line in open(infile):
            linenum += 1
            if linenum < 3: continue
            elif linenum == 3:
                elements = line.split()
                atomnum = int(elements[0])    # Total number of atoms in the molecule
                # Get coordinate system origin:
                xorigin = float(elements[1]) 
                yorigin = float(elements[2])
                zorigin = float(elements[3])
            elif linenum < 7:
                elements = line.split()
                if linenum == 4:
                    xsize = int(elements[0])    # Number of grid points in the x direction
                    for i in range(3): xvector.append(float(elements[i+1]))   # x-axis vector
        #        print xsize, xvector
                if linenum == 5:
                    ysize = int(elements[0])    # Number of grid points in the y direction
                    for i in range(3): yvector.append(float(elements[i+1]))   # y-axis vector
        #        print ysize, yvector
                if linenum == 6:
                    zsize = int(elements[0])    # Number of grid points in the z direction
                    for i in range(3): zvector.append(float(elements[i+1]))   # z-axis vector
        #        print zsize, zvector
                    # Check that the x-, y-, and z-axis vectors are all perpendicular
                    if ( (xvector[0]*yvector[0] + xvector[1]*yvector[1] + xvector[2]*yvector[2] > 1.e-6) or
                                (xvector[0]*zvector[0] + xvector[1]*zvector[1] + xvector[2]*zvector[2] > 1.e-6) or
                                (yvector[0]*zvector[0] + yvector[1]*zvector[1] + yvector[2]*zvector[2] > 1.e-6) ):
                        print("Grid not rectangular")
                        break
                    # Calculate the length of the unit cell in x, y, and z directions
                    else:
                        xunit = math.sqrt(xvector[0]**2 + xvector[1]**2 + xvector[2]**2)
                        yunit = math.sqrt(yvector[0]**2 + yvector[1]**2 + yvector[2]**2)
                        zunit = math.sqrt(zvector[0]**2 + zvector[1]**2 + zvector[2]**2)

            # Calculate the total number of valence electrons
            elif linenum < 7 + atomnum:
                continue

            # Read in density for all grid points
            else:
                elements = line.split()
                for el in elements:
                    znum += 1
                    if znum > zsize:
                        znum = 1
                        ynum += 1
                        if ynum > ysize:
                            ynum = 1
                            xnum += 1
                    # dict key specifies position of unit volume, value specifies density within unit volume
                    cube[(znum-1)+(ynum-1)*zsize+(xnum-1)*ysize*zsize+1] = float(el)
        #        print cube[(znum-1)+(ynum-1)*zsize+(xnum-1)*ysize*zsize+1]
                    numpoints += 1


        # Integrate transition density over all space and correct for the integral not being exactly zero
        posit = 0
        negat = 0
        for key in sorted(cube.keys()):
            if cube[key] <= 0: negat += cube[key]
            if cube[key] > 0: posit += cube[key]
        epsilon = negat+posit

        unitvolume = xunit*yunit*zunit

        # Correction used in Brent Krueger's original paper
        #  correction = -(negat+posit)/key
        #  negat *= unitvolume
        #  posit *= unitvolume

        # Scaling, as done by Donghyun Lee
        pos_scale = (posit-0.5*epsilon)/posit
        neg_scale = (negat-0.5*epsilon)/negat

        tot = 0
        for key in sorted(cube.keys()):
        # Brent Krueger's original method
        #    cube[key] += correction
        # Donghyun Lee's version
            if cube[key] <= 0: cube[key] *= neg_scale
            if cube[key] > 0: cube[key] *= pos_scale
            tot += cube[key]

        # format cube to file in input format for TDC calculations

        print("Formatting")
        self.coords = np.zeros((len(cube), 3), dtype=dtype)
        self.cube_data = np.zeros(len(cube), dtype=dtype)
        self.dV = unitvolume

        for key in sorted(cube.keys()):
            # Reconstruct x, y, and z coordinate of unit volume from key
            xnum = ((key-1) // (ysize*zsize)) + 1
            ynum = (((key-1) - (xnum-1)*ysize*zsize) // zsize) + 1
            znum = ((key-1) - (ynum-1)*zsize - (xnum-1)*ysize*zsize) + 1
            # Get actual coordinate value for volume
            xcoord = xorigin + (xnum-1)*xvector[0] + (ynum-1)*yvector[0] + (znum-1)*zvector[0]
            ycoord = yorigin + (xnum-1)*xvector[1] + (ynum-1)*yvector[1] + (znum-1)*zvector[1]
            zcoord = zorigin + (xnum-1)*xvector[2] + (ynum-1)*yvector[2] + (znum-1)*zvector[2]
            # output.write('{:>15} {:>15} {:>15} {:>20}\n'.format(xcoord, ycoord, zcoord, cube[key]*unitvolume))
    
            self.coords[key-1][0] = xcoord
            self.coords[key-1][1] = ycoord
            self.coords[key-1][2] = zcoord
            self.cube_data[key-1] = cube[key]

        del cube

        self.n_points = len(self.coords)

        print("Done")


    def trim(self, cutoff=0.00001):
        keep_idx = np.where(np.abs(self.cube_data) > cutoff)
        self.coords = self.coords[keep_idx]
        self.cube_data = self.cube_data[keep_idx]


    def get_dipole(self):
        mu_x = np.sum(self.coords[:, 0] * self.cube_data)*self.dV
        mu_y = np.sum(self.coords[:, 1] * self.cube_data)*self.dV
        mu_z = np.sum(self.coords[:, 2] * self.cube_data)*self.dV
        return -np.array((mu_x, mu_y, mu_z))

    def calc_coulomb(self, cube2: 'CubeData'):
        coords_1 = self.coords
        cube_data_1 = self.cube_data*self.dV
        coords_2 = cube2.coords
        cube_data_2 = cube2.cube_data*cube2.dV
        # dV2 = self.dV * cube2.dV
        return CubeData._calc_coulomb(coords_1, cube_data_1, coords_2, cube_data_2)
    
    def save_formated(self, file_loc: str):
        out_data = np.zeros((self.cube_data.shape[0], 4))
        out_data[:, 0:3] = _cube.coords
        out_data[:, 3] = _cube.cube_data
        np.savetxt(file_loc, out_data)

    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def _calc_coulomb(coords_1: np.ndarray, cube_data_1: np.ndarray, coords_2: np.ndarray, cube_data_2: np.ndarray):

        size_1 = coords_1.shape[0]
        total = 0.0
        count = 0.0
        n_threads = numba.get_num_threads()
        for i in numba.prange(size_1):
            if count % 1000 == 0 and numba.get_thread_id() == 0:
                pct = np.round(100.0*count*n_threads/size_1, 2)
                print(count*n_threads, size_1, pct)

            dr = coords_2 - coords_1[i]
            dr2 = dr*dr
            r = np.sqrt(dr2[:, 0] + dr2[:, 1] + dr2[:, 2])
            total += np.sum(cube_data_1[i]*cube_data_2/r)
            count += 1

        return total

if __name__ == '__main__':
    _cube = CubeData(sys.argv[1])
    print(_cube.get_dipole())
    if len(sys.argv) > 2:
        _cube_2 = CubeData(sys.argv[2])
        print(_cube_2.get_dipole())

        coulomb = _cube_2.calc_coulomb(_cube)
        print(coulomb)


    # a = cube.coords
    # b = np.transpose([cube.cube_data])
    # np.save('tmp', np.hstack((a, b)))
