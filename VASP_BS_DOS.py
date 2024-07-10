import csv
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import pandas as pd
import numpy.linalg as LA

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pymatgen.io.vasp.outputs import BSVasprun, Vasprun

import pickle 
from  tqdm import tqdm

import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)


import wannier_loader


class VASP_analyse_3D():
    '''
    base class for analysing qe output files
    '''
    def __init__(self, dir, name):
        self.directory = dir # './'
        self.name = name # 'CrTe2'
        self.parse_xml()
        self.parse_vaspkit_kpath()

    def parse_xml(self):
        self.run = BSVasprun(  './VASP/BS/vasprun.xml', parse_projected_eigen=True)
        self.bs = self.run.get_band_structure()
        acell = self.run.final_structure.lattice.matrix

        self.spin_keys = list(self.bs.bands.keys())
        self.FMq = (len(self.spin_keys) == 2)

        self.nbandsDFT = len(self.bs.bands[self.spin_keys[0]])
        
        self.efermi = self.run.efermi
        
        V = LA.det(acell)
        print(f'Unit Cell Volume:   {V:.4f}  (Ang^3)')
        b1 = 2*np.pi*np.cross(acell[1], acell[2])/V
        b2 = 2*np.pi*np.cross(acell[2], acell[0])/V
        b3 = 2*np.pi*np.cross(acell[0], acell[1])/V
        self.bcell = np.array([b1, b2, b3])
        self.acell = acell
        
        print('Reciprocal-Space Vectors (Ang^-1)')
        with printoptions(precision=10, suppress=True):
            print(self.bcell)

        print('Real-Space Vectors (Ang)')
        with printoptions(precision=10, suppress=True):
            print(acell)


    def parse_vaspkit_kpath(self):
        ''' 
        need to parse KPOINTS file to get kpt names
        '''
        
        self.HighSymPointsNames = []
        self.HighSymPointsDists = []
        self.HighSymPointsCoords = []
        with open('./VASP/BS/KPOINTS') as fin:
            lines = fin.readlines()
        
            # Extract the number of k-points
            num_kpoints = int(lines[1].strip())


            line = lines[4].strip().split()
            Letter_prev = line[3]
            dist = 0.0
            k_prev = np.array(list(map(float, line[:3])))
            self.HighSymPointsNames.append(Letter_prev)
            self.HighSymPointsDists.append(dist)
            self.HighSymPointsCoords.append(k_prev)

            for kpt_ind in range(1, len(lines[4:])//3):
                line = lines[kpt_ind*3+4].strip().split()
                # print(line)
                Letter_new = line[3]
                k_new = np.array(list(map(float, line[:3])))
            
                delta_k = k_new - k_prev
                dist += LA.norm(self.bcell.T@delta_k)
                k_prev = k_new
                self.HighSymPointsNames.append(Letter_new)
                self.HighSymPointsDists.append(dist)
                self.HighSymPointsCoords.append(k_prev)
            
            line = lines[kpt_ind*3+5].strip().split()
            # print(line)
            Letter_new = line[3]
            k_new = np.array(list(map(float, line[:3])))
        
            delta_k = k_new - k_prev
            dist += LA.norm(self.bcell.T@delta_k)
            k_prev = k_new
            self.HighSymPointsNames.append(Letter_new)
            self.HighSymPointsDists.append(dist)
            self.HighSymPointsCoords.append(k_prev)
        
        # print(self.HighSymPointsNames)
        # print(self.HighSymPointsDists)
    
    
    def plot_BS(self):
        '''
        plots nonspinpolarized BS
        '''
        label_ticks = self.HighSymPointsNames
        normal_ticks = self.HighSymPointsDists

        fig, dd = plt.subplots() 

        if self.FMq:
            for spin in self.spin_keys:
                # print(spin)
                for band in range(self.nbandsDFT):
                    dd.plot(self.bs.distance, self.bs.bands[spin][band] - self.efermi,
                                color='b', linewidth=0.7, alpha=1.0,
                                label='up' if band == 0 else "")
            dd.legend(prop={'size': 8}, loc='upper right', frameon=False)  

        else:
            for band in range(self.nbandsDFT):
                dd.plot(self.bs.distance, self.bs.bands[self.spin_keys[0]][band] - self.efermi,
                            color='b', linewidth=0.7, alpha=1.0)
                
        dd.set_ylabel(r'E - $E_f$ [Ev]') 
        plt.xticks(normal_ticks, label_ticks)
        dd.yaxis.set_minor_locator(MultipleLocator(1))
        plt.grid(axis='x')
        dd.axhline(y=0, ls='--', color='k')
        plt.xlim(normal_ticks[0], normal_ticks[-1])
        plt.ylim(-10, 10)

        width = 7
        fig.set_figwidth(width)     
        fig.set_figheight(width/1.6)
        #plt.savefig('./2pub/pics/BS.png', dpi=200, bbox_inches='tight')

        plt.show()


    def print_bands_range(self, band_from=None, band_to=None):
        '''
        prints energy ranges of selected bands
        '''
        if band_from is None:
            band_from = 0
        if band_to is None:
            band_to = self.nbandsDFT

        print(f'efermi {self.efermi:.2f}')


        for ind, spin in enumerate(self.spin_keys):
            if ind == 0:
                if self.FMq == 1: 
                    print("-------------SPIN UP---------------")
                else:
                    print("------------- PM ---------------")
            else:
                print("-------------SPIN DN---------------")

            for band_num in range(band_from,band_to):
                print(f'band {band_num+1} eV from  {min(self.bs.bands[spin][band_num]) :.2f} to  {max(self.bs.bands[spin][band_num]) :.2f} \
                    eV-eF from  {min(self.bs.bands[spin][band_num]) -self.efermi :.2f} to  {max(self.bs.bands[spin][band_num]) - self.efermi:.2f}' )
        

            
    def get_full_DOS(self):
    
        dosrun = Vasprun('./VASP/DOS/vasprun.xml', parse_dos=True)
        dos = dosrun.complete_dos

        self.eDOS = dos.energies
        self.dosup = dos.densities[self.spin_keys[0]]
        if self.FMq:
            self.dosdn = dos.densities[self.spin_keys[1]]

        self.dos = dos
        self.dosEfermi = dosrun.efermi
        

    def plot_FullDOS(self, saveQ=False, picname='DOS'):
        '''
        plots full dos of the system
        '''
        fig, dd = plt.subplots() 
        
        
        if self.FMq:
            for spin, color in [('up', 'red'), ('down', 'blue')]:
                doss = self.dosup if spin == 'up' else self.dosdn
                dd.plot(self.eDOS - self.dosEfermi, doss, 
                        label=spin, color=color, linewidth=0.5)
            dd.legend(prop={'size': 8}, loc='upper right', frameon=False)  # Add a legend.
            plt.fill_between(
                    x= self.eDOS-self.dosEfermi, 
                    y1=self.dosup,
                    y2=-self.dosdn,
                    color= "grey",
                    alpha= 0.1)
            dd.set_ylim((-10, 10))
        else:
            dd.plot(self.eDOS - self.dosEfermi, self.dosup, 
                        color='red', linewidth=0.5)
            dd.set_ylim((0, 10))


        # locator = AutoMinorLocator()
        dd.yaxis.set_minor_locator(MultipleLocator(1))
        dd.yaxis.set_major_locator(MultipleLocator(2))
        dd.xaxis.set_minor_locator(MultipleLocator(1))
        dd.xaxis.set_major_locator(MultipleLocator(2))

        dd.set_ylabel('Density of states')  # Add an x-label to the axes.
        dd.set_xlabel(r'$E-E_f$ [eV]')  # Add a y-label to the axes.
        # dd.set_title("Spinpolarized DOS")
        
        dd.vlines(0, ymin=-30, ymax=30*1.2, colors='black', ls='--', alpha= 1.0, linewidth=1.0)
        dd.hlines(0, xmin=-30, xmax=30*1.2, colors='black', ls='--', alpha= 1.0, linewidth=1.0)
        
        width = 7
        fig.set_figwidth(width)     #  ширина и
        fig.set_figheight(width/1.6)    #  высота "Figure"
        dd.set_xlim((-15, 15))
        
        if saveQ:
            plt.savefig('./'+ picname, dpi=200, bbox_inches='tight')
        plt.show()


    def plot_pDOS(self, element='Fe'):
        '''
        plots pDOS of element         
        '''
        fig, dd = plt.subplots()

        def plot_dos(spin_label, sign, spin_key):


            orbitals = list(self.dos.get_element_spd_dos(element).keys())

            colors = [ 'c',  'red', 'blue']
            atom_tdos = self.dos.densities[spin_key]
            
            for orbital, color in dict(zip(orbitals, colors)).items():
                atom_pdos = self.dos.get_element_spd_dos(element)[orbital].densities[spin_key]
                ens = self.dos.get_element_spd_dos(element)[orbital].energies - self.dosEfermi
                dd.plot(ens, sign * atom_pdos, label=f"{orbital} DOS", color=color, linewidth=0.5)
            
            dd.plot(ens, sign * atom_tdos, color='green', 
                    label=f'TDOS {element}', linewidth=0.8, linestyle='dashed')
            plt.fill_between(x=ens, y1=sign * atom_tdos, color="grey", alpha=0.1)



        if self.FMq:
            # Plot UP spin
            plot_dos('UP', sign=1, spin_key=self.spin_keys[0])

            # Plot DOWN spin
            plot_dos('DOWN', sign=-1, spin_key=self.spin_keys[1])
            dd.set_ylim((-15, 10))
        else:
            plot_dos('PM', sign=1, spin_key=self.spin_keys[0])
            dd.set_ylim((0, 10))

        locator = AutoMinorLocator()
        dd.yaxis.set_minor_locator(locator)
        dd.xaxis.set_minor_locator(locator)

        dd.set_ylabel('Density of states')  
        dd.set_xlabel(r'$E-E_f$ [eV]')
        dd.set_title(element +" pDOS")
        dd.legend() 
        
        dd.vlines(0, ymin=0, ymax=30*1.2, colors='black', ls='--', alpha= 1.0, linewidth=1.0)
        width = 7
        fig.set_figwidth(width)     
        fig.set_figheight(width/1.6)
        dd.set_xlim((-5, 5))
        # plt.savefig('./2pub/pics/pDOS.png', dpi=200, bbox_inches='tight')

        plt.show()



    def get_qe_kpathBS(self, printQ=False):
        '''
        makes kpath for wannier plotting and saves it
        '''
        N_points_direction = 10
        
        NHSP = len(self.HighSymPointsCoords)
        with open(self.directory + "kpaths/kpath_qe2.dat", "w") as fout2:

            Letter_prev = self.HighSymPointsNames[0]
            dist = 0.0
            k_prev = self.HighSymPointsCoords[0]
            print(Letter_prev)

            for HSP_ind in range(1, NHSP):
                
                Letter_new = self.HighSymPointsNames[HSP_ind]
                k_new = self.HighSymPointsCoords[HSP_ind]
                
                delta_k = k_new - k_prev
                
                num_points = 20 
                for point in range(num_points + (HSP_ind==NHSP-1)):
                    k_to_write = k_prev +   delta_k/(num_points)*point 
                    # print(k_to_write)
                    if point == 0:
                        Letter_to_write =  Letter_prev
                    elif (HSP_ind == NHSP-1 and point == num_points):
                        Letter_to_write =  Letter_new
                    else:
                        Letter_to_write = '.'
                    fout2.write( 
                        f'{Letter_to_write} {k_to_write[0]:.8f}  {k_to_write[1]:.8f} {k_to_write[2]:.8f}  \t {dist:.8f} \n'
                    )


                    k_to_write =     np.array(list(map(int,   k_to_write*N_points_direction)))  
                    dist += LA.norm(self.bcell.T@delta_k/(num_points))
                
                print(Letter_new)
                k_prev = k_new[:]
                Letter_prev = Letter_new 
                        
    # Wannier90 interface 
    def load_wannier(self, wannier_hr_filename):
        '''
        loads wannier kpath and calculates wannier BS along it
        '''
        if self.FMq:
            self.wannier = wannier_loader.Wannier_loader_FM(self.directory, wannier_hr_filename)
            self.wannier.load_kpath('./kpaths/kpath_qe2.dat')
            self.BS_wannier_dn = self.wannier.get_wannier_BS(spin=0)
            self.BS_wannier_up = self.wannier.get_wannier_BS(spin=1)
            self.nwa = self.BS_wannier_dn.shape[1]
        else:
            self.wannier = wannier_loader.Wannier_loader_PM(self.directory, wannier_hr_filename)
            self.wannier.load_kpath('./kpaths/kpath_qe2.dat')
            self.BS_wannier = self.wannier.get_wannier_BS()
            self.nwa = self.BS_wannier.shape[1]


    def plot_wannier_BS(self):
        '''
        plots wannier BS in comparison to qe BS
        '''
        
        label_ticks = self.HighSymPointsNames
        normal_ticks = self.HighSymPointsDists

        fig, dd = plt.subplots() 

        def plot_bands(kpath_dists, bandsDFT, label,  color,  alpha, linewidth=0.7):
            for band in range(self.nbandsDFT):
                label = label if band == 0 else ""
                dd.plot(kpath_dists, bandsDFT[band] - self.efermi, 
                        label=label, color=color, linewidth=linewidth, alpha=alpha)
                
        def plot_wannier(kpath_dists, BS_wannier, label,  color,  alpha, linewidth=3):
            for band in range(self.nwa):
                label = label if band == 0 else ""
                dd.plot(kpath_dists, BS_wannier[:, band] - self.efermi, 
                        label=label, color=color, alpha=alpha, linewidth=linewidth)
                
        if self.FMq:

            plot_bands(self.bs.distance, self.bs.bands[self.spin_keys[0]], 'up',  'red',  1.0)
            plot_bands(self.bs.distance, self.bs.bands[self.spin_keys[1]], 'dn',  'blue', 1.0)

            
            plot_wannier(self.wannier.kpath_dists_qe, self.BS_wannier_up, 'up wannier',  'red',  0.3, 4)
            plot_wannier(self.wannier.kpath_dists_qe, self.BS_wannier_dn, 'dn wannier',  'blue', 0.3, 4)

        else:
            plot_bands(self.bs.distance, self.bs.bands[self.spin_keys[0]], 'DFT',  'red',  1.0)
            plot_wannier(self.wannier.kpath_dists_qe, self.BS_wannier, 'wannier',  'blue',  0.3, 2)


        dd.set_ylabel(r'E - $E_f$ [Ev]') 
        dd.legend(prop={'size': 8}, loc='upper right', frameon=False)  
        plt.xticks(normal_ticks, label_ticks)
        dd.yaxis.set_minor_locator(MultipleLocator(1))
        plt.grid(axis='x')
        dd.axhline(y=0, ls='--', color='k')
        plt.xlim(normal_ticks[0], normal_ticks[-1])
        plt.ylim(-15, 15)

        width = 7
        fig.set_figwidth(width)     
        fig.set_figheight(width/1.6)
        # plt.savefig('./2pub/pics/BS_wannier.png', dpi=200, bbox_inches='tight')

        plt.show()
