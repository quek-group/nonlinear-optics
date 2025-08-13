#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 22:06:14 2021

@author: laimingrui
"""
import numpy as np
import numba as nb
import sys

@nb.njit(fastmath = True)
def nb_sum(zero, ssp, e, W, eta, tol):
    out = np.zeros((len(W)), dtype = zero.dtype)
    for w in nb.prange(len(W)):
        for sp in range(ns):
            for s in range(ns):
                out[w] += zero[s,x]*ssp[s,sp,y]*np.conj(zero[sp,y])*eta/(((e[sp]-W[w])**2+eta**2)*e[s]*(W[w]**2))
                if s != sp:
                    out[w] -= zero[s,y]*ssp[s,sp,x]*np.conj(zero[sp,y])*eta*(e[sp]-e[s])/(((e[sp]-W[w])**2+eta**2)*((e[sp]-e[s])**2+eta**2)*(W[w]**2))
    return out



ns = 6000
zero_p_S = np.load('zero_p_S.npy')
S_p_Sp = np.load('S_p_Sp_sym.npy')
eigenvalues = np.load('eigenvalues.npy')

a = np.array([3.926124439, 0, 0,  
   0, 6.750729329, 0,
   0, 0, 23.913364410])
 
a = a.reshape(3,3)
hbar = 1.054571817e-34 # J s
hbar_ev = 6.582119569e-16 # eV s
m_e = 9.10938356e-31 # kg
elec = -1.60217662e-19 # C
alat = a[0,0]*1e-10
kx = 24
ky = 12
kz = 1
nw = 351


x,y = 2,2
omega = np.linspace(1.5,5,nw)
smear = 0.1
vg = nb_sum(zero_p_S, S_p_Sp, eigenvalues, omega, smear, tol) 
vg *= (2*np.pi*hbar/alat)**3
vg /= m_e**3
vg *= 2 * elec**3 /hbar**2
vg *= hbar_ev**4
vg /= (np.dot(a[0], np.cross(a[1], a[2]))*1e-30)  * (kx*ky*kz)
vg *= 1j
np.save('shift_current_exciton_'+str(smear)+"_"+str(ns), vg)

