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
                    out[w] -= zero[s,y]*ssp[s,sp,x]*np.conj(zero[sp,y])*eta*(e[sp]-e[s])/(((e[sp]-W[w])**2+eta**2)*((e[sp]-e[s])**2+4*eta**2)*(W[w]**2))
    return out



ns = 1470
ryd = 13.605684958731
zero_p_S = np.load('CdS_zero_p_S.npy')
S_p_Sp = np.load('CdS_S_p_Sp.npy')
eigenvalues = np.load('CdS_exciton_energy.npy') * ryd

a = np.array([[4.1360001564, 0.0000000000, 0.0000000000],
       [-2.0680000782, 3.5818812055, 0.0000000000],
        [0.0000000000, 0.0000000000, 6.7100000381]])
 
a = a.reshape(3,3)
hbar = 1.054571817e-34 # J s
hbar_ev = 6.582119569e-16 # eV s
m_e = 9.10938356e-31 # kg
elec = -1.60217662e-19 # C
alat = a[0,0]*1e-10
kx = 100
ky = 100
kz = 66
nw = 81


x,y = 2,2
omega = np.linspace(2.3,2.7,nw)
smear = 0.03
vg = nb_sum(zero_p_S, S_p_Sp, eigenvalues, omega, smear, tol) 
vg *= (2*np.pi*hbar/alat)**3
vg /= m_e**3
vg *= 2 * elec**3 /hbar**2
vg *= hbar_ev**4
vg /= (np.dot(a[0], np.cross(a[1], a[2]))*1e-30)  * (kx*ky*kz)
vg *= 1j
vg *= 10e6 # change units to micro A/V^2
np.save('shift_current_exciton_'+str(smear)+"_"+str(ns), vg)

