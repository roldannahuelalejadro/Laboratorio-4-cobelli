#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 18:25:25 2026

@author: nclotta
"""

# Time-stamp: </Users/nclotta/Documents/__UBA/__LABO_4_CUATRO/informe1/graficos.py, 2026-02-06 Friday 00:19:20 nclotta>

import matplotlib.pyplot as plt
import numpy as np
#import math
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mc

datos_medicion = pd.read_csv("./data/datos_array29_13_52_55.csv")

def string_to_list(data_str):
    if isinstance(data_str, str):
        data_str = data_str.strip()
        if data_str.startswith('[') and data_str.endswith(']'):
            data_str = data_str[1:-1]
        try:
            return [float(x.strip()) for x in data_str.split(',') if x.strip()]
        except ValueError:
            print(f"Error convirtiendo: {data_str[:50]}...")
            return []
    elif isinstance(data_str, list):
        return data_str
    else:
        return []

# B,H,V_platino,V_R,Time
B = [string_to_list(x) for x in datos_medicion["B"].iloc[1:]]
H = [string_to_list(x) for x in datos_medicion["H"].iloc[1:]]
V_plat = [string_to_list(x) for x in datos_medicion["V_platino"].iloc[1:]]
V_R = [string_to_list(x) for x in datos_medicion["V_R"].iloc[1:]]
tt = [string_to_list(x) for x in datos_medicion["Time"].iloc[1:]]

k1 = 99
k2 = 499

def resistencia_platino(V_plat, V_R, res=2000):
    return np.abs(V_plat / V_R) * res

def PT100_res2temp_interp(R):
    data = np.loadtxt('Pt100_resistencia_temperatura.csv',delimiter=',') 
    temperature_vals = data[:,0]
    resistance_vals = data[:,1]
    return np.interp(R, resistance_vals, temperature_vals)

v_arr = [[],[]]
indices =[]

for i in range(k1,k2):
    v_arr[0].append((np.mean(np.array(V_plat[i+1]))))
    v_arr[1].append((np.mean(np.array(V_R[i+1]))))
    indices.append(i)

R = resistencia_platino(np.array(v_arr[0]),  np.array(v_arr[1]))
T = np.array(PT100_res2temp_interp(R))

order_mask = T.argsort()
T = T[order_mask]
data_mask = (T > -76) & (T < 22)

T_masked = T[data_mask]
order_mask = T_masked.argsort()

campos_ordenados = [[],[]]

for i,j in enumerate(indices):
    if data_mask[i]:
        campos_ordenados[0].append(H[j])
        campos_ordenados[1].append(B[j])

H_final = np.array(campos_ordenados[0], dtype=object)[order_mask]
B_final = np.array(campos_ordenados[1], dtype=object)[order_mask]

previously_graphed = -73
graph_threshold = 2.5

_, ax = plt.subplots()

#im = plt.imshow([[0,0],[0,0]],cmap='plasma_r')
plt.grid()
graphed_temps = np.array([])
for i in range(0,len(H_final)):
    if T[i] > -76 and T[i] < 22:
        if np.abs(previously_graphed - T[i]) > graph_threshold:
            previously_graphed = T[i]
            graphed_temps = np.append(graphed_temps, T[i])
            plt.scatter(H_final[i],B_final[i],color=cm.plasma_r(np.abs(T[i])/np.max(np.abs(T_masked))),
                        marker='.',zorder=5)#,label=f"T={T[i]}")

cbar = plt.colorbar(cm.ScalarMappable(cmap='plasma_r',
                                      norm=mc.Normalize(vmin=np.min(np.abs(T_masked)),
                                                        vmax=np.max(np.abs(T_masked)))),ax=ax)
temp=np.array([])
textooooo = []
for i in range(0,len(graphed_temps),3):
    temp = np.append(temp, graphed_temps[i])
    textooooo.append(f"{graphed_temps[i]:.0f}")
    
cbar.set_ticks(np.abs(temp), labels=textooooo)
cbar.set_label("    °C", rotation=0)
plt.show()


# eof
