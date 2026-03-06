# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:30:08 2026

@author: tomas
"""

import imageio.v2 as imageio
import matplotlib.pyplot as plt 
from pathlib import Path
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import os
from scipy.signal import find_peaks
from scipy.ndimage import rotate
import scipy.stats as stats
from scipy.optimize import curve_fit
from utils import *

# %%


ROOT = Path(r"C:\Users\tomas\Desktop\Laboratorio-4-cobelli-main\Clase 8\young _2\rendija")
extensiones_validas = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
archivos = [f for f in os.listdir(ROOT) if f.lower().endswith(extensiones_validas)]
# archivos.sort()  # opcional, para orden alfabético
    
images = []
for archivo in archivos:
    ruta_completa = ROOT / archivo
    try:
        imagen = imageio.imread(ruta_completa)
        images.append(imagen)
    except Exception as e:
        print(f"Error al leer {archivo}: {e}")

if not images:
    print("No se encontraron imágenes.")
    exit()
    
imagenes =[images[2],images[4],images[6],images[8],images[10],images[12],images[14],images[16],images[18]]


# %%
#checkeo donde tomar
imagen = images[0]
# plt.figure()
# plt.imshow(imagen)
# plt.show()

angulo_correccion = 1.60
imagen_rotada = rotate(imagen, angulo_correccion, reshape=False, order=3)
plt.figure()
plt.imshow(imagen_rotada)
plt.show()


#Es bastante recta la imagen, no es necesario hacer lo que hicimos antes de ver el error sistematico del angulo de la camara

center_y = 625
center_x = 2455
offset  = 155

matriz = imagen_rotada[center_y - offset : center_y + offset,
                center_x - offset : center_x + offset,
                2].astype(float)
plt.figure()
plt.imshow(matriz)
plt.show()

center_y2 = 1500
center_x2 = 1000
offset  = 155


matriz2 = imagen_rotada[center_y2 - offset : center_y2 + offset,
                center_x2 - offset : center_x2 + offset,
                2].astype(float)

plt.figure()
plt.imshow(matriz2)
plt.show()
# %%
row = 100   #tome esta fila

perfil_matriz = matriz[row, :]


peaks, properties = find_peaks(
    -perfil_matriz,
    prominence = 20,  # <-- AJUSTA ESTE VALOR (ej. 5, 10 o 20) según la altura de tus valles
    distance = 7,      # <-- Evita que tome picos muy pegados entre sí
    width       = None                              
)


if len(peaks) >= 2:
    pasos = np.diff(peaks.astype(float))          # diferencias en píxeles (float para precisión)
    paso_promedio = np.mean(pasos)
    paso_std      = np.std(pasos)
    paso_min      = np.min(pasos)
    paso_max      = np.max(pasos)
    
    print(f"✅ Paso promedio entre picos consecutivos: {paso_promedio:.2f} ± {paso_std:.2f} píxeles")
    print(f"   Mínimo paso: {paso_min:.1f} px  |  Máximo paso: {paso_max:.1f} px")
    
    centro_recorte = row                     
    idx_central = np.argmin(np.abs(peaks - centro_recorte))
    pico_central = peaks[idx_central]
    
    print(f"   Pico central (m=0) en fila: {pico_central} (distancia al centro del recorte: {abs(pico_central - centro_recorte)} px)")
    
    # Asignar órdenes m (izquierda = negativos, derecha = positivos)
    ordenes = np.arange(-idx_central, len(peaks) - idx_central)
    # ====================== GRÁFICO DEL PASO (linealidad del patrón) ======================
    plt.figure(figsize=(12, 6))
    
    # Posición vs Orden (debe ser una recta perfecta en difracción ideal)
    plt.subplot(1, 2, 1)
    plt.plot(ordenes, peaks, 'o-', color='tab:red', markersize=8, linewidth=2.5, label='Posiciones medidas')
    
    # Ajuste lineal (pendiente = paso promedio)
    coef,cov = np.polyfit(ordenes, peaks, 1 ,cov=True)
    paso_fit = coef[0]
    error_coef = np.sqrt(np.diag(cov))
    plt.plot(ordenes, coef[0]*ordenes + coef[1], '--', color='black', label=f'Ajuste lineal\npaso = {paso_fit:.2f} +- {error_coef[0]:.2f} px/mm ')
    
    plt.xlabel('milimetros de paso')
    plt.ylabel('Posición en la imagen (píxeles)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(matriz[row, :],".-" , label='Original', linewidth=1.5, alpha = 0.1)
    plt.plot(perfil_matriz, color='tab:blue', linewidth=2.5, label='Perfil filtrado')
    plt.plot(peaks, perfil_matriz[peaks], "x", markersize=12, color='red', label='Picos')
    plt.show()
else:
    print("no se encontraron picos")

# paso_fit = pixeles / mm


dist_real = 1  # mm (CAMBIAR por referencia real)


escala=16.04
err_escala=0.02
# %%
row2 = 150   #tome esta fila

perfil_matriz2 = matriz2[row2, :]


peaks, properties = find_peaks(
    -perfil_matriz2,
    prominence = 20,  # <-- AJUSTA ESTE VALOR (ej. 5, 10 o 20) según la altura de tus valles
    distance = 7,      # <-- Evita que tome picos muy pegados entre sí
    width       = None                              
)


if len(peaks) >= 2:
    pasos = np.diff(peaks.astype(float))          # diferencias en píxeles (float para precisión)
    paso_promedio = np.mean(pasos)
    paso_std      = np.std(pasos)
    paso_min      = np.min(pasos)
    paso_max      = np.max(pasos)
    
    print(f"✅ Paso promedio entre picos consecutivos: {paso_promedio:.2f} ± {paso_std:.2f} píxeles")
    print(f"   Mínimo paso: {paso_min:.1f} px  |  Máximo paso: {paso_max:.1f} px")
    
    centro_recorte = row                     
    idx_central = np.argmin(np.abs(peaks - centro_recorte))
    pico_central = peaks[idx_central]
    
    print(f"   Pico central (m=0) en fila: {pico_central} (distancia al centro del recorte: {abs(pico_central - centro_recorte)} px)")
    
    # Asignar órdenes m (izquierda = negativos, derecha = positivos)
    ordenes = np.arange(-idx_central, len(peaks) - idx_central)
    # ====================== GRÁFICO DEL PASO (linealidad del patrón) ======================
    plt.figure(figsize=(12, 6))
    
    # Posición vs Orden (debe ser una recta perfecta en difracción ideal)
    plt.subplot(1, 2, 1)
    plt.plot(ordenes, peaks, 'o-', color='tab:red', markersize=8, linewidth=2.5, label='Posiciones medidas')
    
    # Ajuste lineal (pendiente = paso promedio)
    coef,cov = np.polyfit(ordenes, peaks, 1 ,cov=True)
    paso_fit = coef[0]
    error_coef = np.sqrt(np.diag(cov))
    plt.plot(ordenes, coef[0]*ordenes + coef[1], '--', color='black', label=f'Ajuste lineal\npaso = {paso_fit:.2f} +- {error_coef[0]:.2f} px/mm ')
    
    plt.xlabel('milimetros de paso')
    plt.ylabel('Posición en la imagen (píxeles)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(matriz2[row2, :],".-" , label='Original', linewidth=1.5, alpha = 0.1)
    plt.plot(perfil_matriz2, color='tab:blue', linewidth=2.5, label='Perfil filtrado')
    plt.plot(peaks, perfil_matriz2[peaks], "x", markersize=12, color='red', label='Picos')
    plt.show()
else:
    print("no se encontraron picos")

# paso_fit = pixeles / mm


dist_real = 1  # mm (CAMBIAR por referencia real)


escala2=15.93
err_escala2=0.02

# %%
D_m = 0.763
err_D_m = 0.002  
D_um=763000
err_D_um = 2000

# --- Calibración de Escala ---
escala_1 = 16.04
err_escala_1 = 0.02
escala_2 = 15.93
# El error sistemático es la diferencia entre ambas mediciones
err_sist = abs(escala_1 - escala_2) 

# Error total en px/mm (Combinación lineal de errores)
err_px_mm_total = err_escala_1 + err_sist # 0.13 px/mm

# --- Conversión a Metros (Propagación Correcta) ---
# Tamaño del píxel en metros (dx)
pixel_size_m = (1 / escala_1) * 1e-3 

# Propagación: El error relativo de la escala es el mismo que el del píxel
# rel_err = Delta(escala) / escala
rel_err_cal = err_px_mm_total / escala_1 

# Error absoluto del tamaño del píxel en metros (Delta dx)
err_pixel_size_m = pixel_size_m * rel_err_cal 

print(f"Píxel: {pixel_size_m:.4e} m")
print(f"Error Píxel: {err_pixel_size_m:.4e} m") # Debe dar ~5e-7
# %%

imagenes_recortadas= imagenes[1:-1]  #la primera y la ultima no estan lindas y las voy a rotar un poco


angulo_correccion = 91
imagenes_recortadas_rotadas=[]
for i in range(len(imagenes_recortadas)):
    imagen_recortado_rotada = rotate(imagenes_recortadas[i], angulo_correccion, reshape=False, order=3)
    imagenes_recortadas_rotadas.append(imagen_recortado_rotada) 
    
# plt.figure()
# plt.imshow(imagenes_recortadas_rotadas[0])
# plt.show()

matrices_recortadas = []
matrices = []

center_x=1450 
center_y=1030
offset=100
    # Usamos len() para obtener el número de imágenes
for i in range(len(imagenes_recortadas_rotadas)):
    imagen_centrada = preparar_roi(imagenes_recortadas_rotadas[i],center_x=1030, center_y=1450, offset=300, canal=2)
    matrices_recortadas.append(imagen_centrada["matriz"])         
    #plt.figure()
    #plt.imshow(imagen_centrada["matriz"])
    #plt.show()

kes = []
err_kes = []

for i in range(len(imagenes_recortadas_rotadas)):
    k_promedio, err_k, _, _ = calcular_k_desde_espectro_adaptativo(
        imagen=imagenes_recortadas_rotadas[i],
        center_x=890, center_y=1645, offset=650,
        canal=2,
        plot_espectro=False,   # puedes poner True para depurar
        umbral_lobulo=0.8,
        delta_kx=5000,           # ajusta según la extensión del lóbulo
        delta_ky=2000,          # ajusta según la extensión del lóbulo
        k_y_min_inicial=1500,
        pixel_size_m=pixel_size_m,
        err_pixel_size_m=err_pixel_size_m   
    )
    if k_promedio is not None:
        kes.append(k_promedio)
        err_kes.append(err_k)
    else:
        kes.append(np.nan)
        err_kes.append(np.nan)

kes=np.array(kes)
err_kes=np.array(err_kes)
print(kes)
print(err_kes)
#%%
# Datos (ejemplo)
delta_y = 2 * np.pi / kes
err_delta_y = (2 * np.pi / k_promedio**2) * err_kes

# Datos (ejemplo)
#rendija = np.array([45, 50, 55, 60, 65, 70, 75]) * 1e-6  # micrometros a metros si es que era micrometros

rendija = np.array([0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]) * 1e-3  # mm a metros si es que era mm

err_rendija = 0.02 * 1e-3 #mm a m


relativo_delta_y=err_delta_y/delta_y
relativo_rendija=err_rendija/rendija
relativo_kaes=err_kes/kes

print(relativo_rendija)
print(relativo_kaes)
print(relativo_delta_y)
# %%

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit

def f(x, a, b):
    return a * x + b


# ajusto:
init_guess = [0,0] # esto es importantísimo para ajustes no lineales!
popt, pcov = curve_fit(f, kes, rendija, sigma=err_rendija, absolute_sigma=True) # popt son los parámetros del ajuste, pcov la "matriz de covarianza"
perr = np.sqrt(np.diag(pcov)) # los errores de los parámetros del ajuste son la raíz cuadrada de la diagonal de la matriz de covarianza

print('Resultados del ajuste:')
for i in range(len(popt)):
  print('Parámetro ' + str(i) + ': ' + str(popt[i]) + " \u00B1 " + str(perr[i]))
  
  

x_ajuste = np.linspace(np.min(kes),np.max(kes),len(kes)*10) # defino un eje horizontal más fino que los puntos que medí, para que el ajuste se vea suave

plt.figure()
plt.title('Datos ajustados')
plt.xlabel('X')
plt.ylabel('Y')
plt.errorbar(kes, rendija, err_rendija, 0, '.')
plt.plot(x_ajuste,f(x_ajuste,popt[0],popt[1]))
plt.grid(True)
plt.show()


lambdaa_m = popt[0] * (1.43 / (2 * np.pi)) * D_m
lambdaa_nm = lambdaa_m * 1e9
print(lambdaa_nm)
