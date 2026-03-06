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
err_escala_sist=escala-escala2
err_escala_total=err_escala+err_escala_sist
#uso el err err_escala_total y lo que defini como escala

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
espectros_log = []

center_x = 1030 
center_y = 1450
offset = 300

# 1. Procesamiento en bucle
for i in range(len(imagenes_recortadas_rotadas)):
    # Extraer ROI
    resultado_roi = preparar_roi(imagenes_recortadas_rotadas[i], 
                                 center_x=center_x, 
                                 center_y=center_y, 
                                 offset=offset, 
                                 canal=2)
    
    matriz = resultado_roi["matriz"].astype(float)
    matrices_recortadas.append(matriz)
    
    # 2. Calcular FFT (Detrending para eliminar el pico de DC / A_0)
    matriz_detrend = matriz - np.mean(matriz)
    fshift = np.fft.fftshift(np.fft.fft2(matriz_detrend))
    
    # 3. Magnitud en escala logarítmica
    espectro_abs = np.abs(fshift)
    espectro_log = np.log(1 + espectro_abs)
    espectros_log.append(espectro_log)
    
    # 4. Plotear individualmente
    plt.figure(figsize=(10, 4))
    
    # Imagen original (ROI)
    plt.subplot(1, 2, 1)
    plt.title(f"ROI Imagen {i}")
    plt.imshow(matriz)
    
    # Espectro de Fourier
    plt.subplot(1, 2, 2)
    plt.title(f"Espectro Log {i}")
    plt.imshow(espectro_log, cmap='viridis')
    plt.colorbar(label='log10(1+|F|)')
    
    plt.tight_layout()
    plt.show()

# %%
px_por_mm = 16.04
err_px_por_mm = err_escala_total
rel_err_cal = err_px_por_mm / px_por_mm   
pixel_size_um = 1000 / px_por_mm           
pixel_size_m  = pixel_size_um * 1e-6       
D_m = 0.763
err_D_m = 0.002  
D_um=763000
err_D_um = 2000



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
        pixel_size_m=pixel_size_m
    )
    if k_promedio is not None:
        kes.append(k_promedio)
        err_kes.append(err_k)
    else:
        kes.append(np.nan)
        err_kes.append(np.nan)


print(kes)
print(err_kes)
# %%




# %%
# Datos (ejemplo)
rendija = np.array([45, 50, 55, 60, 65, 70, 75]) * 1e-6  # micrometros a metros si es que era micrometros
rendija = np.array([45, 50, 55, 60, 65, 70, 75]) * 1e-3  # mm a metros si es que era mm

lamba1 =  2*np.pi * rendija[0] / ( kes[0]*D_m)
print("asdasd " , lamba1)


plt.errorbar(rendija, kes, yerr=err_kes, fmt=".")
plt.show()

coef, cov_lin = np.polyfit(
    rendija,
    kes,
    1,
    #w=1/np.array(err_kes),
    cov=False
)

print(coef)

pendiente = coef
intercepto = coef[1]

err_pend = np.sqrt(cov_lin[0,0])
err_int = np.sqrt(cov_lin[1,1])


lamba = 2*np.pi / (pendiente * D_m)

print(lamba)
#modelo_lin = pendiente*rendija + intercepto
#residuos_lin = rendijas - modelo_lin

#chi2_lin = np.sum(((rendijas - modelo_lin)/err_rendijas)**2)
#gl_lin = len(rendijas) - 2
#chi2_red_lin = chi2_lin / gl_lin
