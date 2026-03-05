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
# plt.figure()
# plt.imshow(imagen_rotada)
# plt.show()




center_x = 625
center_y = 2455
offset  = 155

matriz = imagen_rotada[center_x - offset : center_x + offset,
                center_y - offset : center_y + offset,
                2].astype(float)

# plt.figure()
# plt.imshow(matriz)
# plt.show()

# %%

#=========================================================================

# # Promedio por filas (promedia cada fila → queda un perfil en eje vertical)
# perfil_filas = np.mean(espectro_log, axis=1)

# # Promedio por columnas (promedia cada columna → perfil en eje horizontal)
# perfil_columnas = np.mean(espectro_log, axis=0)

# # Ejes (índices de píxeles en la FFT)
# y = np.arange(len(perfil_filas))
# x = np.arange(len(perfil_columnas)) 

#############################################################################################

#Calibracion de distancia pixel

#  1. FFT 2D 
f = np.fft.fft2(matriz)
fshift = np.fft.fftshift(f)
fshift_abs = np.abs(fshift)
espectro_log = np.log10(1 + fshift_abs)        
#  2. FILTRO PASA-BAJO (elimina frecuencias altas) 

rows, cols = matriz.shape
crow, ccol = rows // 2, cols // 2


radio = 90          
# radio pequeño (10-40)  → elimina muchas altas frecuencias (suavizado fuerte)
# radio grande (80-150)  → elimina solo las muy altas (suavizado suave, conserva más detalle)
# Máscara circular: True = mantener bajas frecuencias (centro)


y, x = np.ogrid[:rows, :cols]
distancia = np.sqrt((y - crow)**2 + (x - ccol)**2)
mascara = distancia <= radio                     # ←←← PASA-BAJO
# Aplicar filtro

fshift_filtrado = fshift * mascara
#  3. INVERSA FFT 

f_ishift = np.fft.ifftshift(fshift_filtrado)
imagen_filtrada = np.fft.ifft2(f_ishift)
imagen_filtrada = np.real(imagen_filtrada)       # usamos real() porque la parte imaginaria es ~0
row = 100   #tome esta fila
plot_fft = False 
if plot_fft:
    #  VISUALIZACIÓN 
    plt.figure(figsize=(16, 12))

    plt.subplot(2, 3, 1)
    plt.imshow(matriz, cmap='gray', vmin=matriz.min(), vmax=matriz.max())
    plt.title('Original (canal 2)')
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.imshow(espectro_log, cmap='gray')
    plt.title('Espectro de Fourier (log)')
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.imshow(mascara, cmap='gray')
    plt.title(f'Máscara PASA-BAJO\n(Radio = {radio} píxeles)')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.imshow(imagen_filtrada, cmap='gray')
    plt.title('IMAGEN FILTRADA\n(frecuencias altas eliminadas)')
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.imshow(matriz - imagen_filtrada, cmap='gray')
    plt.title('Diferencia (solo altas frecuencias removidas)')
    plt.colorbar()

    plt.subplot(2, 3, 6)
    plt.imshow(np.log10(1 + np.abs(fshift - fshift_filtrado)), cmap='gray')
    plt.title('Espectro removido (solo altas frecuencias)')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    #  PERFIL DE COMPARACIÓN 

    #bineado = np.mean()                  
    plt.figure(figsize=(12, 5))
    plt.plot(matriz[row, :],".-" , label='Original', linewidth=1.5, alpha = 0.1)
    plt.plot(imagen_filtrada[row, :],".-" , label=f'Filtrada pasa-bajo (radio={radio})', linewidth=2)
    plt.title('Perfil vertical central - Original vs Filtrada')
    plt.xlabel('Fila')
    plt.ylabel('Intensidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


perfil_fila = imagen_filtrada[row, :] 
#perfil_fila = np.mean(imagen_filtrada, axis=0) #no usar
#=========================================================================
#  FIND_PEAKS (versión mejorada) ======================
peaks, properties = find_peaks(
    -imagen_filtrada[row, :] ,
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
    
    centro_recorte = offset                     
    idx_central = np.argmin(np.abs(peaks - centro_recorte))
    pico_central = peaks[idx_central]
    
    print(f"   Pico central (m=0) en fila: {pico_central} (distancia al centro del recorte: {abs(pico_central - centro_recorte)} px)")
    
    # Asignar órdenes m (izquierda = negativos, derecha = positivos)
    ordenes = np.arange(-idx_central, len(peaks) - idx_central)
    # # ====================== GRÁFICO DEL PASO (linealidad del patrón) ======================
    # plt.figure(figsize=(12, 6))
    
    # # Posición vs Orden (debe ser una recta perfecta en difracción ideal)
    # plt.subplot(1, 2, 1)
    # plt.plot(ordenes, peaks, 'o-', color='tab:red', markersize=8, linewidth=2.5, label='Posiciones medidas')
    
    # # Ajuste lineal (pendiente = paso promedio)
    # coef,cov = np.polyfit(ordenes, peaks, 1 ,cov=True)
    # paso_fit = coef[0]
    # error_coef = np.sqrt(np.diag(cov))
    # plt.plot(ordenes, coef[0]*ordenes + coef[1], '--', color='black', label=f'Ajuste lineal\npaso = {paso_fit:.2f} +- {error_coef[0]:.2f} px/mm ')
    
    # plt.xlabel('milimetros de paso')
    # plt.ylabel('Posición en la imagen (píxeles)')
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    
    # plt.subplot(1, 2, 2)
    # plt.plot(matriz[row, :],".-" , label='Original', linewidth=1.5, alpha = 0.1)
    # plt.plot(perfil_fila, color='tab:blue', linewidth=2.5, label='Perfil filtrado')
    # plt.plot(peaks, perfil_fila[peaks], "x", markersize=12, color='red', label='Picos')
    # plt.show()
else:
    print("no se encontraron picos")

# paso_fit = pixeles / mm



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
    imagen_centrada = preparar_roi(imagenes_recortadas_rotadas[i],center_x=1030, center_y=1450, offset=490, canal=2)
    matrices_recortadas.append(imagen_centrada["matriz"])         
    # plt.figure()
    # plt.imshow(imagen_centrada["matriz"])
    # plt.show()


# %%
px_por_mm = 16.03
err_px_por_mm = 0.01
rel_err_cal = err_px_por_mm / px_por_mm   
pixel_size_um = 1000 / px_por_mm           
pixel_size_m  = pixel_size_um * 1e-6       
D_m = 0.763
err_D_m = 0.002  
D_um=763000
err_D_um = 2000

# for i in range(len(matrices_recortadas)):
    
#     Ny, Nx = matrices_recortadas[i].shape
#     dx = pixel_size_m

#     kx = np.fft.fftshift(np.fft.fftfreq(Nx, dx)) * 2 * np.pi  #freq nyqist?
#     ky = np.fft.fftshift(np.fft.fftfreq(Ny, dx)) * 2 * np.pi
    
#     fshift = np.fft.fftshift(np.fft.fft2(matrices_recortadas[i]))
#     espectro_log_abs = np.log(1+np.abs(fshift))
#     plt.figure()
#     plt.imshow(espectro_log_abs)
#     plt.show()



def calcular_k_desde_espectro(imagen, center_x=890, center_y=1645, offset=650,
                                  canal=2, plot_espectro=False,
                                  k_min=1500, k_max=3000,
                                  umbral_lobulo=0.6,
                                  ancho_ventana_kx=4000):

    matriz = imagen[center_x - offset:center_x + offset,
                    center_y - offset:center_y + offset,
                    canal].astype(float)

    matriz_detrend = matriz - np.mean(matriz)

    fshift = np.fft.fftshift(np.fft.fft2(matriz_detrend))
    espectro_abs = np.abs(fshift)

    Ny, Nx = matriz.shape
    dx = pixel_size_m

    kx = np.fft.fftshift(np.fft.fftfreq(Nx, dx)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(Ny, dx)) * 2 * np.pi

    idx_kx_ventana = np.where(np.abs(kx) < ancho_ventana_kx / 2)[0]
    if len(idx_kx_ventana) == 0:
        return None, None

    sub_espectro = espectro_abs[0:, idx_kx_ventana]

    idx_ky_pos = np.where((ky > k_min) & (ky < k_max))[0]
    if len(idx_ky_pos) == 0:
        return None, None

    sub_pos = sub_espectro[idx_ky_pos, :]
    amp_max = np.max(sub_pos)

    idx_ky_lobulo, idx_kx_lobulo = np.where(sub_pos > umbral_lobulo * amp_max)
    if len(idx_ky_lobulo) < 3:
        return None, None

    k_lobulo = ky[idx_ky_pos[idx_ky_lobulo]]
    amps_lobulo = sub_pos[idx_ky_lobulo, idx_kx_lobulo]

    # ======================
    # PROMEDIO PONDERADO 
    # ======================
    w = amps_lobulo
    k_promedio = np.sum(w * k_lobulo) / np.sum(w)
    var_kbar = np.sum(w * (k_lobulo - k_promedio)**2) / (np.sum(w)**2)
    err_k = np.sqrt(var_kbar)

    # Plot opcional del espectro calibrado con lóbulo marcado
    if plot_espectro:
        _, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        # Espectro 2D
        axs[0].imshow(np.log10(1 + espectro_abs),
                      extent=[kx[0], kx[-1], ky[0], ky[-1]],
                      origin='lower',
                      cmap='viridis',
                      aspect='auto')
        axs[0].set_xlabel('kₓ  (rad/m)')
        axs[0].set_ylabel('kᵧ  (rad/m)')
        axs[0].set_title(f'Espectro de Fourier 2D calibrado\nk_promedio ≈ {k_promedio:.0f} ± {err_k:.0f} rad/m')
        axs[0].plot(0, k_promedio, 'ro', ms=8, label=f'k_promedio = {k_promedio:.0f} rad/m')
        axs[0].legend()
        
        # Visualización de la ventana 2D utilizada (sub_pos: parte positiva usada para lóbulo)
        axs[1].imshow(np.log10(1 + sub_pos),
                      extent=[kx[idx_kx_ventana[0]], kx[idx_kx_ventana[-1]], ky[idx_ky_pos[0]], ky[idx_ky_pos[-1]]],
                      origin='lower',
                      cmap='viridis',
                      aspect='auto')
        axs[1].set_xlabel('kₓ en ventana (rad/m)')
        axs[1].set_ylabel('kᵧ positivo (rad/m)')
        axs[1].set_title(f'Ventana 2D utilizada para cálculo\n(ancho k_x = {ancho_ventana_kx:.0f} rad/m, k_y > {k_min:.0f})')
        axs[1].axhline(k_promedio, color='red', ls='--', label=f'k_promedio = {k_promedio:.0f} rad/m')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        
        plt.tight_layout()
        plt.show()

    return k_promedio, err_k

kes = []
err_kes = []

for i in range(len(imagenes_recortadas_rotadas)):

    k_proomedio, err_k_promedio = calcular_k_desde_espectro(imagenes_recortadas_rotadas[i],center_x=1030, center_y=1450, offset=490,
                                  canal=2, plot_espectro=True,
                                  k_min=1500, k_max=5500,
                                  umbral_lobulo= 0.8,
                                  ancho_ventana_kx=5000)
    kes.append(k_proomedio)
    err_kes.append(err_k_promedio)



print(kes)
print(err_kes)
# %%
#Ecuacion
kes= np.array(kes)
#rendija = np.array([45, 50, 55, 60, 65, 70, 75]) #supongo um #en recortadas saque la primera y segunda
rendija = np.array([45, 50, 55, 60, 65, 70, 75]) * 1e-3  #en m, porque k lo tnego en m

def f(rendija, lambdaa, b):
   return (2*np.pi/D_m)*(1/lambdaa)*rendija + b



p_inicial = [600e-9, 0]
popt, pcov = curve_fit(f, rendija, kes, sigma=err_kes, absolute_sigma=True) 
perr = np.sqrt(np.diag(pcov)) # los errores de los parámetros del ajuste son la raíz cuadrada de la diagonal de la matriz de covarianza

print('Resultados del ajuste:')
for i in range(len(popt)):
  print('Parámetro ' + str(i) + ': ' + str(popt[i]) + " \u00B1 " + str(perr[i]))


x_ajuste = np.linspace(np.min(rendija),np.max(rendija),len(rendija)*10) # defino un eje horizontal más fino que los puntos que medí, para que el ajuste se vea suave

plt.figure()
plt.title('Datos ajustados')
plt.xlabel('rendijas')
plt.ylabel('kes')
plt.errorbar(rendija, kes, err_kes, 0, '.')
plt.plot(x_ajuste,f(x_ajuste,popt[0],popt[1]))
plt.grid(True)
plt.show()
     

# Recursos necesarios para calcular el chi^2 y su p-valor:
puntos = len(rendija)
params = len(popt)
grados_libertad = puntos - params
y_modelo = f(rendija,popt[0],popt[1])

# calculo el chi^2 y su p-valor:
chi_cuadrado = np.sum(((kes-y_modelo)/err_kes)**2)
p_chi = stats.chi2.sf(chi_cuadrado, grados_libertad)
# interpretamos el resultado:
print('chi^2: ' + str(chi_cuadrado))
print('p-valor del chi^2: ' + str(p_chi))

if err_kes[0]==0:
    print('No se declararon errores en la variable y.')
elif p_chi<0.05:
    print('Se rechaza la hipótesis de que el modelo ajuste a los datos.')
else:
    print('No se puede rechazar la hipótesis de que el modelo ajuste a los datos.')
     