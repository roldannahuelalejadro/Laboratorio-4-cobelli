import scipy.stats as stats
import imageio.v2 as imageio
import matplotlib.pyplot as plt 
from pathlib import Path
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import os
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from utils import *

ROOT = Path(r"C:\Users\User\Desktop\Laboratorio-4-cobelli\Clase 8\young _2\aluminium_")


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
# ====================== CONSTANTES (de tu calibración) ======================
lambda_nm = 650
err_lambda_nm = 1  # ¡DEBES DEFINIR UN ERROR REAL! (ej. 1 nm)
lambda_m = lambda_nm * 1e-9
err_lambda_m = err_lambda_nm * 1e-9

px_por_mm = 23.88
err_px_por_mm = 0.01
rel_err_cal = err_px_por_mm / px_por_mm   # ≈ 0.0004188
pixel_size_um = 1000 / px_por_mm           # ≈ 41.8936 µm/píxel
pixel_size_m = pixel_size_um * 1e-6        # ≈ 4.189e-5 m/píxel
err_pixel_size_m = pixel_size_m * rel_err_cal

D_m = 0.5125
err_D_m = 0.0001

rendijas_fft = []
err_rendijas_fft = []

# IMPORTANTE: Definir 'imagen' dentro del bucle o pasar las imágenes correctamente
# Asumo que tienes una lista 'images' con las 10 imágenes
for i in range(10):
    # Obtener la imagen correspondiente (ajusta el índice según tu lista)
    imagen_actual = images[1+3*i]  # o images[1+3*i] como usabas antes
    
    k_prom, delta_y, a_um, err_a_um = calcular_ancho_desde_espectro_adaptativo(
        imagen=imagen_actual,
        center_x=890, 
        center_y=1645, 
        offset=650,
        canal=2, 
        plot_espectro=False,  # Pon True solo para depurar, si no, satura la salida
        umbral_lobulo=0.6,
        factor_ancho=4.0,
        search_kx_max=5000,
        search_ky_min=1000, 
        search_ky_max=8000,
        pixel_size_m=pixel_size_m,
        lambda_m=lambda_m,
        D_m=D_m,
        err_pixel_size_m=err_pixel_size_m,
        err_lambda_m=err_lambda_m,
        err_D_m=err_D_m      # ¡FALTABA PASAR err_D_m!
    )

    if a_um is not None:
        rendijas_fft.append(a_um)
        err_rendijas_fft.append(err_a_um)
    else:
        rendijas_fft.append(np.nan)
        err_rendijas_fft.append(np.nan)
        print(f"⚠️ Imagen {i} no pudo ser procesada")

rendijas_fft = np.array(rendijas_fft)
err_rendijas_fft = np.array(err_rendijas_fft)

# Verificación rápida
print("\n📊 Resultados:")
for i, (a, err) in enumerate(zip(rendijas_fft, err_rendijas_fft)):
    if not np.isnan(a):
        print(f"Imagen {i}: a = {a:.2f} ± {err:.2f} µm")


# ==================== CONSTANTES Y CALIBRACIÓN ====================
px_por_mm = 23.88
err_px_por_mm = 0.01
rel_err_cal = err_px_por_mm / px_por_mm   # ≈ 0.0004188

# Constantes del experimento (verificar)
g = 9.80665          # m/s²
L = 0.29             # m (longitud de la viga)
x = L - 0.01430      # m (posición de la medición)
d = 0.00596          # m (diámetro de la viga)

# Factor geométrico para deflexión
C_geo = (32 / np.pi) * (1 / d**4) * (L * x**2 - x**3 / 3)

# Masas en gramos (reemplazar con tus valores)
masitas_g = np.array([0.8234, 0.6678, 1.0301, 2.0670, 3.1022,
                      5.1782, 4.1323, 3.7700, 2.7438, 1.4912])

# Convertir a kg
masitas_kg = masitas_g * 1e-3

# Convertir rendijas_fft a metros (si están en µm)
a_data_m = rendijas_fft * 1e-6
err_data_m_fft = err_rendijas_fft * 1e-6   # error estadístico de la FFT en metros

# Filtrar NaNs si los hay
mask = ~np.isnan(a_data_m)
masitas_kg = masitas_kg[mask]
masitas_g = masitas_g[mask]
a_data_m = a_data_m[mask]
err_data_m_fft = err_data_m_fft[mask]

# ==================== MODELO LINEAL ====================
def modelo_lineal(m, A, B):
    return A * m + B

# ==================== PASO 1: AJUSTE PRELIMINAR SIN PESOS ====================
popt_nosigma, _ = curve_fit(modelo_lineal, masitas_kg, a_data_m)
residuos_nosigma = a_data_m - modelo_lineal(masitas_kg, *popt_nosigma)
rmse = np.sqrt(np.mean(residuos_nosigma**2))   # error cuadrático medio en metros

print(f"RMSE del ajuste sin pesos: {rmse*1e6:.3f} µm")

# ==================== PASO 2: ERROR TOTAL POR PUNTO ====================
# Error de calibración (proporcional al valor)
err_cal_m = a_data_m * rel_err_cal

# Error total: combinación de la dispersión (RMSE) y calibración
# Nota: podrías también incluir err_data_m_fft si consideras que aporta algo,
# pero dado que es muy pequeño, lo omitimos. Si quieres incluirlo:
# err_total_m = np.sqrt(rmse**2 + err_cal_m**2 + err_data_m_fft**2)
err_total_m = np.sqrt(rmse**2 + err_cal_m**2)

print(f"Error de calibración típico: {np.mean(err_cal_m*1e6):.3f} µm")
print(f"Error total típico: {np.mean(err_total_m*1e6):.3f} µm")

# ==================== PASO 3: AJUSTE PONDERADO CON ERRORES REALISTAS ====================
popt, pcov = curve_fit(modelo_lineal, masitas_kg, a_data_m,
                       sigma=err_total_m, absolute_sigma=True)

A, B = popt
err_A, err_B = np.sqrt(np.diag(pcov))

# ==================== CÁLCULO DE MAGNITUDES DERIVADAS ====================
E_ajustado = C_geo * g / A
m0_ajustado = B / A

err_E = (C_geo * g / A**2) * err_A

# Error de m0 sin covarianza
err_m0 = np.sqrt((err_B / A)**2 + (B * err_A / A**2)**2)

# ==================== BONDAD DEL AJUSTE ====================
modelo = modelo_lineal(masitas_kg, A, B)
residuos = a_data_m - modelo
chi2 = np.sum((residuos / err_total_m)**2)
gl = len(a_data_m) - 2
chi2_red = chi2 / gl


p_chi = stats.chi2.sf(chi2, gl)
# interpretamos el resultado:
print('chi^2: ' + str(chi2))
print('p-valor del chi^2: ' + str(p_chi))


if p_chi<0.05:
    print('Se rechaza la hipótesis de que el modelo ajuste a los datos.')
else:
    print('No se puede rechazar la hipótesis de que el modelo ajuste a los datos.')
    
    
print(f"\n--- Resultados del ajuste ---")
print(f"Pendiente A = {A:.3e} ± {err_A:.3e} m/kg")
print(f"Ordenada B = {B:.3e} ± {err_B:.3e} m")
print(f"E = {E_ajustado:.3e} ± {err_E:.3e} Pa")
print(f"m0 = {m0_ajustado*1000:.3f} ± {err_m0*1000:.3f} g")
print(f"χ²_red = {chi2_red:.3f}")

# ==================== GRÁFICOS ====================
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Panel superior: datos y ajuste
axes[0].errorbar(masitas_g, a_data_m * 1e6, yerr=err_total_m * 1e6,
                 fmt='o', capsize=4, label='Datos con error total')

# Línea de ajuste
m_line = np.linspace(0, max(masitas_g) * 1.1, 300)
a_line = modelo_lineal(m_line * 1e-3, A, B)
axes[0].plot(m_line, a_line * 1e6, '--', linewidth=2, label='Ajuste lineal')

axes[0].set_ylabel('a [µm]')
axes[0].set_title(f'Ajuste del módulo de Young\n'
                  f'E = {E_ajustado:.3e} ± {err_E:.3e} Pa\n'
                  f'm₀ = {m0_ajustado*1000:.3f} ± {err_m0*1000:.3f} g\n'
                  f'χ²_red = {chi2_red:.3f}')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Panel inferior: residuos
axes[1].errorbar(masitas_g, residuos * 1e6, yerr=err_total_m * 1e6,
                 fmt='o', capsize=4, label='Residuos')
axes[1].axhline(0, linestyle='--', color='black')
axes[1].set_xlabel('Masa aplicada [g]')
axes[1].set_ylabel('Residuos [µm]')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()
