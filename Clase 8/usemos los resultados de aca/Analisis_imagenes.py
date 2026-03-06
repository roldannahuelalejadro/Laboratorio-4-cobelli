import imageio.v2 as imageio
import matplotlib.pyplot as plt 
import csv
import os
import numpy as np
from pathlib import Path
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from utils import *
from datetime import datetime
import scipy.stats as stats
ROOT = Path(r"C:\Users\tomas\Desktop\Laboratorio-4-cobelli-main\Clase 8\young _2\aluminium_")


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

rendijas = []
err_rendijas = []

# ==================== CONSTANTES Y CALIBRACIÓN ====================
px_por_mm = 23.88
err_px_por_mm = 0.01
rel_err_cal = err_px_por_mm / px_por_mm

pixel_size_um = 1000 / px_por_mm
pixel_size_m = pixel_size_um * 1e-6
err_pixel_size_m = pixel_size_m * rel_err_cal

lambda_laser_nm = 650
err_lambda_nm = 1
lambda_m = lambda_laser_nm * 1e-9
err_lambda_m = err_lambda_nm * 1e-9

D_m = 0.5125
err_D_m = 0.001

# Parámetros del ajuste mecánico
g = 9.80665
L = 0.29
x = L  # Ajustar según corresponda
d = 0.00596

print(f"Calibración: {px_por_mm:.2f} ± {err_px_por_mm:.2f} px/mm")
print(f"Tamaño de píxel: {pixel_size_um:.4f} ± {pixel_size_um*rel_err_cal:.4f} µm")

# ==================== PREPARACIÓN ====================
carpeta_resultados = "imagenes_analisis_radio"
os.makedirs(carpeta_resultados, exist_ok=True)

rendijas = []
err_rendijas = []
datos_imagenes = []

# ==================== BUCLE PRINCIPAL ====================
radio_min_pix = 2
radio_max_pix = 30
n_radios = 30

for i in range(10):
    img_idx = 1 + 3*i
    
    # ROI
    roi = preparar_roi(images[img_idx], center_x=890, center_y=1645, offset=650, canal=2)
    kx_rad, ky_rad = calcular_k_rad(roi, pixel_size_m)
    
    # Llamada a la función (sin plot_diagnostico para no guardar optimización)
    resultados = ajustar_filtro_radio_barrido_centrado_en_lobulo(
        roi, kx_rad, ky_rad,
        px_por_mm=px_por_mm,
        err_px_por_mm=err_px_por_mm,
        k_min=1500, k_max=6000, ancho_kx=500,
        radio_min_pix=radio_min_pix,
        radio_max_pix=radio_max_pix,
        n_radios=n_radios,
        plot_diagnostico=False  
    )

    # Desempaquetar
    if resultados[0] is None:
        print(f"⚠️ Imagen {img_idx} no procesada")
        rendijas.append(np.nan)
        err_rendijas.append(np.nan)
        datos_imagenes.append({'imagen_idx': img_idx, 'paso_mean_px': np.nan,
                               'paso_std_px': np.nan, 'error_total_px': np.nan,
                               'r_opt_pix': np.nan, 'delta_y_um': np.nan, 
                               'err_delta_y_um': np.nan, 'a_um': np.nan, 
                               'err_a_um': np.nan, 'n_peaks': 0})
        continue
    
    # Desempaquetar resultados (sin diagnóstico)
    (pasos, paso_mean, paso_std, error_total_px, error_rel,
     r_opt_pix, img_filt, mascara_binaria, peaks) = resultados
    
    # ===== VISUALIZACIONES =====
    f = np.fft.fft2(roi["matriz"])
    fshift = np.fft.fftshift(f)
    
    # 1. Máscara binaria sobre espectro (CON ZOOM)
    fig1 = visualizar_resultado_filtrado_binario(
        matriz_original=roi["matriz"],
        imagen_filtrada=img_filt,
        mascara_binaria=mascara_binaria,
        fshift=fshift,
        kx_rad=kx_rad,
        ky_rad=ky_rad,
        pixel_size_um=pixel_size_um,
        k_crop_rad_m=10000  # Zoom de 10000 rad/m alrededor del lóbulo
    )
    
    fig1.savefig(os.path.join(carpeta_resultados, f'imagen_{img_idx:03d}_mascara_binaria.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Perfil con picos
    col = roi["matriz"].shape[1] // 2
    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.plot(roi["matriz"][:, col], alpha=0.5, label='Original')
    ax.plot(img_filt[:, col], linewidth=2, label='Filtrada')
    ax.scatter(peaks, img_filt[peaks, col], s=70, color='red', zorder=5, label='Picos')
    ax.set_title(f'Perfil vertical - Imagen {img_idx}')
    ax.set_xlabel('Fila')
    ax.set_ylabel('Intensidad')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Añadir estadísticas
    if len(peaks) > 1:
        text = f'Paso: {paso_mean:.2f} ± {error_total_px:.2f} px\nRadio óptimo: {r_opt_pix:.1f} px'
        ax.text(0.02, 0.98, text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig2.savefig(os.path.join(carpeta_resultados, f'imagen_{img_idx:03d}_perfil.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # ===== CÁLCULO DE ANCHO DE RENDIJA =====
    delta_y = paso_mean * pixel_size_m
    err_delta_y = np.sqrt((paso_mean * err_pixel_size_m)**2 + (pixel_size_m * error_total_px)**2)
    
    a = (lambda_m * D_m) / delta_y
    err_a = np.sqrt(
        ((D_m / delta_y) * err_lambda_m)**2 +
        ((lambda_m / delta_y) * err_D_m)**2 +
        ((lambda_m * D_m / delta_y**2) * err_delta_y)**2
    )
    
    # Error adicional (0.5%)
    err_a = np.sqrt(err_a**2 + (a * 0.005)**2)
    
    a_um = a * 1e6
    err_a_um = err_a * 1e6
    delta_y_um = delta_y * 1e6
    err_delta_y_um = err_delta_y * 1e6
    
    print(f"\n[Imagen {img_idx}]")
    print(f"  Radio óptimo: {r_opt_pix:.2f} px")
    print(f"  Paso: {paso_mean:.2f} ± {error_total_px:.2f} px")
    print(f"  Δy: {delta_y_um:.2f} ± {err_delta_y_um:.2f} µm")
    print(f"  a = {a_um:.2f} ± {err_a_um:.2f} µm  |  picos: {len(peaks)}")
    
    rendijas.append(a)
    err_rendijas.append(err_a)
    
    datos_imagenes.append({
        'imagen_idx': img_idx,
        'paso_mean_px': paso_mean,
        'paso_std_px': paso_std,
        'error_total_px': error_total_px,
        'r_opt_pix': r_opt_pix,
        'delta_y_um': delta_y_um,
        'err_delta_y_um': err_delta_y_um,
        'a_um': a_um,
        'err_a_um': err_a_um,
        'n_peaks': len(peaks)
    })
# ==================== GUARDADO CSV ====================
csv_filename = os.path.join(
    carpeta_resultados,
    f"resultados_binario_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['imagen_idx', 'paso_mean_px', 'paso_std_px', 'error_total_px',
                     'r_opt_pix', 'delta_y_um', 'err_delta_y_um',
                     'a_um', 'err_a_um', 'n_peaks'])
    for d in datos_imagenes:
        writer.writerow([
            d['imagen_idx'],
            f"{d['paso_mean_px']:.4f}" if not np.isnan(d['paso_mean_px']) else 'NaN',
            f"{d['paso_std_px']:.4f}" if not np.isnan(d['paso_std_px']) else 'NaN',
            f"{d['error_total_px']:.4f}" if not np.isnan(d['error_total_px']) else 'NaN',
            f"{d['r_opt_pix']:.4f}" if not np.isnan(d['r_opt_pix']) else 'NaN',
            f"{d['delta_y_um']:.4f}" if not np.isnan(d['delta_y_um']) else 'NaN',
            f"{d['err_delta_y_um']:.4f}" if not np.isnan(d['err_delta_y_um']) else 'NaN',
            f"{d['a_um']:.4f}" if not np.isnan(d['a_um']) else 'NaN',
            f"{d['err_a_um']:.4f}" if not np.isnan(d['err_a_um']) else 'NaN',
            d['n_peaks']
        ])

# ==================== ANÁLISIS FINAL ====================
rendijas = np.array(rendijas)
err_rendijas = np.array(err_rendijas)
mask = ~np.isnan(rendijas)

# ==================== AJUSTES ====================
masitas = np.array([0.8234, 0.6678, 1.0301, 2.0670, 3.1022,
                    5.1782, 4.1323, 3.7700, 2.7438, 1.4912])  # g
m_kg = masitas * 1e-3

# Filtrar datos válidos
rendijas_validas = rendijas[mask]
err_validos = err_rendijas[mask]
m_kg_validas = m_kg[mask]
masitas_validas = masitas[mask]

# ==================== MODELO FÍSICO ====================
# Parámetros del ajuste mecánico
g = 9.80665
L = 0.29
x = L  # Ajustar según corresponda
d = 0.00596

def modelo_fisico(m, E, b):
    return (32 / np.pi) * (1 / d**4) * ((m * g) / E) * (L * x**2 - x**3/3) + b

popt, pcov = curve_fit(modelo_fisico, m_kg_validas, rendijas_validas,
                       sigma=err_validos, absolute_sigma=True)
E_opt, b_opt = popt
err_E = np.sqrt(pcov[0, 0])

# Chi² y p-valor para modelo físico
residuos_fisico = rendijas_validas - modelo_fisico(m_kg_validas, E_opt, b_opt)
chi2_fisico = np.sum((residuos_fisico / err_validos)**2)
gl_fisico = len(rendijas_validas) - len(popt)
chi2_red_fisico = chi2_fisico / gl_fisico
p_valor_fisico = stats.chi2.sf(chi2_fisico, gl_fisico)

# ==================== MODELO LINEAL ====================
coef_lin, cov_lin = np.polyfit(m_kg_validas, rendijas_validas, 1,
                                w=1/err_validos, cov=True)
pendiente, intercepto = coef_lin
err_pend, err_int = np.sqrt(np.diag(cov_lin))

# Chi² y p-valor para modelo lineal
modelo_lineal = pendiente * m_kg_validas + intercepto
residuos_lineal = rendijas_validas - modelo_lineal  
chi2_lineal = np.sum((residuos_lineal / err_validos)**2)
gl_lineal = len(rendijas_validas) - 2
chi2_red_lineal = chi2_lineal / gl_lineal
p_valor_lineal = stats.chi2.sf(chi2_lineal, gl_lineal)

# ==================== RESULTADOS DE LOS AJUSTES ====================
print("\n" + "="*60)
print("RESULTADOS DE LOS AJUSTES")
print("="*60)

print("\n📌 MODELO FÍSICO (E young):")
print(f"   E = ({E_opt:.3e} ± {err_E:.3e}) Pa")
print(f"   b = {b_opt:.3e} m")
print(f"   χ² = {chi2_fisico:.3f}")
print(f"   χ²/gl = {chi2_red_fisico:.3f}")
print(f"   p-valor = {p_valor_fisico:.4f}")
if p_valor_fisico > 0.05:
    print("   ✅ El modelo es aceptable (p > 0.05)")
else:
    print("   ⚠️ El modelo podría no ser adecuado (p < 0.05)")

print("\n📌 MODELO LINEAL:")
print(f"   Pendiente = {pendiente:.3e} ± {err_pend:.3e} m/kg")
print(f"   Intercepto = {intercepto:.3e} ± {err_int:.3e} m")
print(f"   χ² = {chi2_lineal:.3f}")
print(f"   χ²/gl = {chi2_red_lineal:.3f}")
print(f"   p-valor = {p_valor_lineal:.4f}")
if p_valor_lineal > 0.05:
    print("   ✅ El modelo es aceptable (p > 0.05)")
else:
    print("   ⚠️ El modelo podría no ser adecuado (p < 0.05)")

# ==================== GRÁFICOS ====================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# Datos y ajustes
ax1.errorbar(masitas_validas, rendijas_validas*1e6, yerr=err_validos*1e6,
             fmt='o', capsize=4, label='Datos', zorder=5)

m_line = np.linspace(masitas.min(), masitas.max(), 200)
ax1.plot(m_line, modelo_fisico(m_line*1e-3, E_opt, b_opt)*1e6,
         '--', linewidth=2, 
         label=f'Modelo físico: E = {E_opt:.2e} Pa\nχ²/gl={chi2_red_fisico:.2f}, p={p_valor_fisico:.3f}')
ax1.plot(m_line, (pendiente*m_line*1e-3 + intercepto)*1e6,
         ':', linewidth=2,
         label=f'Ajuste lineal\nχ²/gl={chi2_red_lineal:.2f}, p={p_valor_lineal:.3f}')

ax1.set_ylabel('a [µm]')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Residuos
ax2.errorbar(masitas_validas, residuos_fisico*1e6, yerr=err_validos*1e6,
             fmt='o', capsize=4, label='Residuos (físico)', alpha=0.7)
ax2.errorbar(masitas_validas, residuos_lineal*1e6, yerr=err_validos*1e6,
             fmt='s', capsize=4, label='Residuos (lineal)', alpha=0.7)
ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
ax2.set_xlabel('Masa [g]')
ax2.set_ylabel('Residuos [µm]')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

plt.suptitle(f'Comparación de modelos - {len(rendijas_validas)} puntos válidos')
plt.tight_layout()
plt.savefig(os.path.join(carpeta_resultados, 'comparacion_modelos.png'), dpi=150)
plt.show()

print(f"\n✅ Resultados guardados en: {carpeta_resultados}")