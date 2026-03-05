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

ROOT = Path(r"C:/Users/publico/Desktop/Labo 4 verano grupo 3 2026/Laboratorio-4-cobelli-main/Clase 8/young _2/aluminium_")


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

# Constantes de calibración con sus errores
px_por_mm = 23.88
err_px_por_mm = 0.01
rel_err_cal = err_px_por_mm / px_por_mm   # ≈ 0.0004188 (0.04%)

# Tamaño de píxel (derivado)
pixel_size_um = 1000 / px_por_mm
pixel_size_m = pixel_size_um * 1e-6

# Error del tamaño de píxel (propagado desde px_por_mm)
# pixel_size_m = (1000 * 1e-6) / px_por_mm = 0.001 / px_por_mm
err_pixel_size_m = pixel_size_m * rel_err_cal  # Mismo error relativo

# Constantes del experimento con sus errores
lambda_laser_nm = 650
err_lambda_nm = 1  # Estimación (podría ser 0.1 nm o lo que corresponda)
lambda_m = lambda_laser_nm * 1e-9
err_lambda_m = err_lambda_nm * 1e-9

D_m = 0.5125
err_D_m = 0.001  # Error en la distancia (metros)

print(f"Error relativo de calibración: {rel_err_cal*100:.4f}%")
print(f"Tamaño de píxel: {pixel_size_um:.4f} ± {pixel_size_um*rel_err_cal:.4f} µm")

# Crear carpeta para guardar resultados
carpeta_resultados = "imagenes_analisis_circulares"
os.makedirs(carpeta_resultados, exist_ok=True)

# Inicializar listas para guardar resultados
rendijas = []
err_rendijas = []

# Lista para guardar TODOS los datos de cada imagen
datos_imagenes = []

# Bucle principal
for i in range(10):
    print(f"\n--- Procesando imagen {i+1}/10 ---")
    
    # Preparar ROI
    img_idx = 1 + 3*i
    roi = preparar_roi(images[img_idx], center_x=890, center_y=1645, offset=650, canal=2)
    
    # Calcular kx_rad, ky_rad (para localización y visualización)
    kx_rad, ky_rad = calcular_k_rad(roi, pixel_size_m)
    
    # Aplicar filtro centrado en lóbulo
    resultados = ajustar_filtro_circular_centrado_en_lobulo(
        roi, kx_rad, ky_rad,
        k_min=1500, k_max=6000, ancho_kx=500
    )

    pasos, paso_mean, paso_std, sx_opt, imagen_filtrada, mascara_pix, peaks = resultados
    
    if paso_mean is None:
        print("⚠️ No se pudo procesar esta imagen")
        rendijas.append(np.nan)
        err_rendijas.append(np.nan)
        # Guardar datos con NaN
        datos_imagenes.append({
            'imagen_idx': img_idx,
            'paso_mean_px': np.nan,
            'paso_std_px': np.nan,
            'sx_opt': np.nan,
            'delta_y_um': np.nan,
            'err_delta_y_um': np.nan,
            'a_um': np.nan,
            'err_a_um': np.nan,
            'n_peaks': 0
        })
        continue
    
    # ===== VISUALIZACIONES (guardadas en archivos) =====
    
    # Calcular FFT para visualización
    f = np.fft.fft2(roi["matriz"])
    fshift = np.fft.fftshift(f)
    
    # 1. Visualización de máscara sobre espectro
    fig1 = visualizar_resultado_filtrado(
        roi["matriz"], 
        imagen_filtrada, 
        mascara_pix,
        fshift=fshift,
        kx_rad=kx_rad,
        ky_rad=ky_rad
    )
    fig1.savefig(os.path.join(carpeta_resultados, f'imagen_{img_idx:03d}_mascara.png'), 
                 dpi=150, bbox_inches='tight')
    plt.show(fig1)
    plt.close(fig1)
    
    # 2. Gráfico del perfil
    col = roi["matriz"].shape[1] // 2
    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.plot(roi["matriz"][:, col], alpha=0.5, label='Original')
    ax.plot(imagen_filtrada[:, col], linewidth=2, label='Filtrada')
    ax.scatter(peaks, imagen_filtrada[peaks, col], s=70, color='red', zorder=5, label='Picos detectados')
    ax.set_title(f'Perfil vertical central - Imagen {img_idx}')
    ax.set_xlabel('Fila')
    ax.set_ylabel('Intensidad')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig2.savefig(os.path.join(carpeta_resultados, f'imagen_{img_idx:03d}_perfil.png'), 
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # ===== CÁLCULO DEL ANCHO DE RENDIJA =====
    
    # 1. Espaciado de franjas en metros (Δy = paso × pixel_size)
    delta_y = paso_mean * pixel_size_m
    
    # Error en Δy
    err_delta_y = np.sqrt(
        (paso_mean * err_pixel_size_m)**2 +
        (pixel_size_m * paso_std)**2
    )
    
    # 2. Ancho de rendija (a = λ·D / Δy)
    a = (lambda_m * D_m) / delta_y
    
    # Propagación de errores
    err_a = np.sqrt(
        ((D_m / delta_y) * err_lambda_m)**2 +
        ((lambda_m / delta_y) * err_D_m)**2 +
        ((lambda_m * D_m / delta_y**2) * err_delta_y)**2
    )
    
    # Error relativo extra (0.5%)
    err_rel_extra = 0.005
    err_a = np.sqrt(err_a**2 + (a * err_rel_extra)**2)
    
    # Convertir a micrómetros
    a_um = a * 1e6
    err_a_um = err_a * 1e6
    delta_y_um = delta_y * 1e6
    err_delta_y_um = err_delta_y * 1e6
    
    # Mostrar resultados en consola
    print(f"\n=== RESULTADO {i+1}: ANCHO DE LA RENDIJA ===")
    print(f"Paso medio = {paso_mean:.2f} ± {paso_std:.2f} px")
    print(f"Δy = {delta_y_um:.2f} ± {err_delta_y_um:.2f} µm")
    print(f"Sigma óptimo x = {sx_opt:.2f} píxeles de frecuencia")
    print(f"a = {a_um:.2f} ± {err_a_um:.2f} µm")
    print(f"Número de picos detectados: {len(peaks)}")
    
    # Guardar resultados en listas
    rendijas.append(a)
    err_rendijas.append(err_a)
    
    # Guardar datos en la lista
    datos_imagenes.append({
        'imagen_idx': img_idx,
        'paso_mean_px': paso_mean,
        'paso_std_px': paso_std,
        'sx_opt': sx_opt,
        'delta_y_um': delta_y_um,
        'err_delta_y_um': err_delta_y_um,
        'a_um': a_um,
        'err_a_um': err_a_um,
        'n_peaks': len(peaks)
    })

# ===== GUARDAR CSV DESPUÉS DEL BUCLE (UNA SOLA VEZ) =====
csv_filename = os.path.join(carpeta_resultados, f"resultados_elipticos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

with open(csv_filename, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Escribir cabecera
    writer.writerow([
        'imagen_idx', 
        'paso_mean_px', 'paso_std_px',
        'sx_opt',
        'delta_y_um', 'err_delta_y_um',
        'a_um', 'err_a_um',
        'n_peaks'
    ])
    
    # Escribir todos los datos
    for datos in datos_imagenes:
        writer.writerow([
            datos['imagen_idx'],
            f"{datos['paso_mean_px']:.4f}" if not np.isnan(datos['paso_mean_px']) else 'NaN',
            f"{datos['paso_std_px']:.4f}" if not np.isnan(datos['paso_std_px']) else 'NaN',
            f"{datos['sx_opt']:.4f}" if not np.isnan(datos['sx_opt']) else 'NaN',
            f"{datos['delta_y_um']:.4f}" if not np.isnan(datos['delta_y_um']) else 'NaN',
            f"{datos['err_delta_y_um']:.4f}" if not np.isnan(datos['err_delta_y_um']) else 'NaN',
            f"{datos['a_um']:.4f}" if not np.isnan(datos['a_um']) else 'NaN',
            f"{datos['err_a_um']:.4f}" if not np.isnan(datos['err_a_um']) else 'NaN',
            datos['n_peaks']
        ])

# ===== ANÁLISIS FINAL =====
rendijas = np.array(rendijas)
err_rendijas = np.array(err_rendijas)
mask = ~np.isnan(rendijas)

if np.any(mask):
    a_mean = np.mean(rendijas[mask])
    a_std = np.std(rendijas[mask])
    a_mean_err = a_std / np.sqrt(np.sum(mask))
    
    print("\n" + "="*50)
    print("RESUMEN FINAL")
    print("="*50)
    print(f"Ancho medio de rendija: {a_mean*1e6:.2f} ± {a_std*1e6:.2f} µm (desviación estándar)")
    print(f"Error estándar de la media: {a_mean_err*1e6:.2f} µm")
    print(f"Error típico individual (promedio): {np.mean(err_rendijas[mask])*1e6:.2f} µm")
    
    # Histograma
    plt.figure(figsize=(10, 6))
    plt.hist(rendijas[mask]*1e6, bins=5, alpha=0.7, edgecolor='black')
    plt.axvline(a_mean*1e6, color='red', linestyle='--', label=f'Media: {a_mean*1e6:.2f} µm')
    plt.xlabel('Ancho de rendija (µm)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de anchos de rendija medidos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("No se obtuvieron mediciones válidas")

masitas =  np.array([0.8234 , 0.6678, 1.0301,2.0670,3.1022,5.1782,4.1323,3.7700,2.7438,1.4912])  #en


plt.figure()

plt.errorbar(masitas,
             rendijas,
             yerr=err_rendijas,
             fmt='o',
             label="Ancho de la rendija")

plt.xlabel("Masa [g]")
plt.ylabel("a [m]")
plt.grid(True)
plt.legend()
plt.show()

#todo en metros
g=  9.80665 #m/s
L=  0.29 # m
x = L # m  #cambiar por el valor debido
d = 0.00596 #m

def f(m, E, b):
    return (32/np.pi)*(1/d**4)*((m*g)/E)*(L*x**2 - (x**3)/3)+ b


m_kg = masitas * 1e-3

popt, pcov = curve_fit(
    f,
    m_kg,
    rendijas,
    sigma=err_rendijas,
    absolute_sigma=True
)

E_ajustado = popt[0]
b_ajustado =  popt[1]
err_E = np.sqrt(pcov[0,0])

print("E =", E_ajustado, "+/-", err_E)

modelo = f(m_kg, E_ajustado, b_ajustado)
residuos = rendijas - modelo

chi2 = np.sum(((rendijas - modelo)/err_rendijas)**2)
gl = len(rendijas) - len(popt)   # N - parámetros
chi2_red = chi2 / gl


fig, axs = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

# ---- Ajuste ----
axs[0].errorbar(masitas,
                rendijas,
                yerr=err_rendijas,
                fmt='o',
                label="Datos")

m_linea = np.linspace(min(masitas), max(masitas), 300)
axs[0].plot(m_linea,
            f(m_linea*1e-3, E_ajustado, b_ajustado),
            '--',
            label=f"Ajuste\nE = {E_ajustado:.2e} ± {err_E:.2e} Pa\n"
                  f"χ²_red = {chi2_red:.2f}")

axs[0].set_ylabel("a [m]")
axs[0].grid(True)
axs[0].legend()

# ---- Residuos ----
axs[1].errorbar(masitas,
                residuos,
                yerr=err_rendijas,
                fmt='o')

axs[1].axhline(0, linestyle='--')
axs[1].set_xlabel("Masa [kg]")
axs[1].set_ylabel("Residuos [m]")
axs[1].grid(True)

plt.tight_layout()
plt.show()


# =========================
# AJUSTE LINEAL POLYFIT (grado 1)
# =========================

coef, cov_lin = np.polyfit(
    m_kg,
    rendijas,
    1,
    w=1/np.array(err_rendijas),
    cov=True
)

pendiente = coef[0]
intercepto = coef[1]

err_pend = np.sqrt(cov_lin[0,0])
err_int = np.sqrt(cov_lin[1,1])

modelo_lin = pendiente*m_kg + intercepto
residuos_lin = rendijas - modelo_lin

chi2_lin = np.sum(((rendijas - modelo_lin)/err_rendijas)**2)
gl_lin = len(rendijas) - 2
chi2_red_lin = chi2_lin / gl_lin


# =========================
# GRÁFICOS
# =========================
fig, axs = plt.subplots(2, 1, figsize=(7, 9), sharex=True)

# ---- Ajustes ----
axs[0].errorbar(masitas,
                rendijas,
                yerr=err_rendijas,
                fmt='o',
                label="Datos")

m_linea = np.linspace(min(masitas), max(masitas), 300)

# Modelo físico
axs[0].plot(m_linea,
            f(m_linea*1e-3, E_ajustado, b_ajustado ),
            '--',
            label=f"Modelo físico\nE = {E_ajustado:.2e} ± {err_E:.2e} Pa\n"
                  f"χ²_red = {chi2_red:.2f}")

# Ajuste lineal libre
axs[0].plot(m_linea,
            pendiente*(m_linea*1e-3) + intercepto,
            ':',
            label=f"Lineal (polyfit)\n"
                  f"χ²_red = {chi2_red_lin:.2f}")

axs[0].set_ylabel("a [m]")
axs[0].grid(True)
axs[0].legend()


# ---- Residuos ----
axs[1].errorbar(masitas,
                residuos,
                yerr=err_rendijas,
                fmt='o',
                label="Residuos modelo físico")

axs[1].errorbar(masitas,
                residuos_lin,
                yerr=err_rendijas,
                fmt='x',
                label="Residuos lineal")

axs[1].axhline(0, linestyle='--')
axs[1].set_xlabel("Masa [g]")
axs[1].set_ylabel("Residuos [m]")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()



