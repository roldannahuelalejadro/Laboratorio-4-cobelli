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

import os, csv
from datetime import datetime

mascaras = {
    "binaria": ajustar_filtro_radio_barrido_centrado_en_lobulo,
    "hamming": ajustar_filtro_hamming_radio_barrido_centrado_en_lobulo,
    "tukey": ajustar_filtro_tukey_radio_barrido_centrado_en_lobulo
}

radio_min_pix = 2
radio_max_pix = 30
n_radios = 30

for nombre_mascara, funcion_filtro in mascaras.items():

    print(f"\n========== PROCESANDO MÁSCARA: {nombre_mascara.upper()} ==========")

    carpeta_resultados = f"imagenes_analisis_{nombre_mascara}"
    os.makedirs(carpeta_resultados, exist_ok=True)

    rendijas = []
    err_rendijas = []
    datos_imagenes = []

    for i in range(10):

        img_idx = 1 + 3*i

        roi = preparar_roi(
            images[img_idx],
            center_x=890,
            center_y=1645,
            offset=650,
            canal=2
        )

        kx_rad, ky_rad = calcular_k_rad(roi, pixel_size_m)

        # ================= FILTRADO =================
        resultados = funcion_filtro(
            roi, kx_rad, ky_rad,
            px_por_mm=px_por_mm,
            err_px_por_mm=err_px_por_mm,
            k_min=1500,
            k_max=6000,
            ancho_kx=500,
            radio_min_pix=radio_min_pix,
            radio_max_pix=radio_max_pix,
            n_radios=n_radios
        )

        if resultados[0] is None:

            print(f"⚠️ Imagen {img_idx} no procesada")

            rendijas.append(np.nan)
            err_rendijas.append(np.nan)

            datos_imagenes.append({
                'imagen_idx': img_idx,
                'paso_mean_px': np.nan,
                'paso_std_px': np.nan,
                'error_total_px': np.nan,
                'r_opt_pix': np.nan,
                'delta_y_um': np.nan,
                'err_delta_y_um': np.nan,
                'a_um': np.nan,
                'err_a_um': np.nan,
                'n_peaks': 0
            })

            continue

        (pasos, paso_mean, paso_std, error_total_px, error_rel,
         r_opt_pix, img_filt, mascara, peaks) = resultados

        # ================= VISUALIZACIÓN =================

        f = np.fft.fft2(roi["matriz"])
        fshift = np.fft.fftshift(f)

        fig1 = visualizar_resultado_filtrado_binario(
            matriz_original=roi["matriz"],
            imagen_filtrada=img_filt,
            mascara_binaria=mascara,
            fshift=fshift,
            kx_rad=kx_rad,
            ky_rad=ky_rad,
            pixel_size_um=pixel_size_um,
            k_crop_rad_m=10000
        )

        fig1.savefig(
            os.path.join(
                carpeta_resultados,
                f'imagen_{img_idx:03d}_mascara_{nombre_mascara}.png'
            ),
            dpi=150,
            bbox_inches='tight'
        )

        plt.close(fig1)

        # ================= PERFIL =================

        col = roi["matriz"].shape[1] // 2

        fig2, ax = plt.subplots(figsize=(12,5))

        ax.plot(roi["matriz"][:, col], alpha=0.5, label='Original')
        ax.plot(img_filt[:, col], linewidth=2, label='Filtrada')

        ax.scatter(
            peaks,
            img_filt[peaks, col],
            s=70,
            color='red',
            zorder=5,
            label='Picos'
        )

        ax.set_title(f'Perfil vertical - Imagen {img_idx} ({nombre_mascara})')

        ax.set_xlabel('Fila')
        ax.set_ylabel('Intensidad')

        ax.legend()
        ax.grid(True, alpha=0.3)

        if len(peaks) > 1:

            text = (
                f'Paso: {paso_mean:.2f} ± {error_total_px:.2f} px\n'
                f'Radio óptimo: {r_opt_pix:.1f} px'
            )

            ax.text(
                0.02, 0.98,
                text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            )

        fig2.savefig(
            os.path.join(
                carpeta_resultados,
                f'imagen_{img_idx:03d}_perfil_{nombre_mascara}.png'
            ),
            dpi=150,
            bbox_inches='tight'
        )

        plt.close(fig2)

        # ================= CÁLCULO FÍSICO =================

        delta_y = paso_mean * pixel_size_m

        err_delta_y = np.sqrt(
            (paso_mean * err_pixel_size_m)**2 +
            (pixel_size_m * error_total_px)**2
        )

        a = (lambda_m * D_m) / delta_y

        err_a = np.sqrt(
            ((D_m / delta_y) * err_lambda_m)**2 +
            ((lambda_m / delta_y) * err_D_m)**2 +
            ((lambda_m * D_m / delta_y**2) * err_delta_y)**2
        )

        err_a = np.sqrt(err_a**2 + (a * 0.005)**2)

        a_um = a * 1e6
        err_a_um = err_a * 1e6

        delta_y_um = delta_y * 1e6
        err_delta_y_um = err_delta_y * 1e6

        print(f"\n[Imagen {img_idx} - {nombre_mascara}]")
        print(f"  Radio óptimo: {r_opt_pix:.2f} px")
        print(f"  Paso: {paso_mean:.2f} ± {error_total_px:.2f} px")
        print(f"  a = {a_um:.2f} ± {err_a_um:.2f} µm")

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

    # ================= GUARDAR CSV =================
    csv_filename = os.path.join(
        carpeta_resultados,
        f"resultados_{nombre_mascara}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'imagen_idx','paso_mean_px','paso_std_px','error_total_px',
            'r_opt_pix','delta_y_um','err_delta_y_um',
            'a_um','err_a_um','n_peaks'
        ])
        for d in datos_imagenes:
            writer.writerow([
                d['imagen_idx'],
                d['paso_mean_px'],
                d['paso_std_px'],
                d['error_total_px'],
                d['r_opt_pix'],
                d['delta_y_um'],
                d['err_delta_y_um'],
                d['a_um'],
                d['err_a_um'],
                d['n_peaks']
            ])

