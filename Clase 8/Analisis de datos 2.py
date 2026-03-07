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


imagen = images[1]
# preparar ROI
roi = preparar_roi(imagen, center_x=890, center_y=1645, offset=650, canal=2)

kx_rad, ky_rad = calcular_k_rad(roi, pixel_size_m)


res_bin = ajustar_filtro_radio_barrido_centrado_en_lobulo(
    roi, kx_rad, ky_rad,
    px_por_mm=px_por_mm,
    err_px_por_mm=err_px_por_mm,
    radio_min_pix=2,
    radio_max_pix=30,
    n_radios=20,
    plot_diagnostico=False
)

(_, _, _, _, _, r_bin, img_bin, mask_bin, peaks_bin) = res_bin


res_ham = ajustar_filtro_hamming_radio_barrido_centrado_en_lobulo(
    roi, kx_rad, ky_rad,
    px_por_mm=px_por_mm,
    err_px_por_mm=err_px_por_mm,
    radio_min_pix=2,
    radio_max_pix=30,
    n_radios=20
)

(_, _, _, _, _, r_ham, img_ham, mask_ham, peaks_ham) = res_ham


res_tuk = ajustar_filtro_tukey_radio_barrido_centrado_en_lobulo(
    roi, kx_rad, ky_rad,
    px_por_mm=px_por_mm,
    err_px_por_mm=err_px_por_mm,
    radio_min_pix=2,
    radio_max_pix=30,
    n_radios=20,
    alpha=0.5
)

(_, _, _, _, _, r_tuk, img_tuk, mask_tuk, peaks_tuk) = res_tuk

def zoom_fourier_region(fshift, mask, kx_rad, ky_rad, k_crop=10000):

    espectro = np.log10(1 + np.abs(fshift))

    ky_vals = ky_rad[:,0]
    kx_vals = kx_rad[0,:]

    idx_ky = np.where(np.abs(ky_vals) <= k_crop)[0]
    idx_kx = np.where(np.abs(kx_vals) <= k_crop)[0]

    slice_y = slice(idx_ky[0], idx_ky[-1]+1)
    slice_x = slice(idx_kx[0], idx_kx[-1]+1)

    espectro_zoom = espectro[slice_y, slice_x]
    mask_zoom = mask[slice_y, slice_x]

    extent = [
        kx_vals[idx_kx[0]],
        kx_vals[idx_kx[-1]],
        ky_vals[idx_ky[-1]],
        ky_vals[idx_ky[0]]
    ]

    return espectro_zoom, mask_zoom, extent


import matplotlib.patches as mpatches
import numpy as np

fshift = np.fft.fftshift(np.fft.fft2(roi["matriz"]))

spec_bin, mask_bin_z, extent = zoom_fourier_region(
    fshift, mask_bin, kx_rad, ky_rad, k_crop=6000
)

spec_ham, mask_ham_z, _ = zoom_fourier_region(
    fshift, mask_ham, kx_rad, ky_rad, k_crop=6000
)

spec_tuk, mask_tuk_z, _ = zoom_fourier_region(
    fshift, mask_tuk, kx_rad, ky_rad, k_crop=6000
)

fig, ax = plt.subplots(3,1, figsize=(6,13))

# ocultar ceros para no tapar el espectro
mask_bin_plot = np.ma.masked_where(mask_bin_z <= 0, mask_bin_z)
mask_ham_plot = np.ma.masked_where(mask_ham_z <= 0, mask_ham_z)
mask_tuk_plot = np.ma.masked_where(mask_tuk_z <= 0, mask_tuk_z)

# --- BINARIA ---
ax[0].imshow(spec_bin, cmap="magma", extent=extent)
im0 = ax[0].imshow(mask_bin_plot, cmap="Reds", alpha=0.9, extent=extent, vmin=0, vmax=1)

# --- HAMMING ---
ax[1].imshow(spec_ham, cmap="magma", extent=extent)
im1 = ax[1].imshow(mask_ham_plot, cmap="Blues", alpha=0.9, extent=extent, vmin=0, vmax=1)

# --- TUKEY ---
ax[2].imshow(spec_tuk, cmap="magma", extent=extent)
im2 = ax[2].imshow(mask_tuk_plot, cmap="Greens", alpha=0.9, extent=extent, vmin=0, vmax=1)

# colorbars con etiqueta
cbar0 = fig.colorbar(im0, ax=ax[0])
cbar1 = fig.colorbar(im1, ax=ax[1])
cbar2 = fig.colorbar(im2, ax=ax[2])

cbar0.set_label(r"$\omega(\rho)$")
cbar1.set_label(r"$\omega(\rho)$")
cbar2.set_label(r"$\omega(\rho)$")

# leyendas
patch_bin = mpatches.Patch(color="red", alpha=0.9, label="Máscara binaria")
patch_ham = mpatches.Patch(color="blue", alpha=0.9, label="Ventana de Hamming")
patch_tuk = mpatches.Patch(color="green", alpha=0.9, label="Ventana de Tukey")

ax[0].legend(handles=[patch_bin], loc="upper right")
ax[1].legend(handles=[patch_ham], loc="upper right")
ax[2].legend(handles=[patch_tuk], loc="upper right")

for a in ax:
    a.set_xlabel("$k_x$ (rad/m)")
    a.set_ylabel("$k_y$ (rad/m)")

plt.tight_layout()
plt.show()

