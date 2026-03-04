import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution



def preparar_roi(imagen,center_x=890, center_y=1645, offset=650, canal=2): # (Region Of Interest)
    """
    Extrae la región de interés y construye
    todas las variables geométricas necesarias
    para los filtros en Fourier.
    """

    matriz = imagen[
        center_x-offset:center_x+offset,
        center_y-offset:center_y+offset,
        canal
    ].astype(float)

    rows, cols = matriz.shape
    crow, ccol = rows//2, cols//2

    y, x = np.ogrid[:rows, :cols]
    kx = x - ccol
    ky = y - crow
    distancia = np.sqrt(kx**2 + ky**2)

    return {
        "matriz": matriz,
        "kx": kx,
        "ky": ky,
        "distancia": distancia,
        "col_central": ccol
    }



def localizar_lobulo(espectro_log, kx, ky, k_min=1500, k_max=6000, ancho_kx=500):
    """
    Localiza la posición aproximada del lóbulo principal en el espectro.
    Devuelve (kx0, ky0) en las mismas unidades que kx, ky.
    """
    # Seleccionar región de búsqueda
    mask_ky = (np.abs(ky) > k_min) & (np.abs(ky) < k_max)
    mask_kx = np.abs(kx) < ancho_kx
    
    # Región 2D de búsqueda
    region = espectro_log[mask_ky][:, mask_kx]
    
    # Encontrar el máximo
    idx_max = np.unravel_index(np.argmax(region), region.shape)
    
    # Mapear a índices globales
    ky_idx = np.where(mask_ky)[0][idx_max[0]]
    kx_idx = np.where(mask_kx)[0][idx_max[1]]
    
    return kx[kx_idx], ky[ky_idx]



def ajustar_filtro_circular_ml(roi,bounds=(8, 120)):
    print("Iniciando optimizacion")
    matriz = roi["matriz"]
    distancia = roi["distancia"]
    col = roi["col_central"]

    def loss_std_pasos(params):
        sigma = params[0]

        mascara = np.exp(-distancia**2 / (2 * sigma**2))

        f = np.fft.fft2(matriz)
        fshift = np.fft.fftshift(f)
        imagen_filtrada = np.real(
            np.fft.ifft2(np.fft.ifftshift(fshift * mascara))
        )

        perfil = imagen_filtrada[:, col]

        peaks, _ = find_peaks(
            perfil,
            height=np.max(perfil)*0.03,
            distance=25,
            prominence=np.max(perfil)*0.03
        )

        if len(peaks) < 4:
            return 1e6

        pasos = np.diff(peaks.astype(float))
        return np.std(pasos)

    result = differential_evolution(
        loss_std_pasos,
        bounds=[bounds],
        popsize=15,
        maxiter=30,
        tol=1e-3,
        workers=1,
        disp=False
    )

    sigma_opt = result.x[0]

    # Filtrado final
    mascara = np.exp(-distancia**2 / (2 * sigma_opt**2))
    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)
    imagen_filtrada = np.real(
        np.fft.ifft2(np.fft.ifftshift(fshift * mascara))
    )

    perfil = imagen_filtrada[:, col]
    peaks, _ = find_peaks(
        perfil,
        height=np.max(perfil)*0.03,
        distance=25,
        prominence=np.max(perfil)*0.03
    )

    if len(peaks) >= 2:
        pasos = np.diff(peaks.astype(float))
        return pasos, np.mean(pasos), np.std(pasos), sigma_opt, imagen_filtrada, mascara, peaks
    else:
        return None, None, None, sigma_opt, imagen_filtrada, mascara, peaks

def ajustar_filtro_eliptico_ml(roi):
    print("Iniciando optimizacion")
    matriz = roi["matriz"]
    kx = roi["kx"]
    ky = roi["ky"]
    col = roi["col_central"]

    def loss(params):
        sx, sy = params
        if sx <= 1 or sy <= 1:
            return 1e6

        mascara = np.exp(-(kx**2/(2*sx**2) +
                           ky**2/(2*sy**2)))

        f = np.fft.fft2(matriz)
        fshift = np.fft.fftshift(f)
        imagen_filtrada = np.real(
            np.fft.ifft2(np.fft.ifftshift(fshift * mascara))
        )

        perfil = imagen_filtrada[:, col]
        peaks, _ = find_peaks(perfil,
                              height=np.max(perfil)*0.03,
                              distance=25,
                              prominence=np.max(perfil)*0.03)

        if len(peaks) < 4:
            return 1e6

        pasos = np.diff(peaks.astype(float))
        return np.std(pasos)

    result = differential_evolution(
        loss,
        bounds=[(8, 120),(8, 120)],
        popsize=20,
        maxiter=40,
        tol=1e-3,
        workers=1,
        disp=False
    )

    sx_opt, sy_opt = result.x

    mascara = np.exp(-(kx**2/(2*sx_opt**2) +
                       ky**2/(2*sy_opt**2)))

    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)
    imagen_filtrada = np.real(
        np.fft.ifft2(np.fft.ifftshift(fshift * mascara))
    )

    perfil = imagen_filtrada[:, col]
    peaks, _ = find_peaks(perfil,
                          height=np.max(perfil)*0.03,
                          distance=25,
                          prominence=np.max(perfil)*0.03)

    if len(peaks) >= 2:
        pasos = np.diff(peaks.astype(float))
        return pasos, np.mean(pasos), np.std(pasos), (sx_opt, sy_opt), imagen_filtrada, mascara, peaks
    else:
        return None, None, None, (sx_opt, sy_opt), imagen_filtrada, mascara, peaks
    
def ajustar_filtro_radio_barrido(roi, radio_min=12, radio_max=200, n_radios=100):
    print("Iniciando optimizacion")
    matriz = roi["matriz"]
    distancia = roi["distancia"]
    col = roi["col_central"]

    radios = np.linspace(radio_min, radio_max, n_radios)

    stds = []
    radios_validos = []

    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)

    for r in radios:

        mascara = distancia <= r
        imagen_filtrada = np.real(
            np.fft.ifft2(np.fft.ifftshift(fshift * mascara))
        )

        perfil = imagen_filtrada[:, col]

        peaks, _ = find_peaks(
            perfil,
            height=np.max(perfil)*0.03,
            distance=25,
            prominence=np.max(perfil)*0.03
        )

        if len(peaks) >= 4:
            pasos = np.diff(peaks.astype(float))
            stds.append(np.std(pasos))
            radios_validos.append(r)

    if len(stds) == 0:
        return None, None, None, None

    r_opt = radios_validos[np.argmin(stds)]

    mascara = distancia <= r_opt
    imagen_filtrada = np.real(
        np.fft.ifft2(np.fft.ifftshift(fshift * mascara))
    )

    perfil = imagen_filtrada[:, col]
    peaks, _ = find_peaks(
        perfil,
        height=np.max(perfil)*0.03,
        distance=25,
        prominence=np.max(perfil)*0.03
    )

    if len(peaks) >= 2:
        pasos = np.diff(peaks.astype(float))
        return pasos, np.mean(pasos), np.std(pasos), r_opt, imagen_filtrada, mascara, peaks
    else:
        return None, None, None, r_opt, imagen_filtrada, mascara, peaks
    
def visualizar_resultado_filtrado(matriz_original, imagen_filtrada, mascara):
    """
    Visualiza únicamente:
    - Imagen original
    - Imagen filtrada
    - Máscara en espacio de Fourier
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Imagen original
    im0 = axes[0].imshow(matriz_original, cmap='gray')
    axes[0].set_title('Imagen Original')
    plt.colorbar(im0, ax=axes[0])

    # Imagen filtrada
    im1 = axes[1].imshow(imagen_filtrada, cmap='gray')
    axes[1].set_title('Imagen Filtrada')
    plt.colorbar(im1, ax=axes[1])

    # Máscara
    im2 = axes[2].imshow(mascara, cmap='jet')
    axes[2].set_title('Máscara (Fourier)')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

