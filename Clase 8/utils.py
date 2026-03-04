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

def calcular_k_rad(roi, pixel_size_m):
    """
    Calcula los arrays 2D de frecuencia angular (rad/m) a partir de la ROI.
    
    Parámetros:
    - roi: diccionario de preparar_roi (contiene 'matriz')
    - pixel_size_m: tamaño de píxel en metros
    
    Retorna:
    - kx_rad, ky_rad: arrays 2D de frecuencia angular (rad/m)
    """
    Ny, Nx = roi["matriz"].shape
    
    # Frecuencias en ciclos/metro, centradas
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, pixel_size_m))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, pixel_size_m))
    
    # Convertir a radianes/metro
    kx = fx * 2 * np.pi
    ky = fy * 2 * np.pi
    
    # Crear grids 2D
    kx_rad, ky_rad = np.meshgrid(kx, ky)
    
    return kx_rad, ky_rad

def localizar_lobulo(roi, kx_rad, ky_rad, k_min=1500, k_max=6000, ancho_kx=500):
    """
    Localiza la posición aproximada del lóbulo principal en el espectro.
    Calcula el espectro internamente a partir de la ROI.
    
    Parámetros:
    - roi: diccionario de preparar_roi (con matriz)
    - kx_rad, ky_rad: arrays 2D calibrados (rad/m)
    - k_min, k_max: rango de búsqueda en ky (rad/m)
    - ancho_kx: semi-ancho de búsqueda en kx (rad/m)
    
    Devuelve:
    - (kx0, ky0): coordenadas del lóbulo en rad/m
    """
    # Calcular espectro a partir de la matriz de la ROI
    matriz = roi["matriz"]
    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)
    espectro_log = np.log10(1 + np.abs(fshift))
    
    # Seleccionar región de búsqueda (asumiendo que kx_rad y ky_rad tienen la forma de la matriz)
    mask_ky = (np.abs(ky_rad) > k_min) & (np.abs(ky_rad) < k_max)
    mask_kx = np.abs(kx_rad) < ancho_kx
    
    # Aplicar máscaras
    espectro_enmascarado = np.where(mask_ky & mask_kx, espectro_log, 0)
    
    # Encontrar el máximo
    idx_max = np.unravel_index(np.argmax(espectro_enmascarado), espectro_enmascarado.shape)
    
    # Obtener coordenadas en rad/m
    kx0 = kx_rad[idx_max]
    ky0 = ky_rad[idx_max]
    
    print(f"Lóbulo localizado en: kx0 = {kx0:.1f} rad/m, ky0 = {ky0:.1f} rad/m")
    
    return kx0, ky0

def localizar_lobulo_en_pixeles(roi, kx_rad, ky_rad, k_min=1500, k_max=6000, ancho_kx=500):
    """
    Localiza la posición aproximada del lóbulo principal en el espectro.
    Devuelve los ÍNDICES (fila, columna) del centro del lóbulo.
    
    Parámetros:
    - roi: diccionario de preparar_roi (con matriz)
    - kx_rad, ky_rad: arrays 2D calibrados (rad/m) - SOLO PARA ENMASCARAR
    - k_min, k_max: rango de búsqueda en ky (rad/m)
    - ancho_kx: semi-ancho de búsqueda en kx (rad/m)
    
    Devuelve:
    - (idx_ky, idx_kx): índices del lóbulo en el array de Fourier
    """
    # Calcular espectro
    matriz = roi["matriz"]
    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)
    espectro = np.abs(fshift)
    espectro_log = np.log10(1 + espectro)
    
    # Crear máscara en rad/m para la región de búsqueda
    mask_ky = (np.abs(ky_rad) > k_min) & (np.abs(ky_rad) < k_max)
    mask_kx = np.abs(kx_rad) < ancho_kx
    mask_total = mask_ky & mask_kx
    
    # Aplicar máscara (fuera de la región de búsqueda, poner -infinito)
    espectro_enmascarado = np.where(mask_total, espectro_log, -np.inf)
    
    # Encontrar el máximo
    idx_max = np.unravel_index(np.argmax(espectro_enmascarado), espectro_enmascarado.shape)
    
    kx0 = kx_rad[idx_max]
    ky0 = ky_rad[idx_max]
    print(f"Lóbulo localizado en: kx0 = {kx0:.1f} rad/m, ky0 = {ky0:.1f} rad/m")
    print(f"Índices: fila={idx_max[0]}, columna={idx_max[1]}")
    
    return idx_max  # (fila, columna) en el array de Fourier

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

def ajustar_filtro_circular_centrado_en_lobulo(roi, kx_rad, ky_rad,
                                               k_min=1500, k_max=6000, ancho_kx=500,
                                               bounds_sigma_pix=(2, 30)):
    """
    Optimiza un filtro gaussiano circular CENTRADO en la posición del lóbulo.
    TRABAJA EN PÍXELES DE FRECUENCIA.
    
    Parámetros:
    - roi: diccionario de preparar_roi (con matriz)
    - kx_rad, ky_rad: arrays 2D calibrados (rad/m) - SOLO PARA LOCALIZAR
    - k_min, k_max, ancho_kx: parámetros para localizar_lobulo_en_pixeles
    - bounds_sigma_pix: rango de búsqueda para sigma en PÍXELES de frecuencia
    
    Retorna:
    - pasos, paso_promedio, std_pasos, sigma_opt_pix, imagen_filtrada, mascara_pix, peaks
    """
    print("Iniciando optimización de filtro centrado en lóbulo (en píxeles)")
    
    # 1. Localizar el lóbulo (obtenemos ÍNDICES)
    idx_ky0, idx_kx0 = localizar_lobulo_en_pixeles(roi, kx_rad, ky_rad, 
                                                    k_min, k_max, ancho_kx)
    
    # 2. Extraer datos del ROI
    matriz = roi["matriz"]
    col_central = roi["col_central"]
    
    # 3. Obtener coordenadas en píxeles de frecuencia
    kx_pix = roi["kx"]  # estos están en píxeles, centrados
    ky_pix = roi["ky"]
    
    # 4. Construir distancia al centro del lóbulo en PÍXELES
    distancia_al_lobulo_pix = np.sqrt((kx_pix - kx_pix[0, idx_kx0])**2 + 
                                       (ky_pix - ky_pix[idx_ky0, 0])**2)
    
    # 5. Precalcular FFT
    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)
    
    # 6. Función de pérdida (trabaja con sigma en píxeles)
    def loss_std_pasos(params):
        sigma_pix = params[0]
        
        # Máscara gaussiana centrada en el lóbulo (en píxeles)
        mascara = np.exp(-distancia_al_lobulo_pix**2 / (2 * sigma_pix**2))
        
        # Aplicar filtro
        imagen_filtrada = np.real(
            np.fft.ifft2(np.fft.ifftshift(fshift * mascara))
        )
        
        # Detectar picos en el perfil vertical central
        perfil = imagen_filtrada[:, col_central]
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
    
    # 7. Optimización
    result = differential_evolution(
        loss_std_pasos,
        bounds=[bounds_sigma_pix],
        popsize=15,
        maxiter=30,
        tol=1e-3,
        workers=1,
        disp=False
    )
    
    sigma_opt_pix = result.x[0]
    print(f"Sigma óptimo: {sigma_opt_pix:.1f} píxeles de frecuencia")
    
    # 8. Filtrado final con sigma óptimo
    mascara_opt_pix = np.exp(-distancia_al_lobulo_pix**2 / (2 * sigma_opt_pix**2))
    imagen_filtrada = np.real(
        np.fft.ifft2(np.fft.ifftshift(fshift * mascara_opt_pix))
    )
    
    # 9. Detección final de picos
    perfil = imagen_filtrada[:, col_central]
    peaks, _ = find_peaks(
        perfil,
        height=np.max(perfil)*0.03,
        distance=25,
        prominence=np.max(perfil)*0.03
    )
    
    if len(peaks) >= 2:
        pasos = np.diff(peaks.astype(float))
        paso_promedio = np.mean(pasos)
        std_pasos = np.std(pasos)
        return pasos, paso_promedio, std_pasos, sigma_opt_pix, imagen_filtrada, mascara_opt_pix, peaks
    else:
        print("⚠️ No se detectaron suficientes picos después del filtrado")
        return None, None, None, sigma_opt_pix, imagen_filtrada, mascara_opt_pix, peaks

def ajustar_filtro_eliptico_centrado_en_lobulo(roi, kx_rad, ky_rad,
                                               k_min=1500, k_max=6000, ancho_kx=500,
                                               bounds_sx_pix=(2, 30), bounds_sy_pix=(2, 30)):
    """
    Optimiza un filtro gaussiano ELÍPTICO CENTRADO en la posición del lóbulo.
    TRABAJA EN PÍXELES DE FRECUENCIA.
    
    Parámetros:
    - roi: diccionario de preparar_roi (con matriz, kx, ky en píxeles, col_central)
    - kx_rad, ky_rad: arrays 2D calibrados (rad/m) - SOLO PARA LOCALIZAR
    - k_min, k_max, ancho_kx: parámetros para localizar_lobulo_en_pixeles
    - bounds_sx_pix, bounds_sy_pix: rangos de búsqueda para sx y sy (en PÍXELES de frecuencia)
    
    Retorna:
    - pasos, paso_promedio, std_pasos, (sx_opt_pix, sy_opt_pix), imagen_filtrada, mascara_pix, peaks
    """
    print("Iniciando optimización de filtro elíptico centrado en lóbulo (en píxeles)")
    
    # 1. Localizar el lóbulo (obtenemos ÍNDICES)
    idx_ky0, idx_kx0 = localizar_lobulo_en_pixeles(roi, kx_rad, ky_rad, 
                                                    k_min, k_max, ancho_kx)
    
    # 2. Extraer datos del ROI
    matriz = roi["matriz"]
    col_central = roi["col_central"]
    
    # 3. Obtener coordenadas en píxeles de frecuencia
    kx_pix = roi["kx"]  # estos están en píxeles, centrados
    ky_pix = roi["ky"]
    
    # 4. Coordenadas relativas al centro del lóbulo (en PÍXELES)
    kx_rel_pix = kx_pix - kx_pix[0, idx_kx0]
    ky_rel_pix = ky_pix - ky_pix[idx_ky0, 0]
    
    # 5. Precalcular FFT
    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)
    
    # 6. Función de pérdida
    def loss(params):
        sx_pix, sy_pix = params
        if sx_pix <= 0.5 or sy_pix <= 0.5:  # Evitar valores demasiado pequeños
            return 1e6
        
        # Máscara gaussiana elíptica centrada en el lóbulo (en píxeles)
        mascara = np.exp(-(kx_rel_pix**2/(2*sx_pix**2) + ky_rel_pix**2/(2*sy_pix**2)))
        
        # Aplicar filtro
        imagen_filtrada = np.real(
            np.fft.ifft2(np.fft.ifftshift(fshift * mascara))
        )
        
        # Detectar picos en el perfil vertical central
        perfil = imagen_filtrada[:, col_central]
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
    
    # 7. Optimización con Differential Evolution
    result = differential_evolution(
        loss,
        bounds=[bounds_sx_pix, bounds_sy_pix],
        popsize=20,
        maxiter=40,
        tol=1e-3,
        workers=1,
        disp=False
    )
    
    sx_opt_pix, sy_opt_pix = result.x
    
    # 8. Filtrado final con parámetros óptimos
    mascara_opt_pix = np.exp(-(kx_rel_pix**2/(2*sx_opt_pix**2) + ky_rel_pix**2/(2*sy_opt_pix**2)))
    imagen_filtrada = np.real(
        np.fft.ifft2(np.fft.ifftshift(fshift * mascara_opt_pix))
    )
    
    # 9. Detección final de picos
    perfil = imagen_filtrada[:, col_central]
    peaks, _ = find_peaks(
        perfil,
        height=np.max(perfil)*0.03,
        distance=25,
        prominence=np.max(perfil)*0.03
    )
    
    if len(peaks) >= 2:
        pasos = np.diff(peaks.astype(float))
        paso_promedio = np.mean(pasos)
        std_pasos = np.std(pasos)
        return pasos, paso_promedio, std_pasos, (sx_opt_pix, sy_opt_pix), imagen_filtrada, mascara_opt_pix, peaks
    else:
        print("⚠️ No se detectaron suficientes picos después del filtrado")
        return None, None, None, (sx_opt_pix, sy_opt_pix), imagen_filtrada, mascara_opt_pix, peaks

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
    
def ajustar_filtro_radio_barrido_centrado_en_lobulo(roi, kx_rad, ky_rad,
                                                    k_min=1500, k_max=6000, ancho_kx=500,
                                                    radio_min_pix=2, radio_max_pix=30, n_radios=20):
    """
    Barrido de radios para filtro circular binario CENTRADO en la posición del lóbulo.
    TRABAJA EN PÍXELES DE FRECUENCIA.
    
    Parámetros:
    - roi: diccionario de preparar_roi (con matriz, kx, ky en píxeles, col_central)
    - kx_rad, ky_rad: arrays 2D calibrados (rad/m) - SOLO PARA LOCALIZAR
    - k_min, k_max, ancho_kx: parámetros para localizar_lobulo_en_pixeles
    - radio_min_pix, radio_max_pix: rango de radios a barrer (en PÍXELES de frecuencia)
    - n_radios: número de radios a probar
    
    Retorna:
    - pasos, paso_promedio, std_pasos, r_opt_pix, imagen_filtrada, mascara_pix, peaks
    """
    print("Iniciando barrido de radio centrado en lóbulo (en píxeles)")
    
    # 1. Localizar el lóbulo (usando la versión que devuelve índices)
    idx_ky0, idx_kx0 = localizar_lobulo_en_pixeles(roi, kx_rad, ky_rad, 
                                                    k_min, k_max, ancho_kx)
    
    # 2. Extraer datos del ROI
    matriz = roi["matriz"]
    col_central = roi["col_central"]
    
    # 3. Obtener coordenadas en píxeles de frecuencia
    kx_pix = roi["kx"]  # estos están en píxeles, centrados
    ky_pix = roi["ky"]
    
    # 4. Construir distancia al centro del lóbulo en PÍXELES
    distancia_al_lobulo_pix = np.sqrt((kx_pix - kx_pix[0, idx_kx0])**2 + 
                                       (ky_pix - ky_pix[idx_ky0, 0])**2)
    
    # 5. Precalcular FFT
    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)
    
    # 6. Barrido de radios (en píxeles)
    radios_pix = np.linspace(radio_min_pix, radio_max_pix, n_radios)
    stds = []
    radios_validos = []
    
    for r_pix in radios_pix:
        # Máscara binaria circular centrada en el lóbulo (en píxeles)
        mascara = distancia_al_lobulo_pix <= r_pix
        
        # Aplicar filtro
        imagen_filtrada = np.real(
            np.fft.ifft2(np.fft.ifftshift(fshift * mascara))
        )
        
        # Detectar picos
        perfil = imagen_filtrada[:, col_central]
        peaks, _ = find_peaks(
            perfil,
            height=np.max(perfil)*0.03,
            distance=25,
            prominence=np.max(perfil)*0.03
        )
        
        if len(peaks) >= 4:
            pasos = np.diff(peaks.astype(float))
            stds.append(np.std(pasos))
            radios_validos.append(r_pix)
    
    # 7. Verificar que se encontraron radios válidos
    if len(stds) == 0:
        print("⚠️ No se encontraron radios con suficientes picos")
        return None, None, None, None, None, None, None
    
    # 8. Seleccionar radio óptimo (menor desviación estándar)
    r_opt_pix = radios_validos[np.argmin(stds)]
    
    # 9. Filtrado final con radio óptimo
    mascara_opt_pix = distancia_al_lobulo_pix <= r_opt_pix
    imagen_filtrada = np.real(
        np.fft.ifft2(np.fft.ifftshift(fshift * mascara_opt_pix))
    )
    
    # 10. Detección final de picos
    perfil = imagen_filtrada[:, col_central]
    peaks, _ = find_peaks(
        perfil,
        height=np.max(perfil)*0.03,
        distance=25,
        prominence=np.max(perfil)*0.03
    )
    
    if len(peaks) >= 2:
        pasos = np.diff(peaks.astype(float))
        paso_promedio = np.mean(pasos)
        std_pasos = np.std(pasos)
        return pasos, paso_promedio, std_pasos, r_opt_pix, imagen_filtrada, mascara_opt_pix, peaks
    else:
        print("⚠️ No se detectaron suficientes picos después del filtrado final")
        return None, None, None, r_opt_pix, imagen_filtrada, mascara_opt_pix, peaks
    
def visualizar_resultado_filtrado(matriz_original, imagen_filtrada, mascara_pix, 
                                  fshift=None, kx_rad=None, ky_rad=None):
    """
    Visualiza con máscara en píxeles, pero ejes calibrados si se proporcionan.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Imagen original
    im0 = axes[0].imshow(matriz_original, cmap='gray', origin='upper')
    axes[0].set_title('Imagen Original')
    plt.colorbar(im0, ax=axes[0])
    
    # Imagen filtrada
    im1 = axes[1].imshow(imagen_filtrada, cmap='gray', origin='upper')
    axes[1].set_title('Imagen Filtrada')
    plt.colorbar(im1, ax=axes[1])
    
    # Espectro + máscara
    if fshift is not None:
        espectro_log = np.log10(1 + np.abs(fshift))
        
        if kx_rad is not None and ky_rad is not None:
            extent = [kx_rad[0, 0], kx_rad[0, -1], ky_rad[-1, 0], ky_rad[0, 0]]
            xlabel = '$k_x$ (rad/m)'
            ylabel = '$k_y$ (rad/m)'
        else:
            extent = None
            xlabel = 'píxeles de frecuencia'
            ylabel = 'píxeles de frecuencia'
        
        axes[2].imshow(espectro_log, cmap='gray', extent=extent, origin='upper', alpha=0.7)
        im2 = axes[2].imshow(mascara_pix, cmap='jet', extent=extent, origin='upper', alpha=0.5)
        axes[2].set_title('Máscara sobre espectro')
        axes[2].set_xlabel(xlabel)
        axes[2].set_ylabel(ylabel)
        plt.colorbar(im2, ax=axes[2])
    else:
        im2 = axes[2].imshow(mascara_pix, cmap='jet', origin='upper')
        axes[2].set_title('Máscara en Fourier')
        plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()