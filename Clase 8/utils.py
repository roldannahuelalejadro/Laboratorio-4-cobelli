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
                                                    px_por_mm=None,
                                                    err_px_por_mm=None,
                                                    k_min=1500, k_max=6000, ancho_kx=500,
                                                    radio_min_pix=2, radio_max_pix=30, n_radios=20,
                                                    plot_diagnostico=False):
    """
    Barrido de radios para filtro circular binario CENTRADO en la posición del lóbulo.
    TRABAJA EN PÍXELES DE FRECUENCIA. Incluye propagación de errores sistemáticos.
    
    Parámetros adicionales:
    - px_por_mm: píxeles por milímetro (para calibración)
    - err_px_por_mm: error en px_por_mm
    - plot_diagnostico: si True, grafica std vs radio
    
    Retorna:
    - pasos, paso_promedio, paso_std, error_total_px, error_rel, 
      r_opt_pix, imagen_filtrada, mascara_pix, peaks, (opcional) radios, stds
    """
    print("Iniciando barrido de radio binario centrado en lóbulo (en píxeles)")
    
    # Calcular error relativo de calibración
    if px_por_mm is not None and err_px_por_mm is not None:
        rel_err_cal = err_px_por_mm / px_por_mm
    else:
        rel_err_cal = 0
        print("⚠️ Advertencia: No se proporcionaron errores de calibración")
    
    # 1. Localizar el lóbulo
    idx_ky0, idx_kx0 = localizar_lobulo_en_pixeles(roi, kx_rad, ky_rad, 
                                                    k_min, k_max, ancho_kx)
    
    # 2. Extraer datos del ROI
    matriz = roi["matriz"]
    col_central = roi["col_central"]
    
    # 3. Obtener coordenadas en píxeles de frecuencia
    kx_pix = roi["kx"]
    ky_pix = roi["ky"]
    
    # 4. Distancia al centro del lóbulo en PÍXELES
    distancia_al_lobulo_pix = np.sqrt((kx_pix - kx_pix[0, idx_kx0])**2 + 
                                       (ky_pix - ky_pix[idx_ky0, 0])**2)
    
    # 5. Precalcular FFT
    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)
    
    # 6. Barrido de radios
    radios_pix = np.linspace(radio_min_pix, radio_max_pix, n_radios)
    stds = []
    radios_validos = []
    imagenes_por_radio = []  # Guardar para diagnóstico
    
    for r_pix in radios_pix:
        # Máscara binaria circular
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
            std_actual = np.std(pasos)
            stds.append(std_actual)
            radios_validos.append(r_pix)
            imagenes_por_radio.append((r_pix, std_actual, imagen_filtrada, mascara, peaks))
    
    # 7. Verificar resultados
    if len(stds) == 0:
        print("⚠️ No se encontraron radios con suficientes picos")
        return None, None, None, None, None, None, None, None, None
    
    # 8. Seleccionar radio óptimo
    idx_opt = np.argmin(stds)
    r_opt_pix = radios_validos[idx_opt]
    std_opt = stds[idx_opt]
    _, _, img_opt, mask_opt, peaks_opt = imagenes_por_radio[idx_opt]
    
    # 9. Estadísticas finales
    pasos_opt = np.diff(peaks_opt.astype(float))
    paso_promedio = np.mean(pasos_opt)
    paso_std = np.std(pasos_opt)
    
    # 10. Propagación de errores sistemáticos
    if px_por_mm is not None and err_px_por_mm is not None:
        err_sist_px = paso_promedio * rel_err_cal
        error_total_px = np.sqrt(paso_std**2 + err_sist_px**2)
        error_rel = error_total_px / paso_promedio if paso_promedio != 0 else 0
    else:
        error_total_px = paso_std
        error_rel = paso_std / paso_promedio if paso_promedio != 0 else 0
    
    print(f"📊 Paso promedio: {paso_promedio:.2f} ± {paso_std:.2f} px (estadístico)")
    if px_por_mm is not None:
        print(f"   Error sistemático calibración: {err_sist_px:.3f} px")
    print(f"   Error total: {error_total_px:.2f} px ({error_rel*100:.2f}%)")
    print(f"   Radio óptimo: {r_opt_pix:.1f} px (std={std_opt:.2f})")
    
    # 11. Diagnóstico opcional
    if plot_diagnostico:
        plt.figure(figsize=(10, 5))
        plt.plot(radios_validos, stds, 'o-', linewidth=2, markersize=8)
        plt.axvline(r_opt_pix, color='r', linestyle='--', 
                   label=f'Óptimo: {r_opt_pix:.1f} px')
        plt.xlabel('Radio (píxeles)')
        plt.ylabel('Desviación estándar de pasos (px)')
        plt.title('Optimización de radio - Filtro binario')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return (pasos_opt, paso_promedio, paso_std, error_total_px, error_rel,
                r_opt_pix, img_opt, mask_opt, peaks_opt, radios_validos, stds)
    
    return (pasos_opt, paso_promedio, paso_std, error_total_px, error_rel,
            r_opt_pix, img_opt, mask_opt, peaks_opt)

def visualizar_resultado_filtrado(matriz_original, imagen_filtrada, mascara_pix, 
                                  fshift=None, kx_rad=None, ky_rad=None,
                                  pixel_size_um=None, k_crop_rad_m=10000):
    """
    Visualiza con máscara en log, espectro recortado a bajas frecuencias,
    y escalas en mm para las imágenes espaciales.
    
    Parámetros nuevos:
    - pixel_size_um: tamaño de píxel en µm (para poner ejes en mm)
    - k_crop_rad_m: recorta el espectro a |kx|, |ky| ≤ este valor (default 10000 rad/m)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    # ====================== IMÁGENES ESPACIALES (con escala en mm) ======================
    pixel_size_mm = pixel_size_um / 1000 if pixel_size_um is not None else None
    
    if pixel_size_mm is not None:
        h, w = matriz_original.shape
        extent_spatial = [
            -w/2 * pixel_size_mm,
             w/2 * pixel_size_mm,
            -h/2 * pixel_size_mm,
             h/2 * pixel_size_mm
        ]
        xlabel_sp = 'x (mm)'
        ylabel_sp = 'y (mm)'
    else:
        extent_spatial = None
        xlabel_sp = 'píxeles'
        ylabel_sp = 'píxeles'
    
    # Imagen original
    im0 = axes[0].imshow(matriz_original, cmap='gray', origin='upper', extent=extent_spatial)
    axes[0].set_title('Imagen Original')
    axes[0].set_xlabel(xlabel_sp)
    axes[0].set_ylabel(ylabel_sp)
    plt.colorbar(im0, ax=axes[0])
    
    # Imagen filtrada
    im1 = axes[1].imshow(imagen_filtrada, cmap='gray', origin='upper', extent=extent_spatial)
    axes[1].set_title('Imagen Filtrada')
    axes[1].set_xlabel(xlabel_sp)
    axes[1].set_ylabel(ylabel_sp)
    plt.colorbar(im1, ax=axes[1])
    
    # ====================== ESPECTRO RECORTADO + MÁSCARA ======================
    if fshift is not None:
        espectro_log = np.log10(1 + np.abs(fshift))
        
        # Recorte a bajas frecuencias
        if kx_rad is not None and ky_rad is not None:
            # Valores únicos de k
            ky_vals = ky_rad[:, 0]
            kx_vals = kx_rad[0, :]
            
            idx_ky = np.where(np.abs(ky_vals) <= k_crop_rad_m)[0]
            idx_kx = np.where(np.abs(kx_vals) <= k_crop_rad_m)[0]
            
            if len(idx_ky) > 20 and len(idx_kx) > 20:   # evita recortes demasiado pequeños
                slice_y = slice(idx_ky[0], idx_ky[-1] + 1)
                slice_x = slice(idx_kx[0], idx_kx[-1] + 1)
                
                espectro_log_crop = espectro_log[slice_y, slice_x]
                mascara_crop      = mascara_pix[slice_y, slice_x]
                
                new_extent = [
                    kx_vals[idx_kx[0]],
                    kx_vals[idx_kx[-1]],
                    ky_vals[idx_ky[-1]],   # ky más negativo abajo
                    ky_vals[idx_ky[0]]     # ky más positivo arriba
                ]
            else:
                espectro_log_crop = espectro_log
                mascara_crop      = mascara_pix
                new_extent = [kx_rad[0,0], kx_rad[0,-1], ky_rad[-1,0], ky_rad[0,0]]
        else:
            # Sin calibración → recortamos central en píxeles (≈ ±300 px)
            cy, cx = espectro_log.shape[0]//2, espectro_log.shape[1]//2
            r = 300
            slice_y = slice(cy - r, cy + r)
            slice_x = slice(cx - r, cx + r)
            espectro_log_crop = espectro_log[slice_y, slice_x]
            mascara_crop      = mascara_pix[slice_y, slice_x]
            new_extent = None
        
        # Plot espectro base
        axes[2].imshow(espectro_log_crop, cmap='gray', extent=new_extent, origin='upper', alpha=0.75)
        
        # Máscara en log
        mascara_log = np.log10(np.maximum(mascara_crop, 1e-10))
        im2 = axes[2].imshow(mascara_log, cmap='jet', extent=new_extent, origin='upper', alpha=0.55)
        
        axes[2].set_title(f'Espectro recortado (|k| ≤ {k_crop_rad_m} rad/m) + Máscara (log)')
        axes[2].set_xlabel('$k_x$ (rad/m)' if kx_rad is not None else 'píxeles de frecuencia')
        axes[2].set_ylabel('$k_y$ (rad/m)' if ky_rad is not None else 'píxeles de frecuencia')
        plt.colorbar(im2, ax=axes[2], label='log₁₀(Máscara)')
        
    else:
        # Caso sin FFT
        mascara_log = np.log10(np.maximum(mascara_pix, 1e-10))
        im2 = axes[2].imshow(mascara_log, cmap='jet', origin='upper')
        axes[2].set_title('Máscara en Fourier (log)')
        plt.colorbar(im2, ax=axes[2], label='log₁₀(Máscara)')
    
    plt.tight_layout()
    return fig

def calcular_ancho_desde_espectro(
    imagen,
    lambda_m,                # longitud de onda en metros (obligatorio)
    pixel_size_m,            # tamaño de píxel en metros (obligatorio)
    D_m,                     # distancia rendija-pantalla en metros (obligatorio)
    center_x=890,
    center_y=1645,
    offset=650,
    canal=2,
    plot_espectro=False,
    umbral_lobulo=0.5,
    delta_kx=500,            # semi-ancho de la ventana en kx (rad/m)
    delta_ky=1000,           # semi-ancho de la ventana en ky (rad/m)
    k_y_min_inicial=1000,    # límite inferior para buscar el pico inicial (rad/m)
    err_lambda_m=None,       # error en lambda_m (opcional)
    err_D_m=None,            # error en D_m (opcional)
    err_pixel_size_m=None    # error en pixel_size_m (opcional)
):
    """
    Calcula el ancho de rendija a partir del espectro de Fourier, expandiendo ventana desde el máximo.
    
    Parámetros obligatorios:
    - lambda_m: longitud de onda en metros
    - pixel_size_m: tamaño de píxel en metros
    - D_m: distancia rendija-pantalla en metros
    
    Parámetros opcionales de recorte:
    - center_x, center_y, offset, canal: para extraer la ROI
    
    Parámetros de control del algoritmo:
    - umbral_lobulo: fracción del máximo de la ventana para seleccionar píxeles del lóbulo (ej. 0.5)
    - delta_kx: semi-ancho en kx alrededor del pico para la ventana (rad/m)
    - delta_ky: semi-ancho en ky alrededor del pico para la ventana (rad/m)
    - k_y_min_inicial: valor mínimo de ky para buscar el pico inicial (evita DC y ruido de baja frecuencia)
    
    Parámetros de error:
    - err_lambda_m, err_D_m, err_pixel_size_m: errores de las constantes (opcionales)
    
    Retorna:
    - k_promedio (rad/m), delta_y (m), a_um (µm), err_a_um (µm)
    """
    
    # --- Extracción de ROI ---
    matriz = imagen[center_x - offset : center_x + offset,
                    center_y - offset : center_y + offset, canal].astype(float)
    matriz_detrend = matriz - np.mean(matriz)
    
    # --- FFT ---
    f = np.fft.fft2(matriz_detrend)
    fshift = np.fft.fftshift(f)
    espectro_abs = np.abs(fshift)
    
    # --- Calibración de ejes en k (rad/m) ---
    Ny, Nx = matriz.shape
    dx = pixel_size_m
    kx = np.fft.fftshift(np.fft.fftfreq(Nx, dx)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(Ny, dx)) * 2 * np.pi
    
    # --- Búsqueda del píxel de máxima intensidad en la región inicial ---
    # Limitamos a ky > k_y_min_inicial (positivo) y kx cercano a 0 (tomamos todo el rango de kx, pero después refinamos)
    idx_ky_pos_inicial = np.where(ky > k_y_min_inicial)[0]
    if len(idx_ky_pos_inicial) == 0:
        print("No hay píxeles en la región inicial de ky.")
        return None, None, None, None
    
    # Subespectro en esa región (todo kx)
    sub_espectro_inicial = espectro_abs[idx_ky_pos_inicial, :]
    # Encontrar el máximo global en esa región
    idx_max_ky_rel, idx_max_kx = np.unravel_index(np.argmax(sub_espectro_inicial), sub_espectro_inicial.shape)
    idx_max_ky = idx_ky_pos_inicial[idx_max_ky_rel]
    
    k_max_x = kx[idx_max_kx]
    k_max_y = ky[idx_max_ky]
    
    # --- Definir ventana alrededor del máximo ---
    idx_kx_ventana = np.where((kx > k_max_x - delta_kx/2) & (kx < k_max_x + delta_kx/2))[0]
    idx_ky_ventana = np.where((ky > k_max_y - delta_ky/2) & (ky < k_max_y + delta_ky/2))[0]
    
    if len(idx_kx_ventana) == 0 or len(idx_ky_ventana) == 0:
        print("La ventana definida no contiene píxeles.")
        return None, None, None, None
    
    # --- Subespectro en la ventana 2D ---
    # Usamos np.ix_ para extraer la submatriz
    sub_espectro = espectro_abs[np.ix_(idx_ky_ventana, idx_kx_ventana)]
    amp_max_ventana = np.max(sub_espectro)
    
    # --- Seleccionar píxeles por encima del umbral ---
    # Obtenemos índices relativos a la ventana
    idx_ky_rel, idx_kx_rel = np.where(sub_espectro > umbral_lobulo * amp_max_ventana)
    if len(idx_ky_rel) < 3:
        print("Muy pocos píxeles sobre el umbral en la ventana.")
        return None, None, None, None
    
    # Convertir a índices globales en ky, kx
    idx_ky_global = idx_ky_ventana[idx_ky_rel]
    idx_kx_global = idx_kx_ventana[idx_kx_rel]
    
    # Extraer valores de ky y amplitudes
    k_lobulo = ky[idx_ky_global]
    amps_lobulo = sub_espectro[idx_ky_rel, idx_kx_rel]  # amplitudes en esos píxeles
    
    # --- Promedio ponderado de ky ---
    w = amps_lobulo
    k_promedio = np.sum(w * k_lobulo) / np.sum(w)
    
    # Varianza de la media ponderada (fórmula correcta)
    var_k = np.sum(w * (k_lobulo - k_promedio)**2) / (np.sum(w)**2)
    err_k = np.sqrt(var_k)
    
    # --- Magnitudes derivadas ---
    delta_y = 2 * np.pi / k_promedio          # espaciado entre franjas (m)
    err_delta_y = (2 * np.pi / k_promedio**2) * err_k   # propagación
    
    a_m = lambda_m * D_m / delta_y            # ancho de rendija (m)
    a_um = a_m * 1e6                          # µm
    
    # --- Propagación de errores (solo estadístico por ahora) ---
    err_a_m_stat = a_m * (err_delta_y / delta_y)
    err_a_um_stat = err_a_m_stat * 1e6
    
    # --- Errores sistemáticos si se proporcionan ---
    err_a_m_sist = 0.0
    if err_lambda_m is not None:
        err_a_m_sist += ((D_m / delta_y) * err_lambda_m)**2
    if err_D_m is not None:
        err_a_m_sist += ((lambda_m / delta_y) * err_D_m)**2
    if err_pixel_size_m is not None:
        rel_err_pix = err_pixel_size_m / pixel_size_m
        err_a_m_sist += (a_m * rel_err_pix)**2
    
    err_a_m_sist = np.sqrt(err_a_m_sist)
    
    # Error total en cuadratura
    err_a_m_total = np.sqrt(err_a_m_stat**2 + err_a_m_sist**2)
    err_a_um_total = err_a_m_total * 1e6
    
    # --- Visualización opcional ---
    if plot_espectro:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Espectro completo en escala log
        espectro_log = np.log10(1 + espectro_abs)
        axs[0].imshow(espectro_log, extent=[kx[0], kx[-1], ky[0], ky[-1]],
                      origin='lower', cmap='viridis', aspect='auto')
        axs[0].set_xlabel('$k_x$ (rad/m)')
        axs[0].set_ylabel('$k_y$ (rad/m)')
        axs[0].set_title('Espectro completo (log)')
        axs[0].plot(k_max_x, k_max_y, 'go', ms=8, label='Máximo inicial')
        axs[0].legend()
        
        # Ventana utilizada (subespectro)
        extent_ventana = [kx[idx_kx_ventana[0]], kx[idx_kx_ventana[-1]],
                          ky[idx_ky_ventana[0]], ky[idx_ky_ventana[-1]]]
        axs[1].imshow(np.log10(1 + sub_espectro), extent=extent_ventana,
                      origin='lower', cmap='viridis', aspect='auto')
        axs[1].set_xlabel('$k_x$ (rad/m)')
        axs[1].set_ylabel('$k_y$ (rad/m)')
        axs[1].set_title(f'Ventana {delta_kx}×{delta_ky} rad/m (log)')
        axs[1].axhline(k_promedio, color='r', ls='--', label=f'$\\bar{{k}}_y$ = {k_promedio:.0f} rad/m')
        axs[1].legend()
        
        # Píxeles seleccionados
        axs[2].scatter(kx[idx_kx_global], ky[idx_ky_global], c=amps_lobulo, cmap='hot', s=10)
        axs[2].set_xlabel('$k_x$ (rad/m)')
        axs[2].set_ylabel('$k_y$ (rad/m)')
        axs[2].set_title(f'{len(idx_ky_global)} píxeles sobre umbral ({umbral_lobulo*100:.0f}%)')
        axs[2].axhline(k_promedio, color='g', ls='-', label=f'$\\bar{{k}}_y$ = {k_promedio:.0f} rad/m')
        axs[2].legend()
        
        plt.tight_layout()
        plt.show()
    
    return k_promedio, delta_y, a_um, err_a_um_total

def calcular_ancho_desde_espectro_adaptativo(imagen,                               
                                             center_x=890, center_y=1645, offset=650,
                                             canal=2, 
                                             plot_espectro=False,
                                             umbral_lobulo=0.5,
                                             factor_ancho=2.0,
                                             search_kx_max=5000,
                                             search_ky_min=1000, search_ky_max=8000,
                                             pixel_size_m=None,
                                             lambda_m=None,
                                             D_m=None,
                                             err_pixel_size_m=None,
                                             err_lambda_m=None,
                                             err_D_m=None):
    """
    Calcula el ancho de rendija a partir del espectro de Fourier de la imagen.
    Versión adaptativa: localiza el pico máximo en una región de búsqueda y
    define una ventana alrededor de él basada en la extensión del lóbulo.
    
    Parámetros:
    - imagen: imagen completa
    - center_x, center_y, offset, canal: para extraer ROI
    - plot_espectro: si True, muestra gráficos
    - umbral_lobulo: fracción del máximo para considerar píxeles del lóbulo (ej. 0.5)
    - factor_ancho: factor para expandir la ventana (ej. 2.0 significa ventana de ancho = factor * ancho_del_lóbulo)
    - search_kx_max: semi-ancho en kx para la búsqueda inicial (rad/m)
    - search_ky_min, search_ky_max: rango en ky para búsqueda inicial (rad/m)
    - pixel_size_m, lambda_m, D_m: constantes físicas (deben pasarse explícitamente)
    - err_*: errores de las constantes para propagación
    
    Retorna:
    - k_promedio, delta_y, a_um, err_a_um
    """
    # Verificar que se pasaron las constantes necesarias
    if None in (pixel_size_m, lambda_m, D_m):
        raise ValueError("Debe proporcionar pixel_size_m, lambda_m y D_m")
    
    # Extraer ROI
    matriz = imagen[center_x - offset:center_x + offset,
                    center_y - offset:center_y + offset,
                    canal].astype(float)
    matriz_detrend = matriz - np.mean(matriz)
    
    # FFT
    fshift = np.fft.fftshift(np.fft.fft2(matriz_detrend))
    espectro_abs = np.abs(fshift)
    espectro_log = np.log10(1 + espectro_abs)
    
    Ny, Nx = matriz.shape
    dx = pixel_size_m
    # Vectores de frecuencia angular (rad/m)
    kx = np.fft.fftshift(np.fft.fftfreq(Nx, dx)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(Ny, dx)) * 2 * np.pi
    
    # Crear mallas 2D
    KX, KY = np.meshgrid(kx, ky)
    
    # ---- Búsqueda inicial del pico ----
    # Excluir la región central (bajas frecuencias) para evitar DC
    mask_search = (np.abs(KX) < search_kx_max) & (KY > search_ky_min) & (KY < search_ky_max)
    espectro_enmascarado = np.where(mask_search, espectro_log, -np.inf)
    
    # Encontrar el máximo
    idx_max = np.unravel_index(np.argmax(espectro_enmascarado), espectro_enmascarado.shape)
    kx0 = KX[idx_max]
    ky0 = KY[idx_max]
    print(f"Pico encontrado en: kx0 = {kx0:.1f} rad/m, ky0 = {ky0:.1f} rad/m")
    
    # ---- Determinar la extensión del lóbulo ----
    # Crear una región alrededor del pico para analizar la forma
    radio_busqueda = 500  # rad/m, puede ajustarse
    mask_lobulo = (np.abs(KX - kx0) < radio_busqueda) & (np.abs(KY - ky0) < radio_busqueda)
    
    # Extraer los valores de amplitud en esa región
    amps_region = espectro_abs[mask_lobulo]
    kx_region = KX[mask_lobulo]
    ky_region = KY[mask_lobulo]
    
    # Normalizar amplitudes para tener una distribución de probabilidad
    amps_norm = amps_region / np.sum(amps_region)
    
    # Calcular centroide (media ponderada) y desviaciones estándar
    kx_mean = np.sum(amps_norm * kx_region)
    ky_mean = np.sum(amps_norm * ky_region)
    
    var_kx = np.sum(amps_norm * (kx_region - kx_mean)**2)
    var_ky = np.sum(amps_norm * (ky_region - ky_mean)**2)
    std_kx = np.sqrt(var_kx)
    std_ky = np.sqrt(var_ky)
    
    print(f"Desviación estándar del lóbulo: σ_kx = {std_kx:.1f} rad/m, σ_ky = {std_ky:.1f} rad/m")
    
    # Definir la ventana como un múltiplo de las desviaciones estándar
    ancho_ventana_kx = factor_ancho * std_kx
    ancho_ventana_ky = factor_ancho * std_ky
    
    # ---- Selección de píxeles dentro de la ventana y sobre el umbral ----
    mask_ventana = (np.abs(KX - kx0) < ancho_ventana_kx) & (np.abs(KY - ky0) < ancho_ventana_ky)
    
    # Obtener índices de la ventana
    indices_ventana = np.where(mask_ventana)  # (filas, columnas)
    amps_ventana = espectro_abs[indices_ventana]
    
    if len(amps_ventana) == 0:
        print("No hay píxeles en la ventana")
        return None, None, None, None
    
    # Umbral basado en el máximo de la ventana
    umbral = umbral_lobulo * np.max(amps_ventana)
    
    # Seleccionar los índices que superan el umbral
    mask_umbral = amps_ventana > umbral
    indices_lobulo = (indices_ventana[0][mask_umbral], indices_ventana[1][mask_umbral])
    
    # Extraer coordenadas y amplitudes de esos píxeles
    ky_lobulo = KY[indices_lobulo]
    kx_lobulo = KX[indices_lobulo]   # opcional, no se usa pero se guarda por si acaso
    amps_lobulo = espectro_abs[indices_lobulo]
    
    if len(ky_lobulo) < 3:
        print("Pocos píxeles sobre el umbral")
        return None, None, None, None
    
    # ---- CÁLCULO ESTADÍSTICO CORREGIDO ----
    # Promedio ponderado
    w = amps_lobulo
    k_promedio = np.sum(w * ky_lobulo) / np.sum(w)
    
    # 1. Varianza ponderada de la muestra
    var_k_muestra = np.sum(w * (ky_lobulo - k_promedio)**2) / np.sum(w)
    dev_k_muestra = np.sqrt(var_k_muestra)
    
    # 2. Tamaño de muestra efectivo
    n_eff = (np.sum(w)**2) / np.sum(w**2)
    
    # 3. Error estándar de la media ponderada (estadístico)
    err_k_stat = dev_k_muestra / np.sqrt(n_eff)
    
    # 4. Error sistemático de calibración
    if err_pixel_size_m is not None:
        err_k_sist = k_promedio * (err_pixel_size_m / pixel_size_m)
    else:
        err_k_sist = 0
    
    # 5. Error total en k
    err_k_total = np.sqrt(err_k_stat**2 + err_k_sist**2)
    
    # 6. Propagar a delta_y y a
    delta_y = 2 * np.pi / k_promedio
    err_delta_y = (2 * np.pi / k_promedio**2) * err_k_total
    
    a_m = lambda_m * D_m / delta_y
    err_a_m = a_m * (err_delta_y / delta_y)
    
    # 7. Agregar otros errores sistemáticos si se proporcionan
    if err_lambda_m is not None:
        err_a_m = np.sqrt(err_a_m**2 + (a_m * err_lambda_m / lambda_m)**2)
    if err_D_m is not None:
        err_a_m = np.sqrt(err_a_m**2 + (a_m * err_D_m / D_m)**2)
    
    # Convertir a micrómetros para salida
    a_um = a_m * 1e6
    err_a_um = err_a_m * 1e6
    
    # ---- VISUALIZACIÓN OPCIONAL ----
    if plot_espectro:
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        
        # Espectro completo
        axs[0].imshow(espectro_log, extent=[kx[0], kx[-1], ky[0], ky[-1]],
                      origin='lower', cmap='viridis', aspect='auto')
        axs[0].set_xlabel('kₓ (rad/m)')
        axs[0].set_ylabel('kᵧ (rad/m)')
        axs[0].set_title('Espectro completo')
        axs[0].plot(kx0, ky0, 'ro', ms=8, label='Pico encontrado')
        axs[0].legend()
        
        # Región de la ventana adaptativa
        extent_ventana = [kx0 - ancho_ventana_kx, kx0 + ancho_ventana_kx, 
                         ky0 - ancho_ventana_ky, ky0 + ancho_ventana_ky]
        axs[1].imshow(espectro_log, extent=[kx[0], kx[-1], ky[0], ky[-1]],
                      origin='lower', cmap='viridis', aspect='auto')
        axs[1].set_xlim(extent_ventana[0], extent_ventana[1])
        axs[1].set_ylim(extent_ventana[2], extent_ventana[3])
        axs[1].set_xlabel('kₓ (rad/m)')
        axs[1].set_ylabel('kᵧ (rad/m)')
        axs[1].set_title(f'Ventana adaptativa (factor={factor_ancho})')
        axs[1].axhline(ky0, color='r', ls='--', alpha=0.5)
        axs[1].axvline(kx0, color='r', ls='--', alpha=0.5)
        
        
        plt.tight_layout()
        plt.show()

    return k_promedio, delta_y, a_um, err_a_um

def calcular_k_desde_espectro_adaptativo(imagen, 
                                         center_x=890, center_y=1645, offset=650,
                                         canal=2, 
                                         plot_espectro=False,
                                         umbral_lobulo=0.5,
                                         delta_kx=500,
                                         delta_ky=1000,
                                         k_y_min_inicial=1000,
                                         pixel_size_m=None,
                                         err_pixel_size_m=None):
    """
    Calcula el número de onda promedio k_y a partir del espectro de Fourier.
    Versión con estadística ponderada correcta y propagación de errores sistemáticos.
    
    Parámetros:
    - imagen: imagen completa
    - center_x, center_y, offset, canal: para extraer ROI
    - plot_espectro: si True, muestra gráficos
    - umbral_lobulo: fracción del máximo de la ventana para seleccionar píxeles (ej. 0.5)
    - delta_kx: semi-ancho en kx alrededor del pico (rad/m)
    - delta_ky: semi-ancho en ky alrededor del pico (rad/m)
    - k_y_min_inicial: valor mínimo de ky para buscar el pico inicial (evita DC)
    - pixel_size_m: tamaño de píxel en metros (OBLIGATORIO)
    - err_pixel_size_m: error en pixel_size_m (opcional, si no se provee solo error estadístico)
    
    Retorna:
    - k_promedio (rad/m), err_k_total (rad/m), None, None
    """
    # ===== VERIFICACIONES =====
    if pixel_size_m is None:
        raise ValueError("Debe proporcionar pixel_size_m")
    
    # ===== EXTRACCIÓN DE ROI =====
    matriz = imagen[center_x - offset:center_x + offset,
                    center_y - offset:center_y + offset,
                    canal].astype(float)
    matriz_detrend = matriz - np.mean(matriz)
    
    # ===== FFT =====
    fshift = np.fft.fftshift(np.fft.fft2(matriz_detrend))
    espectro_abs = np.abs(fshift)
    
    # ===== EJES DE FRECUENCIA (rad/m) =====
    Ny, Nx = matriz.shape
    dx = pixel_size_m
    kx = np.fft.fftshift(np.fft.fftfreq(Nx, dx)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(Ny, dx)) * 2 * np.pi
    
    # ===== BÚSQUEDA DEL PICO MÁXIMO EN ky > k_y_min_inicial =====
    idx_ky_pos_inicial = np.where(ky > k_y_min_inicial)[0]
    if len(idx_ky_pos_inicial) == 0:
        print("No hay píxeles en la región inicial de ky.")
        return None, None, None, None
    
    sub_espectro_inicial = espectro_abs[idx_ky_pos_inicial, :]
    idx_max_ky_rel, idx_max_kx = np.unravel_index(np.argmax(sub_espectro_inicial), sub_espectro_inicial.shape)
    idx_max_ky = idx_ky_pos_inicial[idx_max_ky_rel]
    
    k_max_x = kx[idx_max_kx]
    k_max_y = ky[idx_max_ky]
    
    # ===== VENTANA ALREDEDOR DEL MÁXIMO =====
    idx_kx_ventana = np.where((kx > k_max_x - delta_kx/2) & (kx < k_max_x + delta_kx/2))[0]
    idx_ky_ventana = np.where((ky > k_max_y - delta_ky/2) & (ky < k_max_y + delta_ky/2))[0]
    
    if len(idx_kx_ventana) == 0 or len(idx_ky_ventana) == 0:
        print("La ventana definida no contiene píxeles.")
        return None, None, None, None
    
    # ===== SUBESPECTRO Y SELECCIÓN DE PÍXELES SOBRE UMBRAL =====
    sub_espectro = espectro_abs[np.ix_(idx_ky_ventana, idx_kx_ventana)]
    amp_max_ventana = np.max(sub_espectro)
    
    idx_ky_rel, idx_kx_rel = np.where(sub_espectro > umbral_lobulo * amp_max_ventana)
    if len(idx_ky_rel) < 3:
        print("Muy pocos píxeles sobre el umbral en la ventana.")
        return None, None, None, None
    
    # Índices globales
    idx_ky_global = idx_ky_ventana[idx_ky_rel]
    idx_kx_global = idx_kx_ventana[idx_kx_rel]
    
    # Valores de ky y amplitudes
    k_lobulo = ky[idx_ky_global]
    amps_lobulo = sub_espectro[idx_ky_rel, idx_kx_rel]
    
    # ===== ESTADÍSTICA PONDERADA CORRECTA =====
    w = amps_lobulo
    k_promedio = np.sum(w * k_lobulo) / np.sum(w)
    
    # Varianza ponderada de la muestra (CORREGIDA: divido por Σw, no por (Σw)²)
    var_k_muestra = np.sum(w * (k_lobulo - k_promedio)**2) / np.sum(w)
    dev_k_muestra = np.sqrt(var_k_muestra)
    
    # Tamaño de muestra efectivo (para datos ponderados)
    n_eff = (np.sum(w)**2) / np.sum(w**2)
    
    # Error estándar de la media ponderada (estadístico)
    err_k_stat = dev_k_muestra / np.sqrt(n_eff)
    
    # ===== ERROR SISTEMÁTICO DE CALIBRACIÓN =====
    if err_pixel_size_m is not None:
        # Error relativo de calibración
        rel_err_escala = err_pixel_size_m / pixel_size_m
        # El error en k es proporcional a k (mismo error relativo)
        err_k_sist = abs(k_promedio) * rel_err_escala
    else:
        err_k_sist = 0
        print("Advertencia: No se proporcionó err_pixel_size_m. Error sistemático = 0")
    
    # ===== ERROR TOTAL =====
    err_k_total = np.sqrt(err_k_stat**2 + err_k_sist**2)
    
    # ===== VISUALIZACIÓN OPCIONAL =====
    if plot_espectro:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        espectro_log = np.log10(1 + espectro_abs)
        
        # Espectro completo
        axs[0].imshow(espectro_log, extent=[kx[0], kx[-1], ky[0], ky[-1]],
                      origin='lower', cmap='viridis', aspect='auto')
        axs[0].set_xlabel('$k_x$ (rad/m)')
        axs[0].set_ylabel('$k_y$ (rad/m)')
        axs[0].set_title('Espectro completo')
        axs[0].plot(k_max_x, k_max_y, 'go', ms=8, label='Máximo inicial')
        axs[0].legend()
        
        # Ventana utilizada
        extent_ventana = [kx[idx_kx_ventana[0]], kx[idx_kx_ventana[-1]],
                          ky[idx_ky_ventana[0]], ky[idx_ky_ventana[-1]]]
        axs[1].imshow(np.log10(1 + sub_espectro), extent=extent_ventana,
                      origin='lower', cmap='viridis', aspect='auto')
        axs[1].set_xlabel('$k_x$ (rad/m)')
        axs[1].set_ylabel('$k_y$ (rad/m)')
        axs[1].set_title(f'Ventana {delta_kx}×{delta_ky} rad/m')
        axs[1].axhline(k_promedio, color='r', ls='--', 
                       label=f'$\\bar{{k}}_y$ = {k_promedio:.0f} ± {err_k_total:.0f} rad/m')
        axs[1].legend()
        
        # Píxeles seleccionados
        scatter = axs[2].scatter(kx[idx_kx_global], ky[idx_ky_global], 
                                 c=amps_lobulo, cmap='hot', s=10)
        axs[2].set_xlabel('$k_x$ (rad/m)')
        axs[2].set_ylabel('$k_y$ (rad/m)')
        axs[2].set_title(f'{len(idx_ky_global)} píxeles (umbral={umbral_lobulo*100:.0f}%)')
        axs[2].axhline(k_promedio, color='g', ls='-', 
                       label=f'$\\bar{{k}}_y$ = {k_promedio:.0f} rad/m')
        axs[2].legend()
        plt.colorbar(scatter, ax=axs[2], label='Amplitud')
        
        # Información adicional
        info_text = f'N_efectivo = {n_eff:.1f}\nσ_stat = {err_k_stat:.1f} rad/m\nσ_sist = {err_k_sist:.1f} rad/m'
        axs[2].text(0.02, 0.98, info_text, transform=axs[2].transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    return k_promedio, err_k_total, None, None

def calcular_error_paso_total(paso_promedio_px, paso_std_px, px_por_mm, err_px_por_mm):
    """
    Calcula el error total del paso combinando estadístico y sistemático.
    
    Parámetros:
    - paso_promedio_px: paso promedio en píxeles
    - paso_std_px: desviación estándar de los pasos (error estadístico)
    - px_por_mm: píxeles por milímetro (calibración)
    - err_px_por_mm: error en px_por_mm
    
    Retorna:
    - error_total_px: error combinado en píxeles
    - error_relativo: error relativo total
    """
    # Error sistemático de calibración (proporcional al valor)
    rel_err_cal = err_px_por_mm / px_por_mm
    err_sist_px = paso_promedio_px * rel_err_cal
    
    # Error total (estadístico + sistemático en cuadratura)
    error_total_px = np.sqrt(paso_std_px**2 + err_sist_px**2)
    error_relativo = error_total_px / paso_promedio_px if paso_promedio_px != 0 else 0
    
    return error_total_px, error_relativo

def ajustar_filtro_hamming_radio_barrido_centrado_en_lobulo(roi, kx_rad, ky_rad,
                                                            px_por_mm=None,
                                                            err_px_por_mm=None,
                                                            k_min=1500, k_max=6000, ancho_kx=500,
                                                            radio_min_pix=2, radio_max_pix=30, n_radios=20):
    """
    Barrido de radios para filtro Hamming circular CENTRADO en la posición del lóbulo.
    TRABAJA EN PÍXELES DE FRECUENCIA. Incluye propagación de errores de calibración.
    
    Parámetros adicionales:
    - px_por_mm: píxeles por milímetro (para calibración)
    - err_px_por_mm: error en px_por_mm
    
    Retorna:
    - pasos, paso_promedio, std_pasos, error_total_px, error_rel, r_opt_pix, imagen_filtrada, mascara_pix, peaks
    """
    print("Iniciando barrido de radio con filtro Hamming centrado en lóbulo (en píxeles)")
    
    # Verificar que se proporcionaron datos de calibración
    if px_por_mm is None or err_px_por_mm is None:
        print("⚠️ Advertencia: No se proporcionaron errores de calibración. Se devolverá solo error estadístico.")
        rel_err_cal = 0
    else:
        rel_err_cal = err_px_por_mm / px_por_mm
    
    # 1. Localizar el lóbulo
    idx_ky0, idx_kx0 = localizar_lobulo_en_pixeles(roi, kx_rad, ky_rad, 
                                                    k_min, k_max, ancho_kx)
    
    # 2. Extraer datos del ROI
    matriz = roi["matriz"]
    col_central = roi["col_central"]
    
    # 3. Obtener coordenadas en píxeles de frecuencia
    kx_pix = roi["kx"]
    ky_pix = roi["ky"]
    
    # 4. Distancia al centro del lóbulo en PÍXELES
    distancia_al_lobulo_pix = np.sqrt((kx_pix - kx_pix[0, idx_kx0])**2 + 
                                      (ky_pix - ky_pix[idx_ky0, 0])**2)
    
    # 5. Precalcular FFT
    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)
    
    # 6. Barrido de radios
    radios_pix = np.linspace(radio_min_pix, radio_max_pix, n_radios)
    resultados = []  # Guardará (r, std_pasos, img, mask, peaks)
    
    for r_pix in radios_pix:
        # Máscara Hamming circular
        mascara = np.zeros_like(distancia_al_lobulo_pix)
        idx = distancia_al_lobulo_pix <= r_pix
        if np.any(idx):
            # Fórmula corregida: coseno desde 0 hasta π
            phi = np.pi * distancia_al_lobulo_pix[idx] / r_pix
            mascara[idx] = 0.54 - 0.46 * np.cos(phi)
        
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
            std_actual = np.std(pasos)
            resultados.append((r_pix, std_actual, imagen_filtrada, mascara, peaks))
    
    # 7. Verificar resultados
    if len(resultados) == 0:
        print("⚠️ No se encontraron radios con suficientes picos")
        return None, None, None, None, None, None, None, None
    
    # 8. Seleccionar radio óptimo (menor desviación estándar)
    r_opt_pix, std_opt, img_opt, mask_opt, peaks_opt = min(resultados, key=lambda x: x[1])
    
    # 9. Calcular pasos y estadísticas
    pasos_opt = np.diff(peaks_opt.astype(float))
    paso_promedio = np.mean(pasos_opt)
    paso_std = np.std(pasos_opt)
    
    # 10. PROPAGACIÓN DE ERRORES
    if px_por_mm is not None and err_px_por_mm is not None:
        # Error sistemático de calibración
        err_sist_px = paso_promedio * rel_err_cal
        
        # Error total (estadístico + sistemático)
        error_total_px = np.sqrt(paso_std**2 + err_sist_px**2)
        error_rel = error_total_px / paso_promedio if paso_promedio != 0 else 0
        
        print(f"📊 Paso promedio: {paso_promedio:.2f} ± {paso_std:.2f} px (estadístico)")
        print(f"   Error sistemático calibración: {err_sist_px:.3f} px")
        print(f"   Error total: {error_total_px:.2f} px ({error_rel*100:.2f}%)")
    else:
        error_total_px = paso_std
        error_rel = paso_std / paso_promedio if paso_promedio != 0 else 0
        print(f"📊 Paso promedio: {paso_promedio:.2f} ± {paso_std:.2f} px (solo estadístico)")
    
    return (pasos_opt, paso_promedio, paso_std, error_total_px, error_rel, r_opt_pix, img_opt, mask_opt, peaks_opt)
    
def ajustar_filtro_tukey_radio_barrido_centrado_en_lobulo(roi, kx_rad, ky_rad,
                                                          px_por_mm=None,
                                                          err_px_por_mm=None,
                                                          k_min=1500, k_max=6000, ancho_kx=500,
                                                          radio_min_pix=2, radio_max_pix=20, n_radios=20,
                                                          alpha=0.5):
    """
    Barrido de radios para filtro Tukey circular CENTRADO en la posición del lóbulo.
    Incluye propagación de errores de calibración.
    
    Parámetros adicionales:
    - px_por_mm: píxeles por milímetro (para calibración)
    - err_px_por_mm: error en px_por_mm
    - alpha: parámetro Tukey (0=rectangular, 1=Hann)
    
    Retorna:
    - pasos, paso_promedio, std_pasos, error_total_px, error_rel, r_opt_pix, imagen_filtrada, mascara_pix, peaks
    """
    print(f"Iniciando barrido de radio con filtro Tukey (alpha={alpha}) centrado en lóbulo (en píxeles)")
    
    # Verificar calibración
    if px_por_mm is None or err_px_por_mm is None:
        print("⚠️ Advertencia: No se proporcionaron errores de calibración. Se devolverá solo error estadístico.")
        rel_err_cal = 0
    else:
        rel_err_cal = err_px_por_mm / px_por_mm
    
    # 1. Localizar lóbulo
    idx_ky0, idx_kx0 = localizar_lobulo_en_pixeles(roi, kx_rad, ky_rad, 
                                                    k_min, k_max, ancho_kx)
    
    # 2. Extraer datos
    matriz = roi["matriz"]
    col_central = roi["col_central"]
    kx_pix = roi["kx"]
    ky_pix = roi["ky"]
    
    # 3. Distancia al lóbulo
    distancia_al_lobulo_pix = np.sqrt((kx_pix - kx_pix[0, idx_kx0])**2 + 
                                      (ky_pix - ky_pix[idx_ky0, 0])**2)
    
    # 4. Precalcular FFT
    fshift = np.fft.fftshift(np.fft.fft2(matriz))
    
    # 5. Barrido de radios
    radios_pix = np.linspace(radio_min_pix, radio_max_pix, n_radios)
    resultados = []
    
    for r_pix in radios_pix:
        # Máscara Tukey
        mascara = np.zeros_like(distancia_al_lobulo_pix)
        idx = distancia_al_lobulo_pix <= r_pix
        
        if np.any(idx):
            rho = distancia_al_lobulo_pix[idx] / r_pix
            mascara_flat = np.zeros_like(rho)
            
            # Región izquierda (decaimiento)
            idx_left = rho < (alpha / 2)
            if np.any(idx_left):
                mascara_flat[idx_left] = 0.5 * (1 + np.cos(np.pi * (2 * rho[idx_left] / alpha - 1)))
            
            # Región central
            idx_center = (rho >= (alpha / 2)) & (rho <= (1 - alpha / 2))
            if np.any(idx_center):
                mascara_flat[idx_center] = 1.0
            
            # Región derecha
            idx_right = rho > (1 - alpha / 2)
            if np.any(idx_right):
                mascara_flat[idx_right] = 0.5 * (1 + np.cos(np.pi * (2 * rho[idx_right] / alpha - 2 / alpha + 1)))
            
            mascara[idx] = mascara_flat
        
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
            std_actual = np.std(pasos)
            resultados.append((r_pix, std_actual, imagen_filtrada, mascara, peaks))
    
    # 6. Seleccionar óptimo
    if len(resultados) == 0:
        print("⚠️ No se encontraron radios con suficientes picos")
        return None, None, None, None, None, None, None, None, None
    
    r_opt_pix, std_opt, img_opt, mask_opt, peaks_opt = min(resultados, key=lambda x: x[1])
    
    # 7. Estadísticas
    pasos_opt = np.diff(peaks_opt.astype(float))
    paso_promedio = np.mean(pasos_opt)
    paso_std = np.std(pasos_opt)
    
    # 8. Propagación de errores
    if px_por_mm is not None and err_px_por_mm is not None:
        err_sist_px = paso_promedio * rel_err_cal
        error_total_px = np.sqrt(paso_std**2 + err_sist_px**2)
        error_rel = error_total_px / paso_promedio if paso_promedio != 0 else 0
        
        print(f"📊 Paso promedio: {paso_promedio:.2f} ± {paso_std:.2f} px (estadístico)")
        print(f"   Error sistemático calibración: {err_sist_px:.3f} px")
        print(f"   Error total: {error_total_px:.2f} px ({error_rel*100:.2f}%)")
    else:
        error_total_px = paso_std
        error_rel = paso_std / paso_promedio if paso_promedio != 0 else 0
        print(f"📊 Paso promedio: {paso_promedio:.2f} ± {paso_std:.2f} px (solo estadístico)")
    
    return (pasos_opt, paso_promedio, paso_std, error_total_px, error_rel, r_opt_pix, img_opt, mask_opt, peaks_opt)

def visualizar_resultado_filtrado_binario(matriz_original, imagen_filtrada, mascara_binaria,
                                          fshift=None, kx_rad=None, ky_rad=None,
                                          pixel_size_um=None, k_crop_rad_m=10000):
    """
    Visualización específica para filtro binario con zoom en la región del lóbulo.
    Similar a las visualizaciones anteriores pero con máscara binaria.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Imagen original
    im0 = axes[0].imshow(matriz_original, cmap='gray', origin='upper')
    axes[0].set_title('Imagen Original', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('píxeles')
    axes[0].set_ylabel('píxeles')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='Intensidad')
    
    # 2. Imagen filtrada
    im1 = axes[1].imshow(imagen_filtrada, cmap='gray', origin='upper')
    axes[1].set_title('Imagen Filtrada (binario)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('píxeles')
    axes[1].set_ylabel('píxeles')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Intensidad')
    
    # 3. Máscara binaria sobre espectro (CON ZOOM)
    if fshift is not None:
        # Calcular espectro logarítmico
        espectro = np.abs(fshift)
        espectro_log = np.log10(1 + espectro)
        
        # Determinar extent para ejes calibrados
        if kx_rad is not None and ky_rad is not None:
            extent = [kx_rad[0, 0], kx_rad[0, -1], ky_rad[-1, 0], ky_rad[0, 0]]
            xlabel = '$k_x$ (rad/m)'
            ylabel = '$k_y$ (rad/m)'
        else:
            extent = None
            xlabel = 'píxeles de frecuencia'
            ylabel = 'píxeles de frecuencia'
        
        # Mostrar espectro completo como fondo
        axes[2].imshow(espectro_log, cmap='viridis', extent=extent, 
                      origin='upper', aspect='auto', alpha=0.7)
        
        # Superponer máscara binaria con transparencia y color
        # Usamos masked array para que los ceros sean transparentes
        mascara_mostrar = np.ma.masked_where(mascara_binaria == 0, mascara_binaria)
        im_mask = axes[2].imshow(mascara_mostrar, cmap='Reds', extent=extent,
                                 origin='upper', aspect='auto', alpha=0.6, vmin=0, vmax=1)
        
        # Configurar ejes
        axes[2].set_xlabel(xlabel, fontsize=11)
        axes[2].set_ylabel(ylabel, fontsize=11)
        axes[2].set_title('Máscara binaria sobre espectro de Fourier', 
                         fontsize=12, fontweight='bold')
        
        # Aplicar ZOOM centrado en el lóbulo
        if kx_rad is not None and ky_rad is not None:
            # Encontrar el centro del lóbulo a partir de la máscara
            # (asumimos que la máscara tiene valores 1 donde está el lóbulo)
            y_idx, x_idx = np.where(mascara_binaria > 0)
            if len(y_idx) > 0 and len(x_idx) > 0:
                # Centro aproximado del lóbulo
                ky_center = np.mean(ky_rad[y_idx, x_idx])
                kx_center = np.mean(kx_rad[y_idx, x_idx])
                
                # Aplicar zoom
                axes[2].set_xlim(kx_center - k_crop_rad_m/2, kx_center + k_crop_rad_m/2)
                axes[2].set_ylim(ky_center - k_crop_rad_m/2, ky_center + k_crop_rad_m/2)
        
        # Marcar el centro del espectro (DC) con líneas punteadas
        axes[2].axhline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
        axes[2].axvline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Colorbar para la máscara
        cbar = plt.colorbar(im_mask, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label('Máscara (0/1)', fontsize=10)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['0', '1'])
        
    else:
        # Si no hay espectro, mostrar solo máscara
        im2 = axes[2].imshow(mascara_binaria, cmap='Reds', origin='upper', aspect='auto')
        axes[2].set_title('Máscara binaria', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('píxeles de frecuencia')
        axes[2].set_ylabel('píxeles de frecuencia')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='Máscara')
    
    plt.tight_layout()
    return fig

def visualizar_perfil_con_picos(roi_matriz, imagen_filtrada, peaks, img_idx, col_central=None):
    """
    Visualización del perfil vertical con picos detectados.
    """
    if col_central is None:
        col_central = roi_matriz.shape[1] // 2
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Perfiles
    ax.plot(roi_matriz[:, col_central], alpha=0.5, linewidth=1.5, label='Original')
    ax.plot(imagen_filtrada[:, col_central], linewidth=2, label='Filtrada')
    
    # Picos
    ax.scatter(peaks, imagen_filtrada[peaks, col_central], 
               s=70, color='red', zorder=5, label=f'Picos ({len(peaks)})')
    
    # Líneas verticales para mostrar periodicidad
    if len(peaks) > 1:
        for i, p in enumerate(peaks):
            ax.axvline(p, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    
    ax.set_title(f'Perfil vertical - Imagen {img_idx}')
    ax.set_xlabel('Fila (píxeles)')
    ax.set_ylabel('Intensidad')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Añadir texto con estadísticas
    if len(peaks) > 1:
        pasos = np.diff(peaks)
        text = f'Paso medio: {np.mean(pasos):.2f} ± {np.std(pasos):.2f} px'
        ax.text(0.02, 0.98, text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig