import imageio.v2 as imageio
import matplotlib.pyplot as plt 
from pathlib import Path
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import os
from scipy.signal import find_peaks

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

imagen = images[0]

center_x = 890
center_y = 1660
offset  = 600

matriz = imagen[center_x - offset : center_x + offset,
                center_y - offset : center_y + offset,
                2].astype(float)
#=========================================================================

# # Promedio por filas (promedia cada fila → queda un perfil en eje vertical)
# perfil_filas = np.mean(espectro_log, axis=1)

# # Promedio por columnas (promedia cada columna → perfil en eje horizontal)
# perfil_columnas = np.mean(espectro_log, axis=0)

# # Ejes (índices de píxeles en la FFT)
# y = np.arange(len(perfil_filas))
# x = np.arange(len(perfil_columnas)) 


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

plot_fft = True 
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
    row = offset   
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
    plt.plot(perfil_fila, color='tab:blue', linewidth=2.5, label='Perfil filtrado')
    plt.plot(peaks, perfil_fila[peaks], "x", markersize=12, color='red', label='Picos')
    plt.show()
else:
    print("no se encontraron picos")

# paso_fit = pixeles / mm


dist_real = 1  # mm (CAMBIAR por referencia real)


#=========================================================================
analisis_tipo_imageJ=False
if analisis_tipo_imageJ:
    fig, ax = plt.subplots()
    ax.imshow(imagen)
    ax.set_title("Hacé click en 2 puntos")

    puntos = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            puntos.append((x, y))
            print(f"Punto seleccionado: ({x}, {y})")

            ax.plot(x, y, 'ro')  # punto rojo
            fig.canvas.draw()

            # Cuando hay 2 puntos → calcular distancia
            if len(puntos) == 2:
                p1, p2 = puntos

                dist_pixeles = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

                dist_real = 1  # mm (CAMBIAR por referencia real)
                escala = dist_real / dist_pixeles

                print(f"Distancia en pixeles: {dist_pixeles}")
                print(f"Escala: {escala} mm/pixel")
                

                # Dibujar línea
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-')

                # Mostrar texto en la imagen
                ax.text(p1[0], p1[1],
                    f"Dist: {dist_pixeles:.4f} px\nEscala: {escala:.4f} mm/px",
                    color='red', fontsize=10)

                fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

