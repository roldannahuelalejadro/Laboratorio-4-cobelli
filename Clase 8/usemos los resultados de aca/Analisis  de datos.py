import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import pandas as pd

# ============================================
# DATOS EXPERIMENTALES
# ============================================

# Masas en gramos (masas aplicadas)
masitas_g = np.array([0.8234, 0.6678, 1.0301, 2.0670, 3.1022, 5.1782, 4.1323, 3.7700, 2.7438, 1.4912])

# Convertir a kg para los cálculos
masitas_kg = masitas_g * 1e-3

# Constantes del experimento
g = 9.80665      # m/s²
L = 0.29         # m (longitud de la viga)
x = L-0.01430            # m (posición de la medición, asumo que es en el extremo)
d = 0.00596        # m (diámetro de la viga, ¡VERIFICAR ESTE VALOR!)

# ============================================
# DATOS DE LOS 3 FILTRADOS (en μm)
# ============================================

# Filtrado 1 (primer archivo)
a1_vals = np.array([117.31, 116.20, 117.37, 124.38, 131.51, 147.25, 138.08, 137.24, 129.11, 118.94])
a1_errors = np.array([8.10, 8.32, 12.88, 7.33, 6.98, 7.00, 7.21, 6.96, 7.52, 5.74])

# Filtrado 2 (segundo archivo)
a2_vals = np.array([107.09, 106.02, 107.82, 117.55, 122.04, 138.29, 128.70, 128.53, 121.98, 109.55])
a2_errors = np.array([16.29, 15.40, 15.79, 18.13, 18.33, 18.80, 18.44, 18.82, 18.74, 17.11])

# Filtrado 3 (tercer archivo)
a3_vals = np.array([107.33, 105.90, 107.32, 117.55, 121.90, 138.50, 129.09, 127.84, 121.98, 104.97])
a3_errors = np.array([16.09, 15.34, 14.93, 18.01, 18.19, 18.22, 17.34, 18.39, 17.86, 15.55])

# Convertir de μm a metros
a1 = a1_vals * 1e-6
a2 = a2_vals * 1e-6
a3 = a3_vals * 1e-6

err_a1 = a1_errors * 1e-6
err_a2 = a2_errors * 1e-6
err_a3 = a3_errors * 1e-6

# ============================================
# MODELO CON OFFSET DE MASA
# a(m) = C * (g/E) * (m + m0)
# donde C = (32/π)*(1/d⁴)*(L*x² - x³/3)
# ============================================

# Constante geométrica
C_geo = (32/np.pi) * (1/d**4) * (L*x**2 - x**3/3)

def modelo_con_offset_masa(m, E, m0):
    """
    m: masa aplicada en kg
    E: módulo de Young en Pa
    m0: offset de masa en kg (masa no considerada)
    """
    return C_geo * (g / E) * (m + m0)

# ============================================
# AJUSTE PARA CADA FILTRADO (CON OFFSET DE MASA)
# ============================================

resultados = []

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (a_data, err_data, label, color) in enumerate([
    (a1, err_a1, 'Filtrado 1', 'blue'),
    (a2, err_a2, 'Filtrado 2', 'red'),
    (a3, err_a3, 'Filtrado 3', 'green')
]):
    
    # Ajuste con curve_fit (parámetros: E y m0)
    popt, pcov = curve_fit(
        modelo_con_offset_masa,
        masitas_kg,
        a_data,
        sigma=err_data,
        absolute_sigma=True,
        p0=[1e9, 0.001]  # estimación inicial: E=1 GPa, m0=1 g
    )
    
    E_ajustado, m0_ajustado = popt
    err_E = np.sqrt(pcov[0,0])
    err_m0 = np.sqrt(pcov[1,1])
    
    # Calcular modelo y residuos
    modelo = modelo_con_offset_masa(masitas_kg, E_ajustado, m0_ajustado)
    residuos = a_data - modelo
    
    # Estadísticas
    chi2 = np.sum(((a_data - modelo)/err_data)**2)
    gl = len(a_data) - len(popt)  # gl = N - 2
    chi2_red = chi2 / gl
    
    # Guardar resultados
    resultados.append({
        'Filtrado': label,
        'E (Pa)': E_ajustado,
        'E_error (Pa)': err_E,
        'm0 (kg)': m0_ajustado,
        'm0_error (kg)': err_m0,
        'm0 (g)': m0_ajustado * 1000,
        'm0_error (g)': err_m0 * 1000,
        'chi2': chi2,
        'chi2_red': chi2_red,
        'gl': gl
    })
    
    # Graficar en el subplot correspondiente
    ax = axes[i]
    
    # Datos con errores
    ax.errorbar(masitas_g, a_data*1e6, yerr=err_data*1e6, 
                fmt='o', color=color, capsize=3, label='Datos')
    
    # Línea del ajuste
    m_linea = np.linspace(-0.5, max(masitas_g)*1.05, 300)
    a_linea = modelo_con_offset_masa(m_linea*1e-3, E_ajustado, m0_ajustado)
    ax.plot(m_linea, a_linea*1e6, '--', color=color, 
            linewidth=2, label=f'Ajuste con offset')
    
    # Marcar el offset de masa
    ax.axvline(x=m0_ajustado*1000, color=color, linestyle=':', alpha=0.5, 
               label=f'm₀ = {m0_ajustado*1000:.3f} g')
    
    ax.set_xlabel('Masa aplicada [g]')
    ax.set_ylabel('a [μm]')
    ax.set_title(f'{label}\nE = {E_ajustado:.2e} ± {err_E:.2e} Pa\n'
                 f'm₀ = {m0_ajustado*1000:.3f} ± {err_m0*1000:.3f} g\n'
                 f'χ²/gl = {chi2_red:.3f}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-0.5, max(masitas_g)*1.05)
    
    # Subplot de residuos combinado
    axes[3].errorbar(masitas_g, residuos*1e6, yerr=err_data*1e6, 
                     fmt='o', color=color, capsize=3, 
                     label=f'Residuos {label}', alpha=0.7)

# Configurar el subplot de residuos
axes[3].axhline(0, color='black', linestyle='--', linewidth=1)
axes[3].set_xlabel('Masa aplicada [g]')
axes[3].set_ylabel('Residuos [μm]')
axes[3].set_title('Comparación de residuos (modelo con offset de masa)')
axes[3].grid(True, alpha=0.3)
axes[3].legend()

plt.tight_layout()
plt.show()