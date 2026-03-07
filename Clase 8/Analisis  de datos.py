import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# ==================== ARCHIVOS ====================

archivos = {
"Binaria": r"C:\Users\User\Desktop\Laboratorio-4-cobelli\imagenes_analisis_binaria\resultados_binaria_20260307_170351.csv",
"Hamming": r"C:\Users\User\Desktop\Laboratorio-4-cobelli\imagenes_analisis_hamming\resultados_hamming_20260307_170429.csv",
"Tukey": r"C:\Users\User\Desktop\Laboratorio-4-cobelli\imagenes_analisis_tukey\resultados_tukey_20260307_170514.csv"
}

# ==================== MASAS ====================

masitas = np.array([
0.8234,0.6678,1.0301,2.0670,3.1022,
5.1782,4.1323,3.7700,2.7438,1.4912
])

m_kg = masitas * 1e-3

# ==================== CONSTANTES ====================

g = 9.80665
L = 0.29
x = L
d = 0.00596

def modelo_fisico(m,E,b):
    return (32/np.pi)*(1/d**4)*((m*g)/E)*(L*x**2-x**3/3)+b


# ==================== FIGURA ====================

fig, ax = plt.subplots(4,1, figsize=(7,12), sharex=True)
offsets = {
"Binaria": -0.04,
"Hamming": 0.0,
"Tukey": 0.04
}

markers = {
"Binaria": "o",
"Hamming": "s",
"Tukey": "^"
}

colores = {
"Binaria": "lightcoral",
"Hamming": "cornflowerblue",
"Tukey": "mediumseagreen"
}

m_line = np.linspace(masitas.min(), masitas.max(), 200)

resultados = []

# ==================== LOOP ====================

for i,(nombre,ruta) in enumerate(archivos.items()):

    df = pd.read_csv(ruta)

    rendijas = df["a_um"].values * 1e-6
    err = df["err_a_um"].values * 1e-6

    mask = ~np.isnan(rendijas)

    a = rendijas[mask]
    err = err[mask]
    m = m_kg[mask]
    m_g = masitas[mask]

    popt,pcov = curve_fit(
        modelo_fisico,
        m,
        a,
        sigma=err,
        absolute_sigma=True
    )

    E,b = popt
    err_E = np.sqrt(pcov[0,0])

    modelo = modelo_fisico(m,E,b)
    residuos = a-modelo

    chi2 = np.sum((residuos/err)**2)
    gl = len(a)-len(popt)
    chi2_red = chi2/gl
    p = stats.chi2.sf(chi2,gl)

    resultados.append([nombre,E,err_E,chi2_red,p])

    # ==================== AJUSTE ====================

    ax[i].errorbar(
        m_g,
        a*1e6,
        yerr=err*1e6,
        fmt='o',
        capsize=4,
        color=colores[nombre],
        label=(
            f"{nombre}\n"
            f"E = {E/1e9:.2f} ± {err_E/1e9:.2f} GPa\n"
            f"$\\chi^2_r$ = {chi2_red:.2f}"
        )
    )

    ax[i].plot(
        m_line,
        modelo_fisico(m_line*1e-3,E,b)*1e6,
        color=colores[nombre]
    )

    ax[i].set_ylabel("a [µm]")
    ax[i].legend()
    ax[i].grid(alpha=0.3)

    # ==================== RESIDUOS ====================

    ax[3].errorbar(
    m_g + offsets[nombre],
    residuos*1e6,
    yerr=err*1e6,
    fmt=markers[nombre],
    color=colores[nombre],
    alpha=0.85,
    label=nombre
    )
# ==================== SUBPLOT RESIDUOS ====================

ax[3].axhline(0,color="black",linestyle="--")

ax[3].set_ylabel("residuos [µm]")
ax[3].set_xlabel("masa [g]")
ax[3].legend()
ax[3].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== TABLA ====================

tabla = pd.DataFrame(
    resultados,
    columns=["Mascara","E (Pa)","σE (Pa)","χ²_red","p"]
)

print("\nTabla comparativa:\n")
print(tabla)

print("\nValores en GPa:\n")
print(
tabla.assign(
E_GPa = tabla["E (Pa)"]/1e9,
err_GPa = tabla["σE (Pa)"]/1e9
)[["Mascara","E_GPa","err_GPa","χ²_red","p"]]
)