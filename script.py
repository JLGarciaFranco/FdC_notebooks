import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate
import matplotlib.patches as patches

# Configurar el estilo de las gráficas
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

# Constantes físicas
h = 6.626e-34  # Constante de Planck (J⋅s)
c = 2.998e8    # Velocidad de la luz (m/s)
k = 1.381e-23  # Constante de Boltzmann (J/K)
sigma = 5.670e-8  # Constante de Stefan-Boltzmann (W⋅m⁻²⋅K⁻⁴)
T = 5800  # Temperatura solar (K)

def funcion_planck_radiancia(longitud_onda, temperatura):
    """
    Calcular la función de Planck (radiancia espectral)
    
    Parámetros:
    longitud_onda: longitud de onda en metros
    temperatura: temperatura en Kelvin
    
    Retorna:
    Radiancia espectral en W⋅m⁻²⋅sr⁻¹⋅m⁻¹
    """
    numerador = 2 * h * c**2
    denominador = longitud_onda**5 * (np.exp(h * c / (longitud_onda * k * temperatura)) - 1)
    return numerador / denominador

def funcion_planck_exitancia(longitud_onda, temperatura):
    """
    Calcular la exitancia espectral (integrada sobre hemisferio)
    M(λ,T) = π * B(λ,T)
    
    Esta es la cantidad que se integra para obtener σT⁴
    """
    radiancia = funcion_planck_radiancia(longitud_onda, temperatura)
    return np.pi * radiancia

# Crear arreglo de longitudes de onda (100 nm a 5000 nm)
longitud_onda_nm = np.linspace(100, 5000, 2000)
longitud_onda_m = longitud_onda_nm * 1e-9  # Convertir a metros

# Calcular función de Planck (radiancia)
radiancia_espectral = funcion_planck_radiancia(longitud_onda_m, T)
radiancia_por_nm = radiancia_espectral * 1e-9  # Convertir a por nm

# Calcular exitancia espectral (lo que realmente se integra para σT⁴)
exitancia_espectral = funcion_planck_exitancia(longitud_onda_m, T)
exitancia_por_nm = exitancia_espectral * 1e-9  # Convertir a por nm

# Calcular pico según ley de desplazamiento de Wien
pico_wien_nm = (2.898e-3 / T) * 1e9  # Convertir a nm

# Calcular total de Stefan-Boltzmann
total_stefan_boltzmann = sigma * T**4

# Calcular integral acumulativa (usando exitancia, no radiancia)
integral_acumulativa = np.zeros_like(longitud_onda_nm)
for i in range(1, len(longitud_onda_nm)):
    # Integración trapezoidal
    dλ = (longitud_onda_m[i] - longitud_onda_m[i-1])
    exitancia_promedio = (exitancia_espectral[i] + exitancia_espectral[i-1]) / 2
    integral_acumulativa[i] = integral_acumulativa[i-1] + exitancia_promedio * dλ

# Crear la gráfica con subgráficas
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[2, 1])

# Gráfica principal de la función de Planck (mostramos radiancia, más común)
ax1.plot(longitud_onda_nm, radiancia_por_nm, 'b-', linewidth=2.5, label='Función de Planck B(λ,T)')
ax1.fill_between(longitud_onda_nm, 0, radiancia_por_nm, alpha=0.3, color='skyblue', 
                 label=f'∫ π·B(λ,T) dλ = σT⁴ = {total_stefan_boltzmann:.2e} W⋅m⁻²')

# Añadir anotación de la ley de desplazamiento de Wien
ax1.axvline(pico_wien_nm, color='red', linestyle='--', linewidth=2, alpha=0.8)
ax1.annotate(f'Pico de Wien\nλₘₐₓ = {pico_wien_nm:.0f} nm', 
             xy=(pico_wien_nm, np.max(radiancia_por_nm) * 0.8),
             xytext=(pico_wien_nm + 400, np.max(radiancia_por_nm) * 0.9),
             fontsize=11, ha='left',
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# Resaltar espectro visible
mascara_visible = (longitud_onda_nm >= 380) & (longitud_onda_nm <= 750)
ax1.fill_between(longitud_onda_nm[mascara_visible], 0, radiancia_por_nm[mascara_visible], 
                 alpha=0.6, color='yellow', label='Espectro Visible (380-750 nm)')

# Añadir caja de texto con la ley de Stefan-Boltzmann
texto_ley = f'''Ley de Stefan-Boltzmann:
M = σT⁴
σ = {sigma:.3e} W⋅m⁻²⋅K⁻⁴
T = {T} K
M = {total_stefan_boltzmann:.2e} W⋅m⁻²

∫₀^∞ π·B(λ,T) dλ = σT⁴'''

props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax1.text(0.02, 0.98, texto_ley, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=props, fontfamily='monospace')

# Formato para gráfica principal
ax1.set_xlabel('Longitud de Onda (nm)', fontsize=14)
ax1.set_ylabel('Radiancia Espectral B(λ,T)\n(W⋅m⁻²⋅sr⁻¹⋅nm⁻¹)', fontsize=14)
ax1.set_title(f'Función de Planck para Radiación Solar (T = {T} K)', fontsize=16, pad=20)
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(100, 3000)

# Formato del eje y en notación científica
ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Gráfica de integral acumulativa
porcentaje = (integral_acumulativa / total_stefan_boltzmann) * 100
ax2.plot(longitud_onda_nm, porcentaje, 'g-', linewidth=2.5, label='Integral Acumulativa')
ax2.fill_between(longitud_onda_nm, 0, porcentaje, alpha=0.3, color='lightgreen')

# Añadir línea horizontal en 100%
ax2.axhline(100, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
ax2.text(4000, 102, '100% (σT⁴)', fontsize=10, color='red')

# Añadir anotaciones de porcentajes específicos
porcentajes_marcar = [50, 90, 95, 99]
for pct in porcentajes_marcar:
    # Encontrar longitud de onda donde alcanzamos este porcentaje
    idx = np.argmin(np.abs(porcentaje - pct))
    if idx < len(longitud_onda_nm) - 1 and longitud_onda_nm[idx] < 4000:
        ax2.annotate(f'{pct}% en {longitud_onda_nm[idx]:.0f} nm', 
                     xy=(longitud_onda_nm[idx], pct),
                     xytext=(longitud_onda_nm[idx] + 300, pct + 3),
                     fontsize=9, ha='left',
                     arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1))

ax2.set_xlabel('Longitud de Onda (nm)', fontsize=14)
ax2.set_ylabel('% Acumulativo\nde la Energía Total', fontsize=12)
ax2.set_title('Progreso de Integración: ¿Cuánto de σT⁴ se Captura?', fontsize=14)
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(100, 5000)
ax2.set_ylim(0, 105)

# Añadir título general de la figura
fig.suptitle('Función de Planck y Ley de Stefan-Boltzmann\n' + 
             'Demostrando que ∫₀^∞ π·B(λ,T) dλ = σT⁴', fontsize=18, y=0.98)

# Ajustar diseño
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('wien.png')
# Mostrar la gráfica

# Imprimir algunos resultados clave
print("=" * 70)
print("ANÁLISIS DE LA FUNCIÓN DE PLANCK Y LEY DE STEFAN-BOLTZMANN")
print("=" * 70)
print(f"Temperatura: {T} K")
print(f"Pico de desplazamiento de Wien: {pico_wien_nm:.1f} nm")
print(f"Total Stefan-Boltzmann (σT⁴): {total_stefan_boltzmann:.2e} W⋅m⁻²")
print(f"Resultado integración numérica: {integral_acumulativa[-1]:.2e} W⋅m⁻²")
print(f"Concordancia: {(integral_acumulativa[-1]/total_stefan_boltzmann)*100:.1f}%")
print("\nDistribución de energía:")
print(f"- Energía por debajo de 500 nm (UV): {porcentaje[np.argmin(np.abs(longitud_onda_nm - 500))]:.1f}%")
print(f"- Energía por debajo de 750 nm (Visible): {porcentaje[np.argmin(np.abs(longitud_onda_nm - 750))]:.1f}%")
print(f"- Energía por debajo de 1000 nm (IR cercano): {porcentaje[np.argmin(np.abs(longitud_onda_nm - 1000))]:.1f}%")
print(f"- Energía por debajo de 2000 nm: {porcentaje[np.argmin(np.abs(longitud_onda_nm - 2000))]:.1f}%")
print(f"- Energía por debajo de 3000 nm: {porcentaje[np.argmin(np.abs(longitud_onda_nm - 3000))]:.1f}%")

print("\n" + "="*70)
print("NOTA IMPORTANTE PARA FÍSICA DEL CLIMA:")
print("="*70)
print("La función de Planck B(λ,T) es la radiancia espectral (W⋅m⁻²⋅sr⁻¹⋅m⁻¹)")
print("Para obtener σT⁴, debemos integrar π·B(λ,T), que es la exitancia espectral")
print("El factor π viene de integrar sobre el hemisferio (ángulo sólido = π sr)")
print("Esto es crucial en física del clima para balances energéticos!")
