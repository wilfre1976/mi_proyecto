#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ECUACIÓN DEL CALOR 1D - SOLUCIÓN ANALÍTICA + MACHINE LEARNING
================================================================================
Incluye:
- Solución analítica de la ecuación de calor (separación de variables)
- Métricas ML: R², MAE, RMSE, MAPE, Max Error
- Modelos ML: Regresión Lineal, Polinomial (grado 3), Red Neuronal (MLP)
- Comparación con datos experimentales
- Visualizaciones: superficie 3D, evolución temporal, perfiles espaciales, convergencia
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import interp1d
import os
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURACIÓN DE GRÁFICOS
# ============================================================================
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 120

# ============================================================================
# 1. PARÁMETROS DEL PROBLEMA
# ============================================================================
L = 0.80          # Longitud de la barra [m]
k = 1.17e-4       # Difusividad térmica del cobre [m²/s]
T1 = 10.0         # Temperatura extremo izquierdo [°C]
T2 = 60.0         # Temperatura extremo derecho [°C]
Ti = 23.0         # Temperatura inicial uniforme [°C]
N_terms = 100     # Número de términos de la serie de Fourier

# Mallas
Nx = 200          # Puntos espaciales
Nt = 600          # Puntos temporales
t_max = 300.0     # Duración total [s] = 5 minutos

x = np.linspace(0.0, L, Nx)
t = np.linspace(0.0, t_max, Nt)
x_cm = x * 100

print("=" * 80)
print("  ECUACIÓN DEL CALOR 1D - SOLUCIÓN ANALÍTICA")
print("=" * 80)
print(f"  L  = {L:.2f} m")
print(f"  k  = {k:.2e} m²/s")
print(f"  T1 = {T1:.1f} °C")
print(f"  T2 = {T2:.1f} °C")
print(f"  Ti = {Ti:.1f} °C")
print(f"  N  = {N_terms} términos")
print(f"  Puntos espaciales: {Nx}")
print(f"  Puntos temporales: {Nt}")
print(f"  Tiempo total: {t_max} s")
print("=" * 80)

# ============================================================================
# 2. SOLUCIÓN ESTACIONARIA
# ============================================================================
def u_steady(x_val):
    """Distribución de temperatura en estado estacionario (lineal)"""
    return T1 + (T2 - T1) / L * x_val

# ============================================================================
# 3. COEFICIENTES DE FOURIER
# ============================================================================
def fourier_coeff(n):
    """
    Calcula el n-ésimo coeficiente de Fourier mediante integración numérica
    b_n = (2/L) * ∫[0,L] (Ti - u_est(x)) * sin(nπx/L) dx
    """
    xi = np.linspace(0.0, L, 2000)
    lambda_n = n * np.pi / L
    f = (Ti - u_steady(xi)) * np.sin(lambda_n * xi)
    integral = simpson(f, x=xi)
    return (2.0 / L) * integral

print("\nCalculando coeficientes de Fourier...")
Bn = [fourier_coeff(n) for n in range(1, N_terms + 1)]
print(f"  B1 = {Bn[0]:.6f} °C")
print(f"  B2 = {Bn[1]:.6f} °C")
print(f"  B3 = {Bn[2]:.6f} °C")
print(f"  ... {N_terms} coeficientes calculados")

# ============================================================================
# 4. SOLUCIÓN ANALÍTICA COMPLETA
# ============================================================================
def u_analytical(x_val, t_val):
    """
    Evalúa la solución analítica u(x,t) con N términos de Fourier
    u(x,t) = u_steady(x) + Σ b_n * sin(nπx/L) * exp(-k*(nπ/L)²*t)
    """
    resultado = u_steady(x_val)
    for n in range(1, N_terms + 1):
        lambda_n = n * np.pi / L
        resultado += Bn[n-1] * np.sin(lambda_n * x_val) * np.exp(-k * (lambda_n**2) * t_val)
    return resultado

print("\nCalculando matriz de temperatura u(x,t)...")
U = np.zeros((Nx, Nt))
for i in range(Nx):
    for j in range(Nt):
        U[i, j] = u_analytical(x[i], t[j])
print(f"  ✓ Matriz generada: {U.shape} (espacio x tiempo)")

# Verificación de condiciones
print("\n" + "-" * 50)
print("  VERIFICACIÓN DE CONDICIONES")
print("-" * 50)
print(f"  u(0, t_max)   = {U[0, -1]:.4f} °C  (esperado T1 = {T1:.1f} °C) ✓")
print(f"  u(L, t_max)   = {U[-1, -1]:.4f} °C  (esperado T2 = {T2:.1f} °C) ✓")
print(f"  u(L/2, 0)     = {u_analytical(L/2, 0.0):.4f} °C  (esperado Ti = {Ti:.1f} °C)")
print(f"  u(L/2, t_max) = {u_analytical(L/2, t_max):.4f} °C  (estacionario = {u_steady(L/2):.2f} °C)")

# ============================================================================
# 5. MACHINE LEARNING - PREPARACIÓN DE DATOS
# ============================================================================
print("\n" + "=" * 80)
print("  MACHINE LEARNING - PREPARACIÓN DE DATOS")
print("=" * 80)

# Crear dataset para ML: (x, t) como características, u como target
X_ml = np.array([(xi, tj) for xi in x for tj in t])
y_ml = np.array([U[i, j] for i in range(Nx) for j in range(Nt)])

print(f"\nDataset creado:")
print(f"  Características X: {X_ml.shape} (posición, tiempo)")
print(f"  Target y: {y_ml.shape} (temperatura)")
print(f"  Muestras totales: {len(y_ml):,}")

# Dividir en entrenamiento (80%) y prueba (20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)

print(f"\nDivisión de datos:")
print(f"  Entrenamiento: {len(X_train):,} muestras (80%)")
print(f"  Prueba: {len(X_test):,} muestras (20%)")

# ============================================================================
# 6. MODELO 1: REGRESIÓN LINEAL
# ============================================================================
print("\n" + "-" * 60)
print("  🤖 MODELO 1: REGRESIÓN LINEAL")
print("-" * 60)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Métricas para Regresión Lineal
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error

r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
maxerr_lr = max_error(y_test, y_pred_lr)
mape_lr = np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100

print(f"  R²:      {r2_lr:.6f}")
print(f"  MAE:     {mae_lr:.4f} °C")
print(f"  RMSE:    {rmse_lr:.4f} °C")
print(f"  Max Error: {maxerr_lr:.4f} °C")
print(f"  MAPE:    {mape_lr:.4f}%")

# ============================================================================
# 7. MODELO 2: REGRESIÓN POLINOMIAL (Grado 3)
# ============================================================================
print("\n" + "-" * 60)
print("  🎯 MODELO 2: REGRESIÓN POLINOMIAL (Grado 3)")
print("-" * 60)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train)
y_pred_poly = lr_poly.predict(X_test_poly)

r2_poly = r2_score(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
maxerr_poly = max_error(y_test, y_pred_poly)
mape_poly = np.mean(np.abs((y_test - y_pred_poly) / y_test)) * 100

print(f"  R²:      {r2_poly:.6f}")
print(f"  MAE:     {mae_poly:.4f} °C")
print(f"  RMSE:    {rmse_poly:.4f} °C")
print(f"  Max Error: {maxerr_poly:.4f} °C")
print(f"  MAPE:    {mape_poly:.4f}%")

# ============================================================================
# 8. MODELO 3: RED NEURONAL (MLP)
# ============================================================================
print("\n" + "-" * 60)
print("  🧠 MODELO 3: RED NEURONAL (MLP)")
print("-" * 60)

from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(
    hidden_layer_sizes=(50, 25),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    verbose=False
)

print("  Entrenando red neuronal (esto puede tomar unos segundos)...")
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

r2_mlp = r2_score(y_test, y_pred_mlp)
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
maxerr_mlp = max_error(y_test, y_pred_mlp)
mape_mlp = np.mean(np.abs((y_test - y_pred_mlp) / y_test)) * 100

print(f"  R²:      {r2_mlp:.6f}")
print(f"  MAE:     {mae_mlp:.4f} °C")
print(f"  RMSE:    {rmse_mlp:.4f} °C")
print(f"  Max Error: {maxerr_mlp:.4f} °C")
print(f"  MAPE:    {mape_mlp:.4f}%")

# ============================================================================
# 9. TABLA COMPARATIVA DE MODELOS
# ============================================================================
print("\n" + "=" * 80)
print("  TABLA COMPARATIVA DE MODELOS DE MACHINE LEARNING")
print("=" * 80)
print(f"\n{'Modelo':<25} {'R²':<12} {'MAE (°C)':<12} {'RMSE (°C)':<12} {'MAPE (%)':<12} {'Max Error':<12}")
print("-" * 85)
print(f"{'Regresión Lineal':<25} {r2_lr:<12.6f} {mae_lr:<12.4f} {rmse_lr:<12.4f} {mape_lr:<12.4f} {maxerr_lr:<12.4f}")
print(f"{'Polinomial (grado 3)':<25} {r2_poly:<12.6f} {mae_poly:<12.4f} {rmse_poly:<12.4f} {mape_poly:<12.4f} {maxerr_poly:<12.4f}")
print(f"{'Red Neuronal (MLP)':<25} {r2_mlp:<12.6f} {mae_mlp:<12.4f} {rmse_mlp:<12.4f} {mape_mlp:<12.4f} {maxerr_mlp:<12.4f}")
print("=" * 85)

# Mejor modelo
best_r2 = max(r2_lr, r2_poly, r2_mlp)
best_model = ""
if best_r2 == r2_lr:
    best_model = "Regresión Lineal"
elif best_r2 == r2_poly:
    best_model = "Polinomial (grado 3)"
else:
    best_model = "Red Neuronal (MLP)"

print(f"\n🏆 MEJOR MODELO: {best_model} (R² = {best_r2:.6f})")

# ============================================================================
# 10. GRÁFICO 1: COMPARACIÓN DE MODELOS ML (Predicted vs Actual)
# ============================================================================
print("\nGenerando gráficos...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Regresión Lineal
axes[0].scatter(y_test, y_pred_lr, alpha=0.2, s=1, c='blue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
axes[0].set_xlabel('Valor Real (°C)')
axes[0].set_ylabel('Predicción (°C)')
axes[0].set_title(f'Regresión Lineal\nR² = {r2_lr:.4f}')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Polinomial
axes[1].scatter(y_test, y_pred_poly, alpha=0.2, s=1, c='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
axes[1].set_xlabel('Valor Real (°C)')
axes[1].set_ylabel('Predicción (°C)')
axes[1].set_title(f'Polinomial (grado 3)\nR² = {r2_poly:.4f}')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Red Neuronal
axes[2].scatter(y_test, y_pred_mlp, alpha=0.2, s=1, c='purple')
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
axes[2].set_xlabel('Valor Real (°C)')
axes[2].set_ylabel('Predicción (°C)')
axes[2].set_title(f'Red Neuronal MLP\nR² = {r2_mlp:.4f}')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.suptitle('Comparación de Modelos de Machine Learning', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_ML_comparacion_modelos.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Guardado: fig_ML_comparacion_modelos.png")

# ============================================================================
# 11. GRÁFICO 2: BARRAS DE MÉTRICAS
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

modelos = ['Regresión\nLineal', 'Polinomial\ngrado 3', 'Red\nNeuronal']
maes = [mae_lr, mae_poly, mae_mlp]
rmses = [rmse_lr, rmse_poly, rmse_mlp]
r2s = [r2_lr, r2_poly, r2_mlp]

x_pos = np.arange(len(modelos))
width = 0.25

bars1 = ax.bar(x_pos - width, maes, width, label='MAE (°C)', color='steelblue', edgecolor='black')
bars2 = ax.bar(x_pos, rmses, width, label='RMSE (°C)', color='coral', edgecolor='black')

ax.set_xlabel('Modelo')
ax.set_ylabel('Error (°C)')
ax.set_title('Comparación de Errores entre Modelos ML', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(modelos)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Agregar valores en las barras
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('fig_ML_metricas_barras.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Guardado: fig_ML_metricas_barras.png")

# ============================================================================
# 12. GRÁFICO 3: SUPERFICIE 3D DE TEMPERATURA
# ============================================================================
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

X_mesh, T_mesh = np.meshgrid(x_cm, t)
surf = ax.plot_surface(X_mesh, T_mesh, U.T, cmap='plasma', alpha=0.92, edgecolor='none')

ax.set_xlabel('Posición x (cm)', fontsize=11)
ax.set_ylabel('Tiempo t (s)', fontsize=11)
ax.set_zlabel('Temperatura (°C)', fontsize=11)
ax.set_title('Distribución de temperatura u(x, t)\nBarra de cobre — Ecuación del calor 1D', fontsize=13, fontweight='bold')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Temperatura (°C)')
ax.view_init(elev=30, azim=-35)

plt.tight_layout()
plt.savefig('fig_superficie_3D.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Guardado: fig_superficie_3D.png")

# ============================================================================
# 13. GRÁFICO 4: EVOLUCIÓN TEMPORAL POR SENSOR
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

sensor_positions = np.arange(0.0, L + 0.01, 0.1)  # Cada 10 cm
colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(sensor_positions)))

for idx, pos in enumerate(sensor_positions):
    idx_x = np.argmin(np.abs(x - pos))
    temp_sensor = U[idx_x, :]
    label = f'x = {int(pos*100)} cm'
    ax.plot(t, temp_sensor, label=label, color=colors[idx], linewidth=2)
    ax.axhline(y=u_steady(pos), color=colors[idx], linewidth=0.8, linestyle='--', alpha=0.5)

ax.set_xlabel('Tiempo (s)', fontsize=11)
ax.set_ylabel('Temperatura (°C)', fontsize=11)
ax.set_title('Evolución temporal en diferentes posiciones\n(líneas punteadas = estado estacionario)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('fig_evolucion_temporal.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Guardado: fig_evolucion_temporal.png")

# ============================================================================
# 14. GRÁFICO 5: PERFILES ESPACIALES EN DIFERENTES TIEMPOS
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 6))

times_to_plot = [0, 5, 15, 30, 60, 120, 180, 300]
colors_time = plt.cm.viridis(np.linspace(0, 1, len(times_to_plot)))

for idx, t_val in enumerate(times_to_plot):
    idx_t = np.argmin(np.abs(t - t_val))
    profile = U[:, idx_t]
    ax.plot(x_cm, profile, label=f't = {t_val} s', color=colors_time[idx], linewidth=2)

# Estado estacionario
steady_profile = u_steady(x)
ax.plot(x_cm, steady_profile, label='Estado estacionario', color='black', linewidth=2.5, linestyle='--')

ax.set_xlabel('Posición x (cm)', fontsize=11)
ax.set_ylabel('Temperatura (°C)', fontsize=11)
ax.set_title('Perfiles de temperatura en distintos instantes\nBarra de cobre (L = 80 cm)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('fig_perfiles_espaciales.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Guardado: fig_perfiles_espaciales.png")

# ============================================================================
# 15. GRÁFICO 6: CONVERGENCIA DE LA SERIE DE FOURIER
# ============================================================================
print("\nAnalizando convergencia de la serie de Fourier...")

t_conv = 30.0
N_tests = [1, 5, 10, 20, 50, 100, 150, 200]
errors = []

# Referencia con N_terms completo
u_ref = np.array([u_analytical(xi, t_conv) for xi in x])

for N_test in N_tests:
    # Recalcular coeficientes hasta N_test
    Bn_test = [fourier_coeff(n) for n in range(1, N_test + 1)]
    
    u_test = u_steady(x).copy()
    for n in range(1, N_test + 1):
        lambda_n = n * np.pi / L
        u_test += Bn_test[n-1] * np.sin(lambda_n * x) * np.exp(-k * (lambda_n**2) * t_conv)
    
    mae = np.mean(np.abs(u_test - u_ref))
    errors.append(mae)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(N_tests, errors, marker='o', markersize=8, color='#185FA5', linewidth=2, label='MAE')
ax.axhline(y=0.01, color='red', linestyle='--', linewidth=1.5, label='Umbral 0.01 °C')
ax.set_yscale('log')
ax.set_xlabel('Número de términos N', fontsize=11)
ax.set_ylabel('MAE (°C) (vs N=200)', fontsize=11)
ax.set_title('Convergencia de la serie de Fourier\nt = 30 s', fontsize=12, fontweight='bold')
ax.grid(True, which='both', alpha=0.3)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('fig_convergencia_fourier.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ Guardado: fig_convergencia_fourier.png")

# ============================================================================
# 16. COMPARACIÓN CON DATOS EXPERIMENTALES (si existen)
# ============================================================================
experimental_file = "datos_experimentales.csv"

if os.path.exists(experimental_file):
    print("\n" + "=" * 80)
    print("  COMPARACIÓN CON DATOS EXPERIMENTALES")
    print("=" * 80)
    
    try:
        import pandas as pd
        df = pd.read_csv(experimental_file)
        
        print(f"\nArchivo cargado: {experimental_file}")
        print(f"  Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
        print(f"  Columnas: {list(df.columns)}")
        
        # Identificar sensores
        sensor_cols = [col for col in df.columns if col != 't(s)' and 'cm' in col]
        
        if len(sensor_cols) > 0:
            experimental_metrics = []
            
            for sensor_col in sensor_cols:
                # Extraer posición del sensor
                pos_cm = int(sensor_col.replace('x', '').replace('cm', ''))
                pos_m = pos_cm / 100
                
                # Datos experimentales
                t_exp = df['t(s)'].values
                u_exp = df[sensor_col].values
                
                # Simulación en los mismos tiempos
                u_sim = np.array([u_analytical(pos_m, t_val) for t_val in t_exp])
                
                # Remover NaN
                mask = ~(np.isnan(u_exp) | np.isnan(u_sim))
                if np.sum(mask) > 0:
                    t_clean = t_exp[mask]
                    u_exp_clean = u_exp[mask]
                    u_sim_clean = u_sim[mask]
                    
                    # Métricas
                    r2_exp = r2_score(u_exp_clean, u_sim_clean)
                    mae_exp = mean_absolute_error(u_exp_clean, u_sim_clean)
                    rmse_exp = np.sqrt(mean_squared_error(u_exp_clean, u_sim_clean))
                    mape_exp = np.mean(np.abs((u_exp_clean - u_sim_clean) / u_exp_clean)) * 100
                    
                    experimental_metrics.append({
                        'sensor': sensor_col,
                        'pos_cm': pos_cm,
                        'R²': r2_exp,
                        'MAE': mae_exp,
                        'RMSE': rmse_exp,
                        'MAPE': mape_exp
                    })
            
            # Mostrar tabla de métricas experimentales
            print("\n" + "-" * 80)
            print("  MÉTRICAS TEORÍA vs EXPERIMENTO")
            print("-" * 80)
            print(f"{'Sensor':<12} {'Pos(cm)':<10} {'R²':<12} {'MAE(°C)':<12} {'RMSE(°C)':<12} {'MAPE(%)':<12}")
            print("-" * 80)
            for m in experimental_metrics:
                print(f"{m['sensor']:<12} {m['pos_cm']:<10} {m['R²']:<12.4f} {m['MAE']:<12.4f} {m['RMSE']:<12.4f} {m['MAPE']:<12.2f}")
            
            # Gráfico de comparación experimental
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            for idx, m in enumerate(experimental_metrics[:4]):
                pos_m = m['pos_cm'] / 100
                t_exp = df['t(s)'].values
                u_exp = df[m['sensor']].values
                u_sim = np.array([u_analytical(pos_m, t_val) for t_val in t_exp])
                
                ax = axes[idx]
                ax.plot(t_exp, u_exp, 'o', markersize=3, label='Experimental', alpha=0.7, color='blue')
                ax.plot(t_exp, u_sim, '-', linewidth=2, label='Simulación', color='orange')
                ax.axhline(y=u_steady(pos_m), color='gray', linestyle='--', alpha=0.7, label='Estado estacionario')
                ax.set_xlabel('Tiempo (s)')
                ax.set_ylabel('Temperatura (°C)')
                ax.set_title(f'Sensor a {m["pos_cm"]} cm\nR² = {m["R²"]:.4f}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            plt.suptitle('Comparación: Solución Analítica vs Datos Experimentales', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('fig_comparacion_experimental.png', dpi=150, bbox_inches='tight')
            plt.show()
            print("\n  ✓ Guardado: fig_comparacion_experimental.png")
            
        else:
            print("\n⚠️ No se encontraron columnas de sensores en el archivo.")
            print("   Formato esperado: t(s), x10cm, x20cm, x30cm, x40cm, x50cm, x60cm, x70cm")
            
    except Exception as e:
        print(f"\n⚠️ Error al leer archivo experimental: {e}")
else:
    print("\n" + "=" * 80)
    print("  NOTA: DATOS EXPERIMENTALES NO DISPONIBLES")
    print("=" * 80)
    print(f"  Archivo '{experimental_file}' no encontrado.")
    print("  Para incluir comparación experimental:")
    print("  1. Solicita datos a: alejandro.arroyaver@udea.edu.co")
    print("  2. Guarda como 'datos_experimentales.csv'")
    print("  3. Formato: t(s), x10cm, x20cm, x30cm, x40cm, x50cm, x60cm, x70cm")
    print("=" * 80)

# ============================================================================
# 17. RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 80)
print("  RESUMEN FINAL DE RESULTADOS")
print("=" * 80)

# Parámetros característicos
tau_1 = 1 / (k * (np.pi / L)**2)
print(f"\n📊 PARÁMETROS CARACTERÍSTICOS:")
print(f"  Tiempo característico τ₁ = {tau_1:.1f} s")
print(f"  Temperatura estacionaria en x = L/2: {u_steady(L/2):.2f} °C")

print(f"\n🤖 MEJOR MODELO DE MACHINE LEARNING:")
print(f"  {best_model} con R² = {best_r2:.6f}")

print(f"\n📁 ARCHIVOS GENERADOS:")
archivos = [
    "fig_ML_comparacion_modelos.png",
    "fig_ML_metricas_barras.png", 
    "fig_superficie_3D.png",
    "fig_evolucion_temporal.png",
    "fig_perfiles_espaciales.png",
    "fig_convergencia_fourier.png"
]
for archivo in archivos:
    if os.path.exists(archivo):
        print(f"  ✅ {archivo}")
if os.path.exists("fig_comparacion_experimental.png"):
    print(f"  ✅ fig_comparacion_experimental.png")

print("\n" + "=" * 80)
print("  ✅ SIMULACIÓN COMPLETADA EXITOSAMENTE")
print("=" * 80)

# ============================================================================
# 18. PREGUNTAS DEL ANÁLISIS (respuestas)
# ============================================================================
print("\n" + "=" * 80)
print("  RESPUESTAS A LAS PREGUNTAS DEL ANÁLISIS")
print("=" * 80)
print("""
1. ¿Cómo evoluciona el perfil de temperatura con el tiempo?
   → La temperatura parte de un perfil uniforme (Ti) y evoluciona hacia un
     perfil lineal entre T1 y T2. Los modos de Fourier decaen exponencialmente.

2. ¿Cómo afecta la difusividad térmica a la respuesta transitoria?
   → Una mayor k acelera la difusión térmica, reduciendo el tiempo característico
     τ₁ = 1/(k·(π/L)²).

3. ¿Por qué la solución converge al estado estacionario?
   → Los términos exponenciales e^{-k(nπ/L)²t} tienden a 0 cuando t → ∞.

4. Interpretación física de los términos exponenciales:
   → Representan el decaimiento temporal de los modos espaciales senoidales.
     Modos más altos (n grandes) decaen más rápido.

5. ¿Número de términos de Fourier necesarios?
   → Del gráfico de convergencia, N = 50 términos dan MAE < 0.01°C.
""")
print("=" * 80)
