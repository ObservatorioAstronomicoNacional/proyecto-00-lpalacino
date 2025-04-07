## Ejercicio 00, Laura Camila Palacino Téllez.

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
from astropy.constants import R_earth

# PARÁMETROS ORBITALES

GM = 398600.4405  # km^3 / s^2
R_earth_km = R_earth.to(u.km).value  # 6371.0 km
a = 1.30262 * R_earth_km  # semieje mayor,  km
e = 0.16561  # excentricidad
w = np.radians(15)  # argumento del pericentro, rad
tp = Time("2025-03-31 00:00:00", scale='utc')  # tiempo del periastro
n = np.sqrt(GM / a**3)  # frecuencia angular media, rad/s

# ECUACIÓN DE KEPLER

    # Método de Newton-Raphson
def solve_kepler(M, e, tol=1e-10, max_iter=100):
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        delta_E = -f / f_prime
        E += delta_E
        if abs(delta_E) < tol:
            break
    return E

# ANOMALÍA VERDADERA

def true_anomaly_interpolated(E, e):
    h = 1e-3
    E0, E1, E2 = E - h, E, E + h

    def phi(E):
        return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

    phi0 = phi(E0)
    phi1 = phi(E1)
    phi2 = phi(E2)

    # Interpolación cuadrática
    L0 = ((E - E1)*(E - E2)) / ((E0 - E1)*(E0 - E2))
    L1 = ((E - E0)*(E - E2)) / ((E1 - E0)*(E1 - E2))
    L2 = ((E - E0)*(E - E1)) / ((E2 - E0)*(E2 - E1))

    return L0 * phi0 + L1 * phi1 + L2 * phi2

# POSICIÓN ORBITAL (r, phi)

def position(t):

    if isinstance(t, (float, int)):
        delta_t = t
    else:
        delta_t = (t - tp).to(u.s).value

    M = n * delta_t % (2 * np.pi)
    E = solve_kepler(M, e)
    f = true_anomaly_interpolated(E, e)
    r = a * (1 - e**2) / (1 + e*np.cos(f))
    phi = f + w
    return r, phi

# FECHA

def date(t_string):
    t = Time(t_string, scale='utc')
    r, phi = position(t)
    return r, np.degrees(phi)

# Interpolación cuadrática 
def interpolar_t(r_objetivo, r_vals, t_vals):
    for i in range(1, len(r_vals) - 1):
        if (r_vals[i-1] <= r_objetivo <= r_vals[i+1]) or (r_vals[i+1] <= r_objetivo <= r_vals[i-1]):
            r0, r1, r2 = r_vals[i-1], r_vals[i], r_vals[i+1]
            t0, t1, t2 = t_vals[i-1], t_vals[i], t_vals[i+1]
            break
    else:
        raise ValueError("r_objetivo fuera del rango de datos")

    A = [
        [r0**2, r0, 1],
        [r1**2, r1, 1],
        [r2**2, r2, 1]
    ]
    T = [t0, t1, t2]
    coef = np.linalg.solve(A, T)
    a, b, c = coef
    return a * r_objetivo**2 + b * r_objetivo + c

# Generar arrays de t y r para usar en la interpolación
def generar_t_y_r(N=1000, t_max=2*np.pi/n):
    t_vals = np.linspace(0, t_max, N)
    r_vals = np.array([position(t)[0] for t in t_vals])
    return t_vals, r_vals

def date_from_r(r_objetivo):
    t_vals, r_vals = generar_t_y_r()
    t0 = interpolar_t(r_objetivo, r_vals, t_vals)
    t_resultado = tp + t0 * u.s
    return t_resultado.iso

# ÓRBITA COMPLETA

def orbit():
    # se generan N número de puntos
    N = 1000
    E_vals = np.linspace(0, 2 * np.pi, N)
    r_vals = a * (1 - e * np.cos(E_vals))
    f_vals = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E_vals / 2),
                            np.sqrt(1 - e) * np.cos(E_vals / 2))
    phi_vals = f_vals + w

    # Coordenadas cartesianas para graficar
    x_vals = r_vals * np.cos(phi_vals)
    y_vals = r_vals * np.sin(phi_vals)

    # Gráfica
    plt.figure(figsize=(6, 6))
    plt.plot(x_vals, y_vals, label='Órbita')
    plt.plot(0, 0, 'yo', label='Centro de la Tierra')  # el centro de la órbita
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')
    plt.axis('equal')
    plt.grid(True)
    plt.title('Órbita del satélite')
    plt.legend()
    plt.show()

    return r_vals, phi_vals

orbit()  # Gráfica

# TESTEO DEL CÓDIGO

r_test, phi_test = date("2025-04-01 00:00:00")
print(f"r(t) = {r_test:.12f} km")
print(f"phi(t) = {phi_test:.12f} grados")