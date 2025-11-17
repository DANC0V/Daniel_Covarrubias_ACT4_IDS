#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# -------------------------
# Parámetros
# -------------------------
Fs = 10_000        # Frecuencia de muestreo
T = 2.0            # Duración
t = np.arange(0, T, 1/Fs)

Fm = 5.0           # Frecuencia del mensaje
Am = 1.0           # Amplitud del mensaje

Fc = 100.0         # Frecuencia de portadora
Ac = 2.0           # Amplitud de portadora

mu = Am / Ac       # Índice de modulación
print(f"Índice de modulación mu = {mu:.3f}")

# Carpeta de imágenes
FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------------
# Señales
# -------------------------
m_t = Am * np.sin(2*np.pi*Fm*t)
c_t = Ac * np.cos(2*np.pi*Fc*t)

m_norm = m_t / Am
s_am = Ac * (1 + mu*m_norm) * np.cos(2*np.pi*Fc*t)

# -------------------------
# Funciones para graficar
# -------------------------
def plot_time(x, t, title, fname, tlim=None):
    plt.figure(figsize=(10,4))
    plt.plot(t, x)
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    if tlim:
        plt.xlim(tlim)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    plt.show()   # ← Mostrar la gráfica

def plot_freq(x, Fs, title, fname, xlim=None):
    N = len(x)
    X = fft(x)
    freqs = fftfreq(N, 1/Fs)
    Xmag = np.abs(X)/N

    idx = freqs >= 0

    plt.figure(figsize=(10,4))
    plt.plot(freqs[idx], 2*Xmag[idx])
    plt.title(title)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    if xlim:
        plt.xlim(xlim)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    plt.show()   # ← Mostrar la gráfica

# -------------------------
# Ruido
# -------------------------
def add_awgn(x, SNR_dB):
    Ps = np.mean(x**2)
    Pn = Ps / (10**(SNR_dB/10))
    noise = np.sqrt(Pn) * np.random.randn(len(x))
    return x + noise

# -------------------------
# Distorsión
# -------------------------
def soft_clip(x, level):
    return np.tanh(x / level)

# -------------------------
# Demodulación
# -------------------------
def envelope_detector(x, Fs):
    rect = np.abs(x)
    b = signal.firwin(101, cutoff=20, fs=Fs)
    y = signal.lfilter(b, [1], rect)
    return y

# -------------------------
# Graficar señales limpias
# -------------------------
plot_time(m_t, t, "Señal Mensaje m(t)", "mensaje_tiempo.png", tlim=(0, 0.5))
plot_time(c_t, t, "Portadora c(t)", "portadora_tiempo.png", tlim=(0, 0.05))
plot_time(s_am, t, "Señal AM modulada", "am_tiempo.png", tlim=(0, 0.1))

plot_freq(m_t, Fs, "Espectro mensaje", "mensaje_freq.png", xlim=(0, 50))
plot_freq(s_am, Fs, "Espectro AM", "am_freq.png", xlim=(0, 300))

# -------------------------
# Señal con ruido
# -------------------------
SNR = 10
s_ruido = add_awgn(s_am, SNR)

plot_time(s_ruido, t, f"AM con ruido (SNR={SNR} dB)", "am_ruido_tiempo.png", tlim=(0, 0.1))
plot_freq(s_ruido, Fs, "Espectro AM con ruido", "am_ruido_freq.png", xlim=(0, 300))

# -------------------------
# Señal distorsionada
# -------------------------
s_dist = soft_clip(s_am, 0.3)

plot_time(s_dist, t, "AM con distorsión", "am_dist_tiempo.png", tlim=(0, 0.1))
plot_freq(s_dist, Fs, "Espectro AM distorsionada", "am_dist_freq.png", xlim=(0, 500))

# -------------------------
# Señal atenuada
# -------------------------
atten = 0.4
s_atten = atten * s_am

plot_time(s_atten, t, "AM atenuada", "am_atenuada_tiempo.png", tlim=(0, 0.1))
plot_freq(s_atten, Fs, "Espectro AM atenuada", "am_atenuada_freq.png", xlim=(0, 300))

# -------------------------
# Demodulación de señales
# -------------------------
rec_clean = envelope_detector(s_am, Fs)
rec_noise = envelope_detector(s_ruido, Fs)
rec_dist  = envelope_detector(s_dist, Fs)
rec_att   = envelope_detector(s_atten, Fs)

# -------------------------
# Graficar señales recuperadas
# -------------------------
zoom = (t >= 0.2) & (t <= 0.6)

plot_time(m_t[zoom], t[zoom], "Mensaje original (zoom)", "mensaje_zoom.png")
plot_time(rec_clean[zoom], t[zoom], "Recuperada (limpia)", "recuperada_limpia.png")
plot_time(rec_noise[zoom], t[zoom], "Recuperada (ruido)", "recuperada_ruido.png")
plot_time(rec_dist[zoom], t[zoom], "Recuperada (distorsión)", "recuperada_distorsion.png")
plot_time(rec_att[zoom], t[zoom], "Recuperada (atenuada)", "recuperada_atenuada.png")

print("Simulación completada. Gráficas mostradas y guardadas en /figs/")
