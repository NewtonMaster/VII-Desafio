import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parâmetros da simulação
L = 1.0          # Comprimento do domínio (m)
T = 0.01         # Tempo total de simulação (s)
c = 343.0        # Velocidade do som no ar (m/s)
Nx = 200         # Número de pontos espaciais
Nt = 400         # Número de pontos temporais

# Discretização
dx = L / (Nx - 1)
dt = T / Nt
x = np.linspace(0, L, Nx)

# Garantir estabilidade
assert c * dt / dx <= 1, "Critério de estabilidade não satisfeito!"

# Inicialização das variáveis
p = np.zeros((Nt, Nx))  # Pressão acústica
p_new = np.zeros(Nx)

# Condições iniciais: pulso gaussiano
p[0, Nx//2] = 1

# Simulação
for n in range(0, Nt-1):
    for i in range(1, Nx-1):
        p_new[i] = 2 * p[n, i] - p[n-1, i] + (c * dt / dx)**2 * (p[n, i+1] - 2 * p[n, i] + p[n, i-1])
    p[n+1, :] = p_new

# Animação dos resultados
fig, ax = plt.subplots()
line, = ax.plot(x, p[0, :], lw=2)
ax.set_ylim(-1, 1)
ax.set_xlabel('Distância (m)')
ax.set_ylabel('Pressão acústica')

def update(frame):
    line.set_ydata(p[frame, :])
    return line,

ani = FuncAnimation(fig, update, frames=range(Nt), blit=True)
plt.show()
