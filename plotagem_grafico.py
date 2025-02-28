import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Função de ajuste

# cte: constate inicial, a: amplitude, b: fator de amortecimento, w: freq angular, phi: fase

def oscilacao_amortecida(t, cte, a, b, w, phi):
    # Equação da oscilador amortecido 
    return cte + a * np.exp(-b * t) * np.cos(w * t + phi)

# Carrega os dados para .txt em output
TEMPOS_FILE = "./output/tempos.txt"
ESPACOS_FILE = "./output/espacos.txt"
try:
    x_tempo = np.loadtxt(TEMPOS_FILE, dtype=float)
    y_pos = np.loadtxt(ESPACOS_FILE, dtype=float)
except Exception as e:
    print(f"Erro no carregamento dos dados: {e}")
    exit()

# Plota os dados experimentais
plt.scatter(x_tempo, y_pos, color="orange", label="Dados experimentais")

# Chute inicial para os parâmetros do ajuste
palpite = [20, 10, 0.01, (2 * np.pi)/1.3, 0]

# Ajustando a curva
try:
    popt, pcov = curve_fit(oscilacao_amortecida, x_tempo, y_pos, p0=palpite)
except Exception as e:
    print(f"Erro no ajuste da curva: {e}")
    exit()

print(f"cte: {popt[0]:.3f}")
print(f"a: {popt[1]:.3f}")
print(f"b: {popt[2]:.6f}")
print(f"w: {popt[3]:.3f}")
print(f"phi: {popt[4]:.3f}")

# Extraindo os parâmetros ajustados
cte, a, b, w, phi = popt
w0 = np.sqrt(w**2 + b**2)  # Freqüência angular natural
m = 0.290  # Massa do pendulo em kg
q = w0 / (2 * b)  # Fator de qualidade

# Salvando o fator de qualidade
with open("./output/fator_qualidade.txt", "w") as f:
    f.write(f"{q}")

# Gerando os dados ajustados
x_fit = np.linspace(min(x_tempo), max(x_tempo), 1000)
y_fit = oscilacao_amortecida(x_fit, *popt)

# Plotando a curva ajustada
plt.plot(x_fit, y_fit, color="blue", label="Ajuste da curva")

# Legenda do gráfico
plt.xlabel("Tempo (s)")
plt.ylabel("Posição (cm)")
plt.title("Ajuste de Oscilação Amortecida")
plt.legend()
plt.grid(True)
plt.show()
