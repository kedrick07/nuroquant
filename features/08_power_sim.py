import numpy as np, pandas as pd
from config import RESULTS

T, N, S, B, L = 239, 30, 300, 500, 4
sd = 1 / np.sqrt(N - 1)
rng = np.random.default_rng(7)
nb = int(np.ceil(T / L))
starts = rng.integers(0, T, size=(B, nb))
idx = ((starts[:, :, None] + np.arange(L)) % T).reshape(B, -1)[:, :T]
rows = []
for rho in (0.7, 0.8, 0.9):
    sd_d = sd * np.sqrt(2 * (1 - rho))
    for delta in (0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040):
        d = rng.normal(delta, sd_d, size=(S, T))
        p = np.empty(S)
        for s in range(S):
            m = d[s].mean()
            p[s] = (np.abs((d[s] - m)[idx].mean(axis=1)) >= abs(m)).mean()
        rows.append((rho, delta, (p < 0.01).mean(), (p < 0.05).mean()))
df = pd.DataFrame(rows, columns=["rho", "delta", "power_a01", "power_a05"])
RESULTS.mkdir(exist_ok=True)
df.to_csv(RESULTS / "power_simulation.csv", index=False)
print(df.pivot(index="delta", columns="rho", values="power_a01").round(2))
print(df.pivot(index="delta", columns="rho", values="power_a05").round(2))