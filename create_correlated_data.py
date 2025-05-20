# Salva questo come create_correlated_data.py nella cartella principale del progetto
import pandas as pd
import numpy as np
from pathlib import Path

# Imposta il seed per riproducibilità
np.random.seed(42)

# Numero di righe
n = 100

# Genera join_key e timestamp
join_key = np.arange(1, n+1)
timestamp = pd.date_range(start='2025-01-01', periods=n)

# Genera valori per speed (KPI)
speed = 100 + 15 * np.random.randn(n)

# Genera temperature correlata negativamente con speed (correlazione circa -0.7)
# Formula: temperature = 20 - 0.1*speed + rumore
temperature = 20 - 0.1 * speed + 5 * np.random.randn(n)

# Genera pressure con correlazione debole/moderata positiva (circa 0.3)
pressure = 1 + 0.02 * speed + 0.08 * np.random.randn(n)

# Crea i DataFrame
df_speed = pd.DataFrame({
    'join_key': join_key,
    'timestamp': timestamp,
    'value_speed': speed
})

df_temperature = pd.DataFrame({
    'join_key': join_key,
    'timestamp': timestamp,
    'value_temperature': temperature
})

df_pressure = pd.DataFrame({
    'join_key': join_key,
    'timestamp': timestamp,
    'value_pressure': pressure
})

# Crea il dataset unificato
df_unified = pd.DataFrame({
    'join_key': join_key,
    'timestamp': timestamp.astype(str),  # Converti a stringa per evitare problemi
    'value_pressure': pressure,
    'value_speed': speed,
    'value_temperature': temperature
})

# Directory output
output_dir = Path('examples/driver_csvs')
output_dir.mkdir(exist_ok=True, parents=True)

# Salva i file
df_speed.to_csv(output_dir / 'speed.csv', index=False)
df_temperature.to_csv(output_dir / 'temperature.csv', index=False)
df_pressure.to_csv(output_dir / 'pressure.csv', index=False)
df_unified.to_csv(output_dir / 'unified_dataset.csv', index=False)

# Verifica la correlazione
print("Correlazione tra speed e temperature:", np.corrcoef(speed, temperature)[0, 1])
print("Correlazione tra speed e pressure:", np.corrcoef(speed, pressure)[0, 1])