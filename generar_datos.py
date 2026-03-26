import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generar 70 meses de datos (casi 6 años)
np.random.seed(42)
fechas = pd.date_range(start='2018-01-01', periods=70, freq='MS')
meses = fechas.strftime('%Y-%m')

# Datos aleatorios con cierta tendencia y estacionalidad para 5 productos
datos = {
    "Mes": meses,
    "Producto_A": np.round(np.linspace(100, 200, 70) + np.random.normal(0, 10, 70)),
    "Producto_B": np.round(np.linspace(150, 100, 70) + np.random.normal(0, 15, 70)),
    "Producto_C": np.round(50 + 20 * np.sin(np.arange(70) / 3) + np.random.normal(0, 5, 70)),
    "Producto_D": np.round(np.random.randint(200, 300, 70)),
    "Producto_E": np.round(np.linspace(50, 250, 70) + 15 * np.cos(np.arange(70) / 4) + np.random.normal(0, 8, 70))
}

df = pd.DataFrame(datos)
df.to_csv("ventas_prueba.csv", index=False)
print("Archivo ventas_prueba.csv generado exitosamente con", len(df), "registros.")
