# Importamos librerías necesarias
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

# Modelos Avanzados (Se maneja excepción si no están instalados)
try:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
except ImportError:
    ETSModel = None

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

# Creamos la aplicación Flask
app = Flask(__name__)

# Función helper para generar fechas proyecctadas
def generar_fechas_futuras(ultima_fecha_str, n_futuro):
    fechas = []
    try:
        current = datetime.strptime(str(ultima_fecha_str).strip(), "%Y-%m")
        for _ in range(n_futuro):
            current += relativedelta(months=1)
            fechas.append(current.strftime("%Y-%m"))
    except Exception as e:
        for i in range(n_futuro):
            fechas.append(f"Futuro {i+1}")
    return fechas

class PromedioMovil:
    def __init__(self, N):
        self.N = N
        self.name = "Promedio Móvil"
        self.color = "#ff8c00" # Naranja oscuro
        
    def fit(self, serie):
        self.serie = serie.copy()
        self.ajustados = self.serie.rolling(window=self.N).mean().shift()
        
    def predict(self, n_futuro):
        # El promedio móvil se restringe a solo 1 mes de pronóstico a futuro
        p_futuro_raw = self.serie.iloc[-self.N:].mean()
        futuros = [p_futuro_raw]
        if n_futuro > 1:
            futuros.extend([None] * (n_futuro - 1))
        return self.ajustados, futuros

class SuavizacionExponencial:
    def __init__(self):
        self.name = "Suavización Exponencial"
        self.color = "#2e8b57" # Verde claro/oscuro
        self.model_ts = None
        self.fit_results = None
        
    def fit(self, serie):
        self.serie = serie.copy()
        if ETSModel is not None:
            try:
                # El patrón de las ventas reales es un zigzag constante (sube y baja cada mes).
                # Para que la suavización exponencial lo capte, le agregamos estacionalidad de periodo = 2.
                self.model_ts = ETSModel(self.serie.astype(float), error='add', trend='add', seasonal='add', seasonal_periods=2)
                self.fit_results = self.model_ts.fit(disp=False, maxiter=10000)
                self.ajustados = self.fit_results.fittedvalues
            except Exception as e:
                print("Error ETS:", e)
                self.ajustados = pd.Series([None]*len(self.serie))
        else:
            self.ajustados = pd.Series([None]*len(self.serie))
            
    def predict(self, n_futuro):
        if self.fit_results is not None:
            try:
                futuros = self.fit_results.forecast(steps=n_futuro).tolist()
            except:
                futuros = [None] * n_futuro
        else:
            futuros = [None] * n_futuro
        return self.ajustados, futuros

class ModeloProphet:
    def __init__(self):
        self.name = "Prophet"
        self.color = "#8a2be2" # Morado
        self.m = None
        
    def fit(self, df_raw, producto):
        df_train = df_raw[["Mes", producto]].copy()
        df_train.columns = ["ds", "y"]
        
        # Prevenir que números simples (1, 2, 3...) se interpreten como nanosegundos
        is_numeric = pd.to_numeric(df_train["ds"], errors='coerce').notna().all()
        if is_numeric:
            df_train["ds"] = [datetime(2020, 1, 1) + relativedelta(months=i) for i in range(len(df_train))]
        else:
            df_train["ds"] = pd.to_datetime(df_train["ds"], errors='coerce')
        
        self.longitud = len(df_train)
        
        if Prophet is not None and not df_train['ds'].isna().all():
            try:
                # Añadir componente estacional artificial de "2 meses" (~60.8 días) para atrapar el zigzag
                self.m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False, changepoint_prior_scale=0.8)
                self.m.add_seasonality(name='zigzag_bimensual', period=60.875, fourier_order=5)
                self.m.fit(df_train)
                predict_train = self.m.predict(df_train)
                self.ajustados = predict_train['yhat']
            except Exception as e:
                print("Error Prophet:", e)
                self.ajustados = pd.Series([None]*self.longitud)
        else:
            self.ajustados = pd.Series([None]*self.longitud)
            
    def predict(self, n_futuro):
        if self.m is not None:
            try:
                future = self.m.make_future_dataframe(periods=n_futuro, freq='MS')
                forecast = self.m.predict(future)
                futuros = forecast['yhat'].iloc[-n_futuro:].tolist()
            except:
                futuros = [None] * n_futuro
        else:
            futuros = [None] * n_futuro
        return self.ajustados, futuros

def evaluar_modelo(modelo, df, producto, n_futuro, n_hist):
    valores_reales = df[producto].tolist()
    
    # 1. Aplicamos Selección de Modelo, FIT y PREDICT internamente como se requirió
    if isinstance(modelo, ModeloProphet):
        modelo.fit(df, producto)
        ajustados, futuros = modelo.predict(n_futuro)
    else:
        modelo.fit(df[producto])
        ajustados, futuros = modelo.predict(n_futuro)

    ajustados_list = [x if not pd.isna(x) else None for x in ajustados]
    
    # Calcular errores
    df_error = pd.DataFrame({
        "Real": valores_reales,
        "Pronostico": ajustados_list
    }).replace({np.nan: None}).dropna()
    
    if len(df_error) > 0:
        error = df_error["Pronostico"] - df_error["Real"]
        error_abs = abs(error)
        ape = (error_abs / df_error["Real"]).replace([np.inf, -np.inf], np.nan).dropna()
        MYPE = ape.mean()
        
        ape_p = (error_abs / df_error["Pronostico"]).replace([np.inf, -np.inf], np.nan).dropna()
        MYPE_p = ape_p.mean()
        
        MSE = (error**2).mean()
        RMSE = np.sqrt(MSE)
    else:
        MYPE = MYPE_p = MSE = RMSE = 0

    # Construimos la línea de tiempo completa validando formato entero para evitar .0
    linea_completa = []
    for item in (ajustados_list + futuros):
        try:
             # Solo metemos números enteros a la linea de proyección
             linea_completa.append(int(round(item)) if item is not None and not pd.isna(item) else None)
        except:
             linea_completa.append(None)

    return {
        "nombre": modelo.name,
        "color": modelo.color,
        "MYPE": round(MYPE, 4) if not pd.isna(MYPE) else 0,
        "MYPEp": round(MYPE_p, 4) if not pd.isna(MYPE_p) else 0,
        "MSE": round(MSE, 2) if not pd.isna(MSE) else 0,
        "RMSE": round(RMSE, 2) if not pd.isna(RMSE) else 0,
        "linea_tiempo": linea_completa
    }

@app.route("/", methods=["GET", "POST"])
def index():
    resultados = None
    
    # Solo mostrar advertencia si no están instaladas las dependencias
    librerias_faltantes = []
    if ETSModel is None: librerias_faltantes.append("statsmodels")
    if Prophet is None: librerias_faltantes.append("prophet")
    
    if request.method == "POST":
        try:
            N = int(request.form.get("n", 3))
            n_futuro = int(request.form.get("n_futuro", 1))
            metodo = request.form.get("metodo", "Promedio Movil")
            archivo_csv = request.files.get("csv_file")
            
            if archivo_csv and archivo_csv.filename.endswith('.csv'):
                resultados = {}
                df = pd.read_csv(archivo_csv)
                
                df = df.replace({np.nan: None})
                fechas_hist = df["Mes"].tolist()
                fechas_fut = generar_fechas_futuras(fechas_hist[-1], n_futuro)
                fechas_totales = fechas_hist + fechas_fut
                
                productos = [col for col in df.columns if col.startswith("Producto_")]
                
                for prod in productos:
                    valores_reales = df[prod].tolist()
                    reales_extendidos = valores_reales + [None]*n_futuro
                    
                    modelos_a_evaluar = []
                    if metodo == "Promedio Movil":
                        modelos_a_evaluar.append(PromedioMovil(N))
                    elif metodo == "Suavizacion Exponencial":
                        modelos_a_evaluar.append(SuavizacionExponencial())
                    elif metodo == "Prophet":
                        modelos_a_evaluar.append(ModeloProphet())
                        
                    res_modelos = []
                    for mod in modelos_a_evaluar:
                        rmod = evaluar_modelo(mod, df, prod, n_futuro, N)
                        res_modelos.append(rmod)
                        
                    resultados[prod] = {
                        "producto": prod.replace("Producto_", "Producto "),
                        "fechas": fechas_totales,
                        "reales": [int(round(r)) if r is not None and not pd.isna(r) else None for r in reales_extendidos],
                        "modelos": res_modelos
                    }
        except Exception as e:
            print("ERROR INTERNO:", e)
            pass

    return render_template("pronostico.html", resultados=resultados, faltantes=librerias_faltantes)

if __name__ == "__main__":
    app.run(debug=True)
