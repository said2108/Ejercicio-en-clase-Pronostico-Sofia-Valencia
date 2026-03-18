# Importamos librerías necesarias
from flask import Flask, render_template, request #Flak para crear la aplicación web, renderizar plantillas HTML y manejar solicitudes HTTP
import pandas as pd #Manejo de datos tipo tabla
import numpy as np #Operaciones matemáticas y estadísticas
import matplotlib.pyplot as plt #Creación de gráficos
import base64 #Codificación de datos en formato base64 para incluir gráficos en HTML
from io import BytesIO #Manejo de flujos de bytes para guardar gráficos en memoria


# Creamos la aplicación Flask
app = Flask(__name__)

# Cargamos los datos de ventas históricas desde un archivo CSV
datos = pd.read_csv("ventas_historicas.csv")

# Función para calcular el pronóstico utilizando el método de media móvil
def pronostico(producto, N):

    df = datos.copy() #Creamos una copia del DataFrame original para trabajar con los datos sin modificar el original

    df["Pronostico"] = df[producto].rolling(window=N).mean().shift() #Calculamos el pronóstico utilizando el promedio movil de las ventas históricas con ventana de N periodos y desplazando 1 periodo hacia adelante para obtener el pronóstico del siguiente periodo 
    df["error"] = df["Pronostico"] - df[producto] # Error = Pronóstico - valor real
    df["error_abs"] = abs(df["error"])
    df["ape"] = df["error_abs"] / df[producto] #Error absoluto porcentual
    df["ape_prima"] = df["error_abs"] / df["Pronostico"] #Error absoluto porcentual con respecto al pronóstico
    df["error_cuadratico"] = df["error"] ** 2

    # Calculamos las medidas de error para evaluar el pronóstico
    MYPE = df["ape"].mean()
    MYPE_prima = df["ape_prima"].mean()
    MSE = df["error_cuadratico"].mean()
    RMSE = np.sqrt(MSE)

    # último valor del pronóstico (el que me interesa)
    pron_final = df["Pronostico"].iloc[-1]

    return df, pron_final, MYPE, MYPE_prima, MSE, RMSE


# Función para crear el gráfico de ventas y pronóstico
def crear_grafico(df, producto): #Creamos un gráfico de líneas que muestra las ventas históricas y el pronóstico para el producto seleccionado

    plt.figure() 
    
    # linea de ventas reales
    plt.plot(df["Mes"], df[producto], label="Ventas")
    
    # Linea de pronóstico
    plt.plot(df["Mes"], df["Pronostico"], label="Pronóstico")

    plt.legend() #Muestra la leyenda (Ventas vs Pronóstico)
    plt.title(producto) #Título del gráfico con el nombre del producto

    # Guardar imagen en memoria
    buffer = BytesIO()
    plt.savefig(buffer, format="png")

    buffer.seek(0) # Volver al inicio del buffer

    # Convertir imagen a base64 (texto)
    grafico = base64.b64encode(buffer.getvalue()).decode()

    plt.close()

    return grafico


# Ruta principal de la aplicación web, que maneja tanto solicitudes GET como POST
@app.route("/", methods=["GET", "POST"])
def index():

    # Variables para almacenar el resultado del pronóstico y la gráfica
    resultado = None
    grafica = None

    # Si el usuario presiona el botón
    if request.method == "POST":

        # Obtenemos el producto seleccionado desde el formulario HTML
        producto = request.form["producto"]

        # Agregamos el prefijo "Producto_" al nombre del producto para que coincida con el formato de las columnas en el DataFrame
        producto = "Producto_" + producto

        # Obtenemos el valor de N desde el formulario HTML
        N = int(request.form["n"])

        # Llamamos a la función de pronóstico
        df, pron, MYPE, MYPEp, MSE, RMSE = pronostico(producto, N)

        # Creamos el gráfico
        grafica = crear_grafico(df, producto)

        # Guardamos los resultados
        resultado = {
            "producto": producto,
            "pronostico": round(pron, 2),
            "MYPE": round(MYPE, 4),
            "MYPEp": round(MYPEp, 4),
            "MSE": round(MSE, 2),
            "RMSE": round(RMSE, 2)
        }

    # Renderizamos la página HTML
    return render_template("pronostico.html", resultado=resultado, grafica=grafica)


# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(debug=True)