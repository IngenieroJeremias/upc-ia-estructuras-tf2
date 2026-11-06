# Predicción de respuesta sísmica con IA — UPC T1

**Curso:** Aplicaciones de IA en Estructuras  
**Docente:** Ing. Kurt Soncco Sinchi  
**Integrantes:**
Franklin Fustamante Sánchez e201568658  
Carlos David Vargas Marin e202527282
Abrahán Ramirez More e202527900
Jheremy Richard Molleapaza Infantes 

## Objetivo
Predecir el **desplazamiento espectral elástico Sd (cm)** a T=0.80 s y 5% de amortiguamiento, para un oscilador SDOF, a partir de parámetros sísmicos de PEER NGA-West2. No equivale a deriva ni desplazamiento real de una edificación.

## Fuente de datos
- https://peer.berkeley.edu/node/74 (NGA-West2 flatfiles)
- https://ngawest2.berkeley.edu/

**Estado actual (T1):** fuente identificada y descarga pendiente. No se reportan datos descargados, métricas, ni modelos entrenados. Una vez obtenido el flatfile, documentaremos número exacto de filas, columnas, unidades y faltantes.

## Variables propuestas
- Predictoras: Mw, Rrup o Rjb, Vs30 (sujetas a disponibilidad).
- Objetivo: Sd(cm), obtenido de Sa(T=0.8s,5%) mediante Sd = Sa(m/s²) × (T/(2π))² × 100. Si Sa está en g: multiplicar previamente por 9.80665.
- El identificador del evento se usa para separar entrenamiento/prueba sin compartir sismos.

## Plan de trabajo
**T1:** definición, fuente, documentación y marco PCS.
**T2:** descarga, diccionario y limpieza, EDA en Python.
**T3:** línea base, regresión lineal, Random Forest; MAE, RMSE, R² y pruebas de estabilidad.

## Estructura
- `docs/`: informe y exposición
- `data/`: instrucciones de adquisición, sin datos privados o no autorizados
- `src/`: código inicial en Python
- `requirements.txt`: dependencias

## Inicio rápido (opcional; después de obtener datos)
1. Instalar Python.
2. En una terminal ejecutar `pip install -r requirements.txt`.
3. Ejecutar `python src/preparar_datos.py` para consultar instrucciones.

## Integridad académica
El informe contiene una bitácora del apoyo de ChatGPT. El equipo verificará datos, fuentes, código y conclusiones; no se atribuyen mediciones o resultados inexistentes.

## Referencias
Yu, B., & Barter, R. (2024). *Veridical Data Science*. MIT Press. https://vdsbook.com/
Yu, B., & Kumbier, K. (2020). *Veridical data science*. PNAS. https://doi.org/10.1073/pnas.1901326117
