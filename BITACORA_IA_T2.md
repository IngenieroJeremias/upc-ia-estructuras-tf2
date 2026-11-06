# Bitácora de uso de IA — Entrega T2

**Proyecto:** Predicción del desplazamiento espectral elástico Sd(T = 0.80 s, 5 %) con PEER NGA-West2
**Curso:** Aplicaciones de IA en Estructuras — UPC · Grupo 06
**Notebook:** [`src/eda_nga_west2.ipynb`](../src/eda_nga_west2.ipynb) (Tramos 1–7)
**Asistente de IA:** Claude Code (Anthropic), integrado en VS Code
**Entorno:** Python 3.12 · `.venv` creado con uv 0.13.7 · numpy 2.5.4 · pandas 3.1.1 · scikit-learn 1.9.1 · matplotlib 3.11.2 · seaborn 0.13.2 · scipy 1.18.1

> **Advertencia de alcance.** Todos los resultados de esta bitácora se obtuvieron con un
> **dataset representativo sintético** (1514 registros, 120 sismos) que reproduce la
> estructura de NGA-West2: varios registros por evento, Vs30 con nulos, Z1.0 según
> Chiou & Youngs (2014) y Sa(0.8 s) con forma tipo GMPE. **No son métricas de datos
> reales de PEER.** El flujo está preparado para reemplazar el bloque de generación por
> `pd.read_csv(<flatfile NGA-West2>)` sin cambiar el resto.

---

## Esquema de cada tramo

Cada sección sigue el formato de la clase:

1. **Prompt inicial.** Anatomía Rol – Contexto – Tarea – Restricciones.
2. **Respuesta de la IA.** Qué entregó el asistente y qué resultados produjo.
3. **Auditoría crítica del ingeniero.** Qué se revisó y qué problemas se encontraron.
4. **Corrección / refinamiento.** Qué se cambió y qué queda pendiente.

| # | Tramo de la clase | Celdas del notebook |
|---|---|---|
| 1 | Geotecnia y target físico Sd | Tramo 2 + paneles geotécnicos del Tramo 4 |
| 2 | Partición GroupKFold sin fuga | Tramo 3 + panel de auditoría del Tramo 4 |
| 3 | Benchmarking supervisado | Tramo 5 + modelo MLP del Tramo 6 |
| 4 | xAI y cierre | Permutation Importance del Tramo 6 + Tramo 7 |

---

## 1. Geotecnia y transformación física del target (Tramo 2)

### 1.1 Prompt inicial

| Componente | Contenido |
|---|---|
| **Rol** | Ingeniero Estructural y Científico de Datos experto en ML para dinámica de estructuras. |
| **Contexto** | Proyecto de predicción de demanda sísmica con PEER NGA-West2. El notebook ya tiene el entorno `.venv` (Python 3.12). |
| **Tarea** | Escribir y ejecutar la celda "Tramo 2: Carga, Limpieza Geotécnica y Transformación Física del Target Sd". |
| **Restricciones** | (1) Dataset representativo con `EQID, Mw, Rrup_km, Vs30_ms` (con nulos), `Z10_m, Mechanism, Sa_08s_g`. (2) Imputar Vs30 con la mediana agrupada por `Mechanism`. (3) Sa_ms2 = Sa_g · 9.80665 y Sd_08s_cm = Sa_ms2 · (T/2π)² · 100. (4) One-hot de `Mechanism`. (5) Reporte de nulos antes/después y `head(3)`. |

### 1.2 Respuesta de la IA

- **Dataset generado:** 120 eventos, con más registros por evento a mayor Mw (1514 en total).
  - Vs30 lognormal (mediana ≈ 400 m/s), con un 11.03 % de nulos (167 registros).
  - Z1.0 derivado de Vs30 con la relación de Chiou & Youngs (2014).
  - Sa(0.8 s) con escalamiento en Mw y Mw², atenuación ln√(R² + h²), término de sitio en ln Vs30, término de mecanismo y residuos inter-evento (σ = 0.35) e intra-evento (σ = 0.55).
- **Imputación:** se usó la mediana por mecanismo (Normal 384.5 · Normal-Oblique 442.6 · Reverse 401.9 · Reverse-Oblique 381.0 · Strike-Slip 413.0 m/s). No quedan nulos después de imputar.
- **Columna adicional `Vs30_imputado` (0/1):** registra qué filas se imputaron.
- **Comprobación de la transformación:** fila 0, Sa = 0.02405 g → 0.2359 m/s² → **Sd = 0.382 cm**. Coincide con el cálculo manual.
- **Tabla final:** `df_proc` con 1514 × 14 columnas (5 dummies `Mech_*`).

### 1.3 Auditoría crítica del ingeniero

| Hallazgo | Riesgo | Severidad |
|---|---|---|
| Los datos son sintéticos y no vienen del flatfile de PEER. | Cualquier conclusión sobre importancia física se cumple por construcción. | Alta (alcance) |
| La mediana de Vs30 se calcula con **todo** el dataset, antes de partir en train/test. | Fuga leve de información de test hacia train. | Media |
| La imputación por mediana colapsa 167 sitios en 5 valores de Vs30 (visible en el Tramo 4, panel 0,1). | Se pierde variabilidad de sitio y el modelo puede aprender el valor imputado. | Media |
| `Vs30_imputado` entra como feature en el Tramo 3. | Sesgo si en los datos reales los nulos no son aleatorios. | Baja |
| Recorte de Rrup a 300 km al generar los datos. | Artefacto: una columna vertical de puntos en R = 300 km en el EDA. | Baja (solo afecta a los datos sintéticos) |
| Sd tiene una cola larga a la derecha (mediana 0.86 cm, máximo 32.3 cm). | Pearson y las métricas en cm quedan dominados por pocos registros. | Media |

### 1.4 Corrección / refinamiento

- **Ya aplicado:**
  - Columna de trazabilidad `Vs30_imputado` y `assert` de nulos = 0.
  - Comprobación numérica de la fórmula de Sd.
  - El target se modela en ln(Sd) desde el Tramo 5 (ver §3).
- **Pendiente para T3:**
  - Mover la imputación a un `SimpleImputer` agrupado **dentro del Pipeline**, para que la mediana se ajuste solo con train en cada fold.
  - Repetir el tramo con el flatfile real.
  - Evaluar imputar Vs30 con un proxy (pendiente topográfica, geología) en vez de la mediana por mecanismo.

---

## 2. Partición robusta sin fuga: GroupKFold por EQID (Tramos 3 y 4)

### 2.1 Prompt inicial

| Componente | Contenido |
|---|---|
| **Rol** | Ingeniero Estructural y Científico de Datos experto en ML para ingeniería sísmica. |
| **Contexto** | `df_proc` ya procesado, con el target `Sd_08s_cm` y la columna de grupo `EQID`. |
| **Tarea** | "Tramo 3: Partición Robusta sin Fuga de Datos (GroupKFold por EQID)". |
| **Restricciones** | (1) X sin `EQID, Sa_08s_g, Sa_ms2, Sd_08s_cm`. (2) y = Sd, groups = EQID. (3) `GroupKFold(n_splits=5)`. (4) Primer fold → X/y train/test. (5) Intersección de EQID train ∩ test y mensaje explícito de fuga. (6) Dimensiones finales. |

Prompt complementario (Tramo 4, EDA). **Rol:** experto en visualización científica. **Tarea:** figura de 2×2 (16×12). **Restricciones:** estilo `whitegrid`, paletas `viridis`/`coolwarm`; paneles de atenuación log-log, efecto de sitio con medidos vs. imputados y límites de 360/760 m/s, heatmap de Pearson con máscara triangular y auditoría train/test en el plano Mw–Rrup.

### 2.2 Respuesta de la IA

- **Features (10):** `Mw, Rrup_km, Vs30_ms, Z10_m, Vs30_imputado` y 5 `Mech_*`.
- **Fold 1:** 96 sismos en train y 24 en test. **Intersección = 0** → "✅ SIN FUGA DE INFORMACIÓN".
- **Dimensiones:** `X_train` (1211, 10) y `X_test` (303, 10). Test es el 20.0 % de los registros.
- **Comprobación extra:** `assert` de que los **5 folds** son disjuntos por EQID.
- **Hallazgos del EDA (Tramo 4):**
  - Ajuste log-log Sd ∝ R^−1.13, con saturación de la mediana por debajo de unos 8 km.
  - Correlaciones de Pearson de Sd: r = +0.32 con Mw y −0.37 con Rrup.
  - Correlación Vs30–Z1.0: **r = −0.73**.
  - Test cubre Mw 4.58–7.54 (train: 4.53–7.82) y todo el rango de distancias.

### 2.3 Auditoría crítica del ingeniero

- **Intersección = 0:** es correcta, pero es una condición necesaria y no suficiente para descartar fuga. La fuga por imputación (§1.3) sigue presente.
- **Un solo fold con 24 sismos:** la estimación del error tiene alta varianza.
- **Correlación con Sd sin transformar (Pearson):** subestima relaciones multiplicativas. Por ejemplo, Vs30–Sd da r = −0.13 aunque el efecto de sitio existe.
- **Vs30–Z1.0 (r = −0.73):** apunta a colinealidad entre las variables de sitio y anticipa importancias inestables para Z1.0.
- **Gráficos rechazados:** se descartó un gráfico con dos ejes Y (RMSE y R² en el mismo eje). Se usaron ejes separados.

### 2.4 Corrección / refinamiento

- **Ya aplicado:**
  - Leyendas con fondo blanco para que no tapen los datos.
  - Títulos acortados para que no se solapen.
  - Eje de Vs30 con números legibles (150, 200, 300… m/s).
  - Clasificación NEHRP D/E (180 m/s), C/D (360 m/s) y B/C (760 m/s).
- **Pendiente para T3:**
  - Reportar las métricas como media ± desviación en los 5 folds.
  - Calcular la correlación de Pearson en escala log (ln Sd, ln R, ln Vs30).
  - Considerar `StratifiedGroupKFold` por tramos de Mw si los folds reales quedan desbalanceados.

---

## 3. Benchmarking supervisado (Tramos 5 y 6)

### 3.1 Prompt inicial

| Componente | Contenido |
|---|---|
| **Rol** | Ingeniero Estructural y Científico de Datos experto en ML y dinámica de estructuras. |
| **Contexto** | X/y train/test ya particionados con GroupKFold. |
| **Tarea** | Tramo 5: benchmark de Ridge (baseline que emula una GMM) vs. `RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)`. Tramo 6: `MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42, early_stopping=True)` dentro de `TransformedTargetRegressor(func=np.log, inverse_func=np.exp)` con `prep_rf`. |
| **Restricciones** | StandardScaler ajustado solo en train; MAE, RMSE (cm) y R² en test; gráfico real vs. predicho con línea 1:1; importancia de variables del RF; tabla resumen. |

### 3.2 Respuesta de la IA

**Decisiones técnicas que propuso la IA** (no estaban en el prompt):

1. **Target en ln(Sd)** mediante `TransformedTargetRegressor`, como en cualquier GMM (la variabilidad sísmica es lognormal). Las métricas se siguen reportando en cm.
2. **Ridge con variables tipo GMM:** Mw, Mw², ln√(R² + 6²), ln Vs30, ln Z1.0 y los dummies de mecanismo. Estas transformaciones no tienen parámetros que aprender, así que no introducen fuga.
3. **Métrica adicional R² (ln Sd)**, que es la escala en la que se evalúan las GMM.

**Resultados en test (Fold 1, 24 sismos no vistos):**

| Rank | Modelo | MAE (cm) | RMSE (cm) | R² (cm) | R² (ln Sd) |
|---|---|---|---|---|---|
| 1 | **Random Forest** | **1.275** | **2.557** | **0.505** | **0.743** |
| 2 | Ridge (GMM lineal) | 1.371 | 2.814 | 0.400 | 0.733 |
| 3 | Red Neuronal MLP | 1.364 | 2.822 | 0.396 | 0.729 |

- **MLP:** el early stopping lo cortó en 57 épocas. En la validación interna (sacada de train) llegó a R² ln = 0.835, pero en test cae a 0.729.
- **Importancia interna del RF (MDI, train):** Rrup 0.565 · Mw 0.335 · Vs30 0.042 · Z1.0 0.038 · resto < 0.01.

### 3.3 Auditoría crítica del ingeniero

- **Dos escalas de R²:** R² en cm (0.40–0.50) y R² en ln (≈ 0.74) cuentan historias distintas. En cm, la ventaja del RF viene de los registros de alta demanda. En ln, los tres modelos son prácticamente equivalentes.
- **Techo de rendimiento:** la dispersión aleatoria total de los datos sintéticos es σ_ln ≈ 0.65 y no se puede explicar. Los tres modelos ya están cerca de ese límite, así que la diferencia RF vs. GMM (−9.2 % de RMSE) **no está probada** con un solo fold.
- **Sesgo de `exp()`:** al volver de ln a cm se predice la **mediana**, no la media. Eso sesga a la baja el error en cm.
- **MLP:** la diferencia entre validación interna (0.835) y test (0.729) indica que la validación de early stopping (aleatoria por registro) **mezcla registros del mismo sismo**. Es una fuga interna de la red.
- **Importancia MDI:** se calcula en train y favorece a las variables continuas. Por eso hace falta la auditoría con Permutation Importance en test (§4).

### 3.4 Corrección / refinamiento

- **Ya aplicado:**
  - Gráfico real vs. predicho en escala log-log, con línea 1:1 y banda de factor 2.
  - RMSE y R² en subgráficos separados, sin dos ejes Y.
  - Selección del modelo justificada en la celda de cierre (Tramo 7).
- **Pendiente para T3:**
  - Validación cruzada con los 5 folds y prueba de significancia de la diferencia RF vs. GMM.
  - Corrección de sesgo lognormal: exp(μ + σ²/2) si se requiere la media.
  - Para el MLP: validación interna agrupada por EQID en lugar de `early_stopping` aleatorio.
  - Búsqueda de hiperparámetros con `GroupKFold` anidado.

---

## 4. xAI: Permutation Importance bajo el marco PCS (Tramos 6 y 7)

### 4.1 Prompt inicial

| Componente | Contenido |
|---|---|
| **Rol** | Ingeniero Estructural y Auditor Técnico de IA. |
| **Contexto** | Benchmark Ridge / RF / MLP completado. |
| **Tarea** | `permutation_importance` del RF sobre X_test (`scoring='neg_root_mean_squared_error'`, `n_repeats=10`). Comprobar si el orden físico Mw, Rrup, Vs30 se mantiene en datos no vistos (marco PCS de Yu & Barter). Celda de cierre con la tabla final, conclusiones y figuras a 300 dpi. |
| **Restricciones** | Figura de 2 paneles (benchmark RMSE/R² + boxplot de importancia por permutación); tabla consolidada; figuras de los Tramos 4–6 guardadas en `docs/`. |

### 4.2 Respuesta de la IA

**Aumento de RMSE al permutar cada variable (test):**

| Variable | ΔRMSE (cm) | Rank en test (permutación) | Rank en train (MDI) |
|---|---|---|---|
| Rrup | 1.619 ± 0.173 | 1 | 1 |
| Mw | 0.713 ± 0.139 | 2 | 2 |
| Vs30 | 0.054 ± 0.015 | 3 | 3 |
| Z1.0 | 0.032 ± 0.061 | 4 | 4 |
| Mecanismos / flag de imputación | −0.037 a 0.004 | 5–10 | 5–10 (en otro orden) |

- **Variables físicas:** las 3 primeras de test son {Rrup, Mw, Vs30}, igual que en train → "✅ ESTABLE (PCS)".
- **Top-2:** {Rrup, Mw} aparece en las 10 repeticiones (100 %).
- **Ranking completo:** Spearman entre los rankings de train y test, ρ = 0.68. El orden cambia de la 5.ª posición hacia abajo.
- **Celda de cierre (Tramo 7):**
  - Tabla final ordenada por RMSE.
  - Conclusiones generadas a partir de los resultados, sin valores escritos a mano.
  - Comprobación de las 3 figuras en `docs/` a 300 dpi.

### 4.3 Auditoría crítica del ingeniero

- **El ranking físico es estable, pero confirma lo que se puso al generar los datos:** con datos sintéticos, esto valida que el pipeline *puede* recuperar la física; **no** valida la física de NGA-West2.
- **Z1.0 no es confiable:** su caja cruza el 0, y la permutación de Vs30 y Z1.0 por separado es engañosa porque están correlacionadas (r = −0.73).
- **`Mech_Normal` sale negativa (−0.037 cm):** permutarla *mejora* el RMSE. Es ruido por los pocos eventos normales en test, no una señal física.
- **Una sola partición:** la estabilidad se midió solo frente a permutaciones. El marco PCS exige también estabilidad frente a **perturbaciones de los datos** (otros folds, bootstrap por EQID) y del **modelo** (otras semillas).
- **Frase corregida:** en las conclusiones automáticas, la IA afirmó que el MLP era "más sensible a la semilla y a la partición" sin haberlo comprobado. El auditor la eliminó.

### 4.4 Corrección / refinamiento

- **Ya aplicado:**
  - Conclusiones con f-strings, para que se actualicen solas al cambiar los datos.
  - Eliminada la afirmación sin evidencia sobre el MLP.
  - Figuras guardadas a 300 dpi:
    - `docs/fig_tramo4_eda_fisico.png` (4865 × 3533 px)
    - `docs/fig_tramo5_benchmark_ridge_rf.png` (4636 × 1451 px)
    - `docs/fig_tramo6_benchmark_xai.png` (4116 × 1603 px)
- **Pendiente para T3:**
  - Importancia por permutación **agrupada** para las variables de sitio (Vs30 + Z1.0 juntas) y para los mecanismos.
  - Repetir la importancia por permutación en los 5 folds y con otras semillas del RF.
  - Complementar con SHAP o PDP/ICE para Mw × Rrup.

---

## 5. Lección aprendida de entorno: bloqueo de hardlinks en OneDrive (`--link-mode=copy`)

**Síntoma.** El repositorio vive dentro de una carpeta sincronizada por OneDrive
(`C:\Users\...\OneDrive\DOCUMENTOS\...`). Al instalar paquetes en `.venv` con **uv**,
la instalación fallaba o se interrumpía al enlazar archivos desde la caché.

**Causa raíz.**
- **uv no copia los paquetes, los enlaza:** guarda los paquetes descomprimidos en una caché global (`%LOCALAPPDATA%\uv\cache`) y en Windows, por defecto, los enlaza al entorno con **hardlinks**, sin copiarlos.
- **Un hardlink solo funciona dentro de un mismo volumen** y en un sistema de archivos que lo permita.
- **OneDrive no tolera bien los hardlinks:** las carpetas que sincroniza pasan por su filtro de sincronización (Files On-Demand), que no los soporta bien. Además, OneDrive intenta subir y bloquear los archivos mientras uv escribe.
- **Resultado:** errores de enlace, archivos bloqueados y entornos a medio instalar.

**Solución aplicada.** Forzar a uv a **copiar** los archivos en lugar de enlazarlos:

```powershell
# Opción puntual
uv pip install -r requirements.txt --link-mode=copy

# Opción persistente para la sesión (PowerShell)
$env:UV_LINK_MODE = "copy"
uv pip install -r requirements.txt
```

Para que quede fijo en el proyecto, se puede agregar `link-mode = "copy"` en un `uv.toml`
en la raíz del repositorio.

**Lecciones y buenas prácticas.**
1. **No dejar que OneDrive sincronice `.venv`.** Ocupa cientos de MB y genera conflictos. Hay dos opciones: crear el entorno fuera de OneDrive (`uv venv C:\venvs\upc-t1`) o excluir la carpeta de la sincronización.
2. **Agregar un `.gitignore`.** El repositorio no tiene uno, así que hay que excluir `.venv/`, `__pycache__/` y `.ipynb_checkpoints/` para no subir el entorno a GitHub.
3. **Completar `requirements.txt`.** El notebook usa además `seaborn`, `scipy` e `ipykernel` (y `pillow` para comprobar las figuras), que no están listados. Hay que agregarlos para que el entorno sea reproducible.
4. **Fijar `--link-mode=copy`** cuando el proyecto esté en una carpeta sincronizada en la nube (OneDrive, Dropbox, Google Drive).

---

## 6. Síntesis de la entrega T2

| Aspecto | Estado |
|---|---|
| Limpieza geotécnica y target Sd | ✅ Implementado y comprobado (datos sintéticos) |
| Partición sin fuga por EQID | ✅ Intersección = 0 en los 5 folds · ⚠️ falta corregir la fuga por imputación |
| Benchmark Ridge / RF / MLP | ✅ RF seleccionado (RMSE 2.56 cm) · ⚠️ validado solo en 1 fold |
| xAI bajo PCS | ✅ Ranking Rrup > Mw > Vs30 estable · ⚠️ faltan perturbaciones de datos y del modelo |
| Figuras de entrega a 300 dpi | ✅ 3 figuras en `docs/` |
| Datos reales de PEER NGA-West2 | ❌ Pendiente: requisito para validar las conclusiones físicas |

**Decisión:** Random Forest como modelo principal para T3/TF, con la GMM (Ridge) como
baseline físico obligatorio. La decisión es provisional hasta cumplir tres condiciones:
validación con los 5 folds, imputación dentro del Pipeline y repetición con el flatfile real.
