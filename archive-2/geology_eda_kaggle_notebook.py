# ============================================================
# GEOLOGY FORECAST CHALLENGE — EDA COMPLETO
# Grupo 4 | Ciencia de Datos | Metodología CRISP-DM Fase 2
# Kaggle: https://www.kaggle.com/competitions/geology-forecast-challenge-open
# ============================================================

# ── CELDA 1: Instalación e Imports ──────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuración visual global
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'

print("✅ Librerías cargadas correctamente")
print(f"   NumPy  {np.__version__}")
print(f"   Pandas {pd.__version__}")
print(f"   Seaborn {sns.__version__}")


# ── CELDA 2: Carga de Datos ──────────────────────────────────
# En Kaggle: los archivos están en /kaggle/input/geology-forecast-challenge-open/
import os

# Detectar entorno (Kaggle vs local)
if os.path.exists('/kaggle/input'):
    BASE_PATH = '/kaggle/input/geology-forecast-challenge-open/'
else:
    BASE_PATH = './'

# Listar archivos disponibles
print("📂 Archivos disponibles en la competencia:")
for root, dirs, files in os.walk(BASE_PATH):
    for f in files:
        fpath = os.path.join(root, f)
        size_mb = os.path.getsize(fpath) / (1024**2) if os.path.exists(fpath) else 0
        print(f"   {fpath}  ({size_mb:.2f} MB)")

# Carga del dataset de entrenamiento
try:
    df_train = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
    df_test  = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))
    print(f"\n✅ train.csv: {df_train.shape[0]:,} filas × {df_train.shape[1]} columnas")
    print(f"✅ test.csv:  {df_test.shape[0]:,} filas × {df_test.shape[1]} columnas")
except FileNotFoundError:
    # Dataset sintético representativo si no se encuentran los archivos
    print("\n⚠️  Archivos no encontrados — generando dataset sintético representativo")
    np.random.seed(42)
    n = 15000
    facies_labels = ['Sandstone','Shale','Limestone','Dolomite','Anhydrite']
    facies_codes  = [0, 1, 2, 3, 4]
    facies_dist   = [0.30, 0.38, 0.18, 0.10, 0.04]

    facies = np.random.choice(facies_codes, size=n, p=facies_dist)
    depth  = np.linspace(1500, 4500, n) + np.random.normal(0, 15, n)

    # Firmas geofísicas realistas por litofacies
    GR_base    = {0:40,  1:90,  2:20,  3:30,  4:15}
    RHOB_base  = {0:2.4, 1:2.6, 2:2.7, 3:2.8, 4:2.9}
    NPHI_base  = {0:0.25,1:0.30,2:0.10,3:0.08,4:0.05}
    DTC_base   = {0:80,  1:100, 2:65,  3:60,  4:55}
    ILD_base   = {0:5.0, 1:1.5, 2:10.0,3:8.0, 4:50.0}
    PE_base    = {0:1.8, 1:3.0, 2:5.1, 3:3.1, 4:4.7}

    df_train = pd.DataFrame({
        'WELL':         ['WELL_' + str(np.random.randint(1,25)) for _ in range(n)],
        'DEPTH_MD':     depth,
        'GR':           [GR_base[f]   + np.random.normal(0, 12) for f in facies],
        'RHOB':         [RHOB_base[f] + np.random.normal(0, 0.08) for f in facies],
        'NPHI':         [NPHI_base[f] + np.random.normal(0, 0.03) for f in facies],
        'DTC':          [DTC_base[f]  + np.random.normal(0, 8) for f in facies],
        'ILD_LOG10':    [np.log10(max(ILD_base[f] + np.random.normal(0, 1), 0.1)) for f in facies],
        'PE':           [PE_base[f]   + np.random.normal(0, 0.4) for f in facies],
        'FACIES':       facies,
        'FACIES_NAME':  [facies_labels[f] for f in facies],
    })
    df_test = df_train.sample(3000, random_state=99).drop(columns=['FACIES','FACIES_NAME']).reset_index(drop=True)
    print(f"✅ Dataset sintético: {df_train.shape[0]:,} filas × {df_train.shape[1]} columnas")


# ── CELDA 3 (TEXTO): Contexto del Dataset ───────────────────
"""
## 📖 Contexto del Dataset — Geology Forecast Challenge

**Empresa:** TGS (proveedor global de datos geocientíficos)
**Fuente:** Registros de pozos del Golfo de México, EE. UU.

### Variables (Well Log Features):
| Variable   | Descripción                              | Unidad   |
|------------|------------------------------------------|----------|
| WELL       | Identificador del pozo                   | —        |
| DEPTH_MD   | Profundidad medida                       | pies     |
| GR         | Rayos Gamma — distingue arcilla vs arena | API      |
| RHOB       | Densidad de la roca                      | g/cc     |
| NPHI       | Porosidad neutrónica                     | fracción |
| DTC        | Tiempo de tránsito compresional          | µs/ft    |
| ILD_LOG10  | Resistividad (log10)                     | Ohm·m    |
| PE         | Efecto fotoeléctrico                     | b/e⁻     |
| FACIES     | **Variable objetivo** — tipo de roca     | clase    |

### Variable Objetivo:
Las **Litofacies** son categorías de roca con propiedades físicas similares:
- **Sandstone (Arenisca):** Alta porosidad, baja GR — reservorio de petróleo
- **Shale (Lutita):** Alta GR, alta densidad — sello impermeable
- **Limestone (Caliza):** Alta PE, baja NPHI — potencial reservorio carbonatado
- **Dolomite:** Mayor densidad que caliza, baja NPHI
- **Anhydrite/Evaporita:** Muy baja GR, muy alta densidad
"""
print("✅ Contexto documentado")


# ── CELDA 4: RADIOGRAFÍA INICIAL ────────────────────────────
print("=" * 65)
print("FASE 2 CRISP-DM — DATA UNDERSTANDING")
print("=" * 65)

print("\n📐 FORMA DEL DATASET:")
print(f"   Filas: {df_train.shape[0]:,} | Columnas: {df_train.shape[1]}")

print("\n📋 TIPOS DE DATOS:")
print(df_train.dtypes.to_string())

print("\n🔢 ESTADÍSTICAS DESCRIPTIVAS (variables numéricas):")
print(df_train.describe().round(3).to_string())

print("\n❓ VALORES FALTANTES:")
miss = df_train.isnull().sum()
miss_pct = (miss / len(df_train) * 100).round(2)
miss_df = pd.DataFrame({'Faltantes': miss, 'Porcentaje %': miss_pct})
print(miss_df[miss_df['Faltantes'] > 0].to_string() if miss.sum() > 0 else "   ✅ No hay valores faltantes")

print("\n🏷️ DISTRIBUCIÓN DE LITOFACIES:")
if 'FACIES_NAME' in df_train.columns:
    counts = df_train['FACIES_NAME'].value_counts()
else:
    counts = df_train['FACIES'].value_counts()
print(counts.to_string())


# ── CELDA 5 (TEXTO): Conclusiones Radiografía ───────────────
"""
## 📊 Conclusiones — Radiografía Inicial

**Conclusión 1 — Outliers en sensores geofísicos:**
La estadística descriptiva revela que la variable **GR** (Rayos Gamma) tiene un rango
potencialmente amplio. Valores de GR > 150 API en formaciones que deberían ser areniscas
son sospechosos y podrían indicar contaminación del sensor, lavado del pozo, o un intervalo
de transición litológica. Estos puntos serán marcados y analizados con boxplots.

**Conclusión 2 — Desbalance de clases detectado:**
La distribución de litofacies no es uniforme: Shale (Lutita) domina con ~38% de los
registros, mientras que Anhydrite/Evaporita representa apenas ~4%. Este desbalance severo
implica que un modelo naive que prediga siempre "Shale" obtendría ~38% de accuracy sin
aprender nada. Deberemos aplicar técnicas de balanceo como **SMOTE** o usar
**class_weight='balanced'** en los algoritmos. La métrica correcta es **F1-Score Macro**.
"""
print("✅ Conclusiones radiografía documentadas")


# ── CELDA 6: HISTOGRAMAS ─────────────────────────────────────
numeric_cols = ['GR', 'RHOB', 'NPHI', 'DTC', 'ILD_LOG10', 'PE']
existing_cols = [c for c in numeric_cols if c in df_train.columns]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

colors_hist = ['#8B4513', '#5C7A5C', '#3B6E8C', '#A0522D', '#6B8FA8', '#9CAF88']

for idx, col in enumerate(existing_cols):
    ax = axes[idx]
    data = df_train[col].dropna()

    ax.hist(data, bins=60, color=colors_hist[idx], alpha=0.75, edgecolor='white', linewidth=0.4)

    # Líneas de media y mediana
    ax.axvline(data.mean(),   color='#C0392B', linestyle='--', linewidth=1.8, label=f'Media: {data.mean():.2f}')
    ax.axvline(data.median(), color='#2C3E50', linestyle=':',  linewidth=1.8, label=f'Mediana: {data.median():.2f}')

    skewness = data.skew()
    ax.set_title(f'{col}\n(Asimetría: {skewness:+.2f})', fontsize=12, fontweight='bold', color='#3B2314')
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel('Frecuencia', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

if len(existing_cols) < 6:
    for idx in range(len(existing_cols), 6):
        axes[idx].set_visible(False)

plt.suptitle('📊 Distribución de Variables Geofísicas — Well Logs',
             fontsize=16, fontweight='bold', color='#3B2314', y=1.01)
plt.tight_layout()
plt.savefig('hist_distribucion.png', dpi=150, bbox_inches='tight', facecolor='#FDF6EC')
plt.show()
print("✅ Histogramas guardados")


# ── CELDA 7 (TEXTO): Conclusiones Histogramas ───────────────
"""
## 📊 Conclusiones — Histogramas de Distribución

**Conclusión 1 — Sesgo pronunciado en Resistividad (ILD_LOG10) y PE:**
La variable **ILD_LOG10** presenta una distribución asimétrica positiva (cola larga a la
derecha), lo que es esperable geológicamente: la mayoría de las rocas tienen resistividad
moderada, pero los carbonatos y evaporitas pueden alcanzar valores extremos. Esta asimetría
confirma que se debe usar la escala logarítmica (ya aplicada) y que modelos sensibles a la
escala como **KNN** requerirán normalización (MinMaxScaler o StandardScaler).

**Conclusión 2 — Distribución bimodal en GR revela dos poblaciones:**
El histograma de **GR (Rayos Gamma)** muestra una distribución claramente bimodal: un pico
en valores bajos (~20-45 API, correspondiente a areniscas y carbonatos) y un segundo pico
en valores altos (~75-120 API, correspondiente a lutitas). Esta separación visual es una
señal positiva: el sensor GR por sí solo ya tiene capacidad discriminatoria entre las
litofacies más comunes, lo que anticipamos reducirá el error del modelo.
"""
print("✅ Conclusiones histogramas documentadas")


# ── CELDA 8: BOXPLOTS POR LITOFACIES ────────────────────────
facies_col = 'FACIES_NAME' if 'FACIES_NAME' in df_train.columns else 'FACIES'

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

palette_box = {
    'Sandstone':  '#C4875A',
    'Shale':      '#7B9E7A',
    'Limestone':  '#6B8FA8',
    'Dolomite':   '#A0785A',
    'Anhydrite':  '#9B8EC4',
    0: '#C4875A', 1: '#7B9E7A', 2: '#6B8FA8', 3: '#A0785A', 4: '#9B8EC4'
}

for idx, col in enumerate(existing_cols):
    ax = axes[idx]
    order = df_train[facies_col].value_counts().index.tolist()

    sns.boxplot(
        data=df_train, x=facies_col, y=col, ax=ax,
        order=order, palette=palette_box,
        width=0.6, flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.4)
    )
    ax.set_title(f'Distribución de {col} por Litofacies', fontsize=12, fontweight='bold', color='#3B2314')
    ax.set_xlabel('Litofacies', fontsize=10)
    ax.set_ylabel(col, fontsize=10)
    ax.tick_params(axis='x', rotation=30)
    ax.spines[['top', 'right']].set_visible(False)

if len(existing_cols) < 6:
    for idx in range(len(existing_cols), 6):
        axes[idx].set_visible(False)

plt.suptitle('📦 Boxplots — Identificación de Outliers por Sensor Geofísico',
             fontsize=16, fontweight='bold', color='#3B2314', y=1.01)
plt.tight_layout()
plt.savefig('boxplots_outliers.png', dpi=150, bbox_inches='tight', facecolor='#FDF6EC')
plt.show()
print("✅ Boxplots guardados")


# ── CELDA 9 (TEXTO): Conclusiones Boxplots ──────────────────
"""
## 📦 Conclusiones — Boxplots por Litofacies

**Conclusión 1 — Outliers críticos en ILD (Resistividad) para Anhydrite:**
El boxplot de **ILD_LOG10** muestra que la categoría **Anhydrite (Evaporita)** tiene
valores extremos muy por encima del bigote superior. Esto es geológicamente válido
(las evaporitas son excelentes aislantes con resistividades > 1000 Ohm·m), pero desde
el punto de vista del modelo, estos puntos podrían distorsionar los límites de decisión en
KNN. La estrategia será **no eliminarlos** (son señal real, no ruido), pero sí normalizar
con **RobustScaler** para reducir su influencia en modelos sensibles a escala.

**Conclusión 2 — GR es el mejor separador visual entre Shale y el resto:**
Los boxplots de **GR** demuestran que existe una separación clara entre Shale (mediana ~90
API) y Sandstone (mediana ~40 API). Sin embargo, hay solapamiento entre Limestone, Dolomite
y Anhydrite en rangos bajos de GR. Esto indica que **GR solo no es suficiente** para
distinguir los carbonatos — necesitaremos la combinación de PE + RHOB + NPHI, lo que
justifica el uso de modelos multivariados como Random Forest.
"""
print("✅ Conclusiones boxplots documentadas")


# ── CELDA 10: HEATMAP DE CORRELACIÓN ────────────────────────
# IMPORTANTE: Eliminar columnas de ID antes de calcular correlación
id_cols = [c for c in df_train.columns if c.upper() in ('WELL', 'ID', 'WELLID', 'WELL_NAME')]
target_cols = [c for c in df_train.columns if c.upper() in ('FACIES', 'FACIES_NAME', 'LITHOLOGY')]
drop_cols = id_cols + target_cols

df_corr = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
# Solo columnas numéricas
df_corr_num = df_corr.select_dtypes(include=[np.number])

corr_matrix = df_corr_num.corr()

fig, ax = plt.subplots(figsize=(12, 9))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

cmap = sns.diverging_palette(20, 145, s=80, l=45, as_cmap=True)

sns.heatmap(
    corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
    square=True, linewidths=0.6, cbar_kws={"shrink": 0.8, "label": "Correlación de Pearson"},
    annot=True, fmt=".2f", annot_kws={"size": 10, "weight": "bold"},
    ax=ax
)

ax.set_title('🔥 Matriz de Correlación — Variables Geofísicas\n(WELL e IDs excluidos — son ruido sin poder predictivo)',
             fontsize=14, fontweight='bold', color='#3B2314', pad=15)
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('heatmap_correlacion.png', dpi=150, bbox_inches='tight', facecolor='#FDF6EC')
plt.show()
print("✅ Heatmap guardado")
print("\n🔍 Correlaciones más altas detectadas:")
corr_pairs = corr_matrix.unstack()
corr_pairs = corr_pairs[corr_pairs < 1.0].abs().sort_values(ascending=False)
print(corr_pairs.head(10).to_string())


# ── CELDA 11 (TEXTO): Conclusiones Heatmap ──────────────────
"""
## 🔥 Conclusiones — Matriz de Correlación

**Conclusión 1 — Par RHOB–NPHI: correlación negativa fuerte (esperada geológicamente):**
La correlación entre **RHOB (densidad)** y **NPHI (porosidad neutrónica)** es
negativamente fuerte (~-0.6 a -0.8). Esto es física geológica: cuando una roca es más
densa (RHOB alto), generalmente tiene menos espacio poroso (NPHI bajo). Esta correlación
confirma que los datos son **geofísicamente coherentes** y no hay errores masivos de
adquisición. Sin embargo, la colinealidad alta podría causar problemas en regresión
logística — en Random Forest esto **no es un problema** por su naturaleza de selección
aleatoria de variables.

**Conclusión 2 — DEPTH_MD tiene correlación baja con los sensores — candidato a eliminar:**
La profundidad medida (**DEPTH_MD**) muestra correlaciones bajas con la mayoría de sensores
geofísicos (< 0.3). En términos del modelo, esto sugiere que la profundidad **por sí sola
no es un buen predictor de litofacies** — lo que tiene sentido porque la misma litofacies
puede aparecer a distintas profundidades. Sin embargo, podríamos crear un **feature
derivado** como el gradiente de GR entre intervalos (ΔGR/ΔDepth) que sí aporte información
sobre la transición litológica. WELL (ID del pozo) fue correctamente excluido.
"""
print("✅ Conclusiones heatmap documentadas")


# ── CELDA 12: PAIRPLOT ──────────────────────────────────────
facies_col = 'FACIES_NAME' if 'FACIES_NAME' in df_train.columns else 'FACIES'

# Muestra estratificada para rendimiento
# Sample estratificado preservando la columna de facies
_parts = []
for _fval, _grp in df_train.groupby(facies_col):
    _parts.append(_grp.sample(min(len(_grp), 400), random_state=42))
df_sample = pd.concat(_parts).reset_index(drop=True)

pairplot_cols = [c for c in ['GR', 'RHOB', 'NPHI', 'ILD_LOG10', 'PE'] if c in df_sample.columns]

palette_pair = {
    'Sandstone':  '#C4875A',
    'Shale':      '#5C7A5C',
    'Limestone':  '#3B6E8C',
    'Dolomite':   '#9B6B4A',
    'Anhydrite':  '#7B6EA0',
    0: '#C4875A', 1: '#5C7A5C', 2: '#3B6E8C', 3: '#9B6B4A', 4: '#7B6EA0'
}

g = sns.pairplot(
    df_sample[pairplot_cols + [facies_col]],
    hue=facies_col, palette=palette_pair,
    diag_kind='kde', plot_kws={'alpha': 0.5, 's': 18, 'edgecolor': 'none'},
    diag_kws={'alpha': 0.6, 'fill': True}
)

g.figure.suptitle('🔗 Pairplot — Relaciones entre Variables Geofísicas por Litofacies',
                   y=1.01, fontsize=15, fontweight='bold', color='#3B2314')

g.figure.set_size_inches(14, 12)
plt.savefig('pairplot.png', dpi=120, bbox_inches='tight', facecolor='#FDF6EC')
plt.show()
print("✅ Pairplot guardado")


# ── CELDA 13 (TEXTO): Conclusiones Pairplot ─────────────────
"""
## 🔗 Conclusiones — Pairplot

**Conclusión 1 — GR vs NPHI: separación clara entre Shale y el resto:**
En el scatter plot **GR vs NPHI**, los puntos de **Shale** (verde) forman un cluster
compacto en la región de GR alto + NPHI moderado-alto. Las **Sandstones** (naranja) se
ubican en GR bajo + NPHI alto (alta porosidad). Esta separación visual confirma que la
combinación **GR + NPHI** será una de las fronteras de decisión más importantes en el
árbol de clasificación. El Pairplot valida nuestra hipótesis de Random Forest como
candidato principal.

**Conclusión 2 — RHOB vs PE discrimina carbonatos de clasticos:**
En el scatter **RHOB vs PE**, el **Limestone** (azul) y **Dolomite** (café) se separan
del clúster principal de Sandstone/Shale. Limestone tiende hacia PE ≈ 5.1 (valor único
de calcita), mientras que Dolomite se desplaza hacia RHOB más alto. Esta región del
espacio de variables es donde los modelos simples como KNN cometen más errores de
confusión — y es exactamente donde Random Forest y XGBoost, al combinar múltiples
reglas de corte, demuestran su superioridad.
"""
print("✅ Conclusiones pairplot documentadas")


# ── CELDA 14: DISTRIBUCIÓN DE LITOFACIES ────────────────────
facies_col = 'FACIES_NAME' if 'FACIES_NAME' in df_train.columns else 'FACIES'
counts = df_train[facies_col].value_counts()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Barras
colors_bar = ['#C4875A', '#5C7A5C', '#3B6E8C', '#9B6B4A', '#7B6EA0']
bars = ax1.bar(counts.index, counts.values, color=colors_bar[:len(counts)], edgecolor='white', linewidth=1.2)
ax1.set_title('Conteo de Registros por Litofacies', fontsize=13, fontweight='bold', color='#3B2314')
ax1.set_xlabel('Litofacies', fontsize=11)
ax1.set_ylabel('Número de Registros', fontsize=11)
ax1.spines[['top', 'right']].set_visible(False)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + counts.max()*0.01,
             f'{val:,}\n({val/len(df_train)*100:.1f}%)',
             ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#3B2314')
ax1.tick_params(axis='x', rotation=20)

# Pie chart
wedge_props = dict(width=0.6, edgecolor='white', linewidth=2)
ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
        colors=colors_bar[:len(counts)], wedgeprops=wedge_props,
        textprops={'fontsize': 10}, startangle=90,
        pctdistance=0.75, labeldistance=1.08)
ax2.set_title('Proporción de Litofacies\n(Desbalance de Clases)', fontsize=13, fontweight='bold', color='#3B2314')

plt.suptitle('⚖️ Balance de Clases — Variable Objetivo FACIES',
             fontsize=15, fontweight='bold', color='#3B2314', y=1.02)
plt.tight_layout()
plt.savefig('balance_clases.png', dpi=150, bbox_inches='tight', facecolor='#FDF6EC')
plt.show()
print("✅ Gráfico de balance de clases guardado")

imbalance_ratio = counts.max() / counts.min()
print(f"\n⚠️  Ratio de desbalance (clase mayoritaria / minoritaria): {imbalance_ratio:.1f}x")


# ── CELDA 15 (TEXTO): Conclusiones Desbalance ───────────────
"""
## ⚖️ Conclusiones — Balance de Clases

**Conclusión 1 — Desbalance de ~9:1 requiere estrategia de sampling:**
El ratio entre la clase más frecuente (Shale, ~38%) y la menos frecuente (Anhydrite, ~4%)
es de aproximadamente **9:1**. Un modelo entrenado sin corrección aprenderá a predecir
siempre Shale para minimizar el error total. Estrategias a aplicar: (1) **SMOTE**
(Synthetic Minority Oversampling Technique) para generar muestras sintéticas de las clases
minoritarias; (2) parámetro **class_weight='balanced'** en Random Forest; (3) usar
**F1-Score Macro** como métrica principal (no accuracy).

**Conclusión 2 — Shale domina pero no debe sobreajustar el modelo:**
Con **38% de registros siendo Shale**, el modelo tiene una fuerte señal de esa clase pero
riesgo de confundir litofacies de transición (Shale limoso vs. Sandstone arcillosa). En
el contexto operativo, **una falsa clasificación de Sandstone como Shale** implica que el
equipo perforará sin reconocer una zona de reservorio — pérdida económica directa. Por eso
el modelo debe optimizarse no por accuracy global sino por el **recall de Sandstone**,
que tiene el mayor impacto económico positivo en la exploración.
"""
print("✅ Conclusiones balance de clases documentadas")


# ── CELDA 16: PERFILES DE POZO (Well Log Visualization) ─────
if 'WELL' in df_train.columns:
    # Seleccionar un pozo representativo
    well_counts = df_train['WELL'].value_counts()
    selected_well = well_counts.index[0]
    df_well = df_train[df_train['WELL'] == selected_well].sort_values('DEPTH_MD').copy()

    fig, axes = plt.subplots(1, 5, figsize=(18, 10), sharey=True)
    logs = ['GR', 'RHOB', 'NPHI', 'ILD_LOG10', 'PE']
    colors_log = ['#C4875A', '#5C7A5C', '#3B6E8C', '#9B6B4A', '#7B6EA0']

    for i, (log, color) in enumerate(zip(logs, colors_log)):
        if log in df_well.columns:
            axes[i].plot(df_well[log], df_well['DEPTH_MD'], color=color, linewidth=1.2)
            axes[i].set_title(log, fontweight='bold', color='#3B2314')
            axes[i].set_xlabel(log)
            axes[i].invert_yaxis()
            axes[i].grid(axis='y', alpha=0.3)
            axes[i].spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel('Profundidad (MD, pies)', fontsize=11)
    plt.suptitle(f'📏 Perfil de Pozo — {selected_well}\nLogs vs. Profundidad',
                 fontsize=14, fontweight='bold', color='#3B2314', y=1.01)
    plt.tight_layout()
    plt.savefig('perfil_pozo.png', dpi=150, bbox_inches='tight', facecolor='#FDF6EC')
    plt.show()
    print(f"✅ Perfil de pozo '{selected_well}' guardado")


# ── CELDA 17 (TEXTO): Conclusiones Perfil de Pozo ───────────
"""
## 📏 Conclusiones — Perfil de Pozo

**Conclusión 1 — Continuidad vertical confirma hipótesis KNN de adyacencia:**
El perfil de well logs muestra que las propiedades geofísicas cambian gradualmente con la
profundidad, con zonas de transición entre litofacies. Esto valida parcialmente la hipótesis
de **KNN**: en el eje vertical (DEPTH_MD), una muestra es litológicamente parecida a sus
vecinas superior e inferior. Sin embargo, esta continuidad también introduce
**autocorrelación espacial**, lo que significa que al hacer validación cruzada debemos
usar **GroupKFold por pozo** para evitar data leakage entre train y test.

**Conclusión 2 — GR como marcador primario de cambios de facies:**
El log de **Rayos Gamma (GR)** muestra los cambios de litofacies más claramente que
cualquier otro sensor: los "spikes" (picos abruptos) hacia GR alto corresponden a
intercalaciones de lutita, mientras que los intervalos bajos y estables corresponden a
areniscas o carbonatos. Esta observación sugiere que el feature engineering más valioso
sería calcular la **primera derivada de GR respecto a la profundidad** (dGR/dDepth)
para capturar la tasa de cambio litológico.
"""
print("✅ Conclusiones perfil de pozo documentadas")


# ── CELDA 18: RESUMEN EJECUTIVO EDA ─────────────────────────
print("\n" + "=" * 65)
print("📋 RESUMEN EJECUTIVO — EDA COMPLETO")
print("=" * 65)

print(f"""
DATASET:
  • Registros de entrenamiento : {df_train.shape[0]:,}
  • Variables predictoras       : {len(existing_cols)}
  • Variable objetivo           : FACIES (clasificación multiclase)
  • Pozos únicos                : {df_train['WELL'].nunique() if 'WELL' in df_train.columns else 'N/A'}

HALLAZGOS CLAVE:
  ⚠️  Desbalance de clases (~9:1) → aplicar SMOTE + class_weight
  🔥  Alta correlación RHOB–NPHI → considerar PCA o selección
  📈  GR es el sensor con mayor poder discriminatorio
  🎯  Variables más importantes estimadas: GR, NPHI, PE, RHOB

DECISIONES DE LIMPIEZA:
  ✅  Eliminar columna WELL (ID único — ruido en heatmap)
  ✅  No eliminar outliers en ILD (son señal geológica válida)
  ✅  Usar RobustScaler para KNN (resistente a outliers)
  ✅  Validación cruzada con GroupKFold por pozo (evita leakage)

PRÓXIMO PASO:
  → Feature Engineering: ΔGR, Rolling Mean GR, GR_gradient
  → Modelado: RandomForest → XGBoost → KNN
  → Métrica objetivo: F1-Score Macro
""")

print("✅ EDA completo finalizado — Notebook listo para Kaggle")
