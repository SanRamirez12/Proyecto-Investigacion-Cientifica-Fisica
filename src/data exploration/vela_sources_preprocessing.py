
# =================== Extracción de fuentes de Vela según nombres ===================
import os
import re
import utils_data_exp as ude

# Lista de fuentes Vela (podría tener duplicados o nombres incorrectos)
nombres_vela = [
    "4FGL J0838.4-3952", "4FGL J0837.8-4048c", "4FGL J0844.9-4117", "4FGL J0847.8-4138",
    "4FGL J0840.5-4122c", "4FGL J0827.0-4111c", "4FGL J0822.8-4207", "4FGL J0853.2-4218c",
    "4FGL J0850.8-4239", "4FGL J0857.7-4256c", "4FGL J0853.6-4306", "4FGL J0902.8-4633",
    "4FGL J0859.2-4729", "4FGL J0802.5-4727", "4FGL J0911.6-4738c", "4FGL J0917.9-4755",
    "4FGL J0910.3-4816c", "4FGL J0901.3-4848c", "4FGL J0918.9-4904c", "4FGL J0903.6-5025",
    "4FGL J0753.8-4700", "4FGL J0826.1-5053", "4FGL J0813.1-5049", "4FGL J0751.6-5029c",
    "4FGL J0808.3-5121", "4FGL J0803.5-5145c", "4FGL J0825.6-5216c", "4FGL J0817.7-5258c",
    "4FGL J0859.3-4342", "4FGL J0857.0-4353c", "4FGL J0900.1-4402c", "4FGL J0900.5-4434c",
    "4FGL J0901.1-4456c", "4FGL J0832.2-4322c", "4FGL J0833.3-4342c", "4FGL J0833.8-4400",
    "4FGL J0828.4-4444", "4FGL J0830.5-4451", "4FGL J0848.8-4328", "4FGL J0844.1-4330",
    "4FGL J0853.1-4407", "4FGL J0849.2-4410c", "4FGL J0854.8-4504", "4FGL J0854.9-4426",
    "4FGL J0850.3-4448", "4FGL J0848.2-4527", "4FGL J0856.0-4724c", "4FGL J0851.2-4737",
    "4FGL J0846.6-4747", "4FGL J0859.8-4530c", "4FGL J0900.2-4608", "4FGL J0858.4-4615c"
]

# === Paso 1: Lectura del catálogo completo ===
df = ude.leer_fits_vela('gll_psc_v35.fit')
print(df)

# === Paso 2: Normalizar lista de nombres (eliminar espacios, etc.) ===
nombres_normalizados = [re.sub(r'\s+', ' ', n.strip()) for n in set(nombres_vela)]
nombres_catalogo = df['Source_Name'].values

nombres_encontrados = [n for n in nombres_normalizados if n in nombres_catalogo]
nombres_no_encontrados = [n for n in nombres_normalizados if n not in nombres_catalogo]

print(f"Se encontraron {len(nombres_encontrados)} fuentes únicas de Vela en el catálogo.")
if nombres_no_encontrados:
    print("No se encontraron las siguientes fuentes:")
    for nombre in nombres_no_encontrados:
        print(" -", nombre)

# === Paso 3: Filtrar solo las fuentes de Vela ===
df_vela = df[df['Source_Name'].isin(nombres_encontrados)].drop_duplicates(subset='Source_Name')
print(df_vela[['Source_Name', 'CLASS1']])

# === Paso 3.1: Renombrar la clase como SrcVela ===
df_vela['CLASS1'] = 'SrcVela'

# === Paso 4: Preprocesamiento estándar ===
df_vela = ude.onehot_encode_spectrum_type(df_vela)
df_vela = ude.inf_a_nan(df_vela)
df_vela = ude.eliminar_filas_nans(df_vela)
print(df_vela)

# === Paso 4.1: Eliminación de la columna Source_Name ===
df_vela = df_vela.drop(columns=['Source_Name'])
print(df_vela)

# === Paso 5: Exportación ===
directorio_base = os.path.dirname(os.path.abspath(__file__))
carpeta = os.path.join(directorio_base, '..', '..', 'data', 'post preliminary analysis')
os.makedirs(carpeta, exist_ok=True)

df_vela.to_csv(f'{carpeta}/fuentes_vela.csv', index=False)
df_vela.to_parquet(f'{carpeta}/fuentes_vela.parquet', index=False)
print("Exportación completada exitosamente a CSV y Parquet.")
