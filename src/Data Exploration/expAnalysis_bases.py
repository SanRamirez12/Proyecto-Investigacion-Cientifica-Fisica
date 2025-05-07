import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.table import Table
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

#Listo
def read_fits_to_dataframe(fits_path):
    table = Table.read(fits_path)
    df = table.to_pandas()
    return df

#Listo
def select_features(df):
    selected_columns = [
        'Flux1000', 'Energy_Flux100', 'SpectrumType', 'PL_Flux_Density',
        'PL_Index', 'LP_Flux_Density', 'LP_Index', 'LP_beta', 'LP_SigCurv',
        'LP_EPeak', 'PLEC_Flux_Density', 'PLEC_IndexS', 'PLEC_ExpfactorS',
        'PLEC_Exp_Index', 'PLEC_SigCurv', 'PLEC_EPeak', 'Npred',
        'Flux_Band', 'nuFnu_Band', 'Variability_Index', 'Frac_Variability',
        'Flux_Peak', 'Flux_History', 'CLASS1'
    ]
    df_selected = df[selected_columns].copy()
    return df_selected

#Listo
def clean_class_labels(df):
    def classify_label(label):
        if pd.isna(label):
            return 'UnAss'
        label = label.upper()
        if label.startswith('FSRQ'):
            return 'FSRQ'
        elif label.startswith('BLL'):
            return 'BLL'
        elif label.startswith('BCU'):
            return 'BCU'
        elif label in ['RDG', 'NLSY1', 'SEY', 'AGN', 'CSS', 'SSRQ']:  # Puedes agregar más aquí si ves más tipos
            return 'OtroAGN'
        else:
            return 'OtroAGN'
    df['CLASS1'] = df['CLASS1'].apply(classify_label)
    return df

#Listo
def encode_spectrum_type(df):
    spectrum_mapping = {'PowerLaw': 0, 'LogParabola': 1, 'PLSuperExpCutoff': 2}
    df['SpectrumType'] = df['SpectrumType'].map(spectrum_mapping)
    return df


def impute_missing_values(df):
    imputer = SimpleImputer(strategy='mean')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df


def normalize_features(df):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('SpectrumType')  # Mantener SpectrumType como categorico codificado
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def split_train_test(df):
    train_df = df[df['CLASS1'].isin(['FSRQ', 'BLL', 'BCU', 'OtroAGN'])].copy()
    test_df = df[df['CLASS1'] == 'UnAss'].copy()
    return train_df, test_df

def plot_pairplot(df, features, label_col='CLASS1'):
    sns.pairplot(df[features + [label_col]], hue=label_col)
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12,10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
    plt.title('Correlation Heatmap')
    plt.show()

def save_dataframe(df, path):
    df.to_csv(path, index=False)

# ---------------- MAIN PIPELINE ----------------

def main_pipeline(fits_path):
    df = read_fits_to_dataframe(fits_path)
    df = select_features(df)
    df = clean_class_labels(df)
    df = encode_spectrum_type(df)
    df = impute_missing_values(df)
    df = normalize_features(df)
    train_df, test_df = split_train_test(df)

    print("Train dataset size:", train_df.shape)
    print("Test dataset size:", test_df.shape)

    # Opcionales: visualización inicial
    plot_pairplot(train_df, features=['Flux1000', 'Energy_Flux100', 'PL_Index', 'Variability_Index'])
    plot_correlation_heatmap(train_df)

    # Guardar datasets si quieres
    save_dataframe(train_df, 'train_data.csv')
    save_dataframe(test_df, 'test_data.csv')

    return train_df, test_df

# --------------------------------------------------

# Para correrlo simplemente harías:
# train_df, test_df = main_pipeline('ruta/a/tu/archivo.fits')


