import os
import pandas as pd
import kagglehub

def load_and_explore_data():
    """
    Carga datos de Kaggle Competition en cualquier entorno.
    """

    # ==============================
    # 1. DETECTAR ENTORNO
    # ==============================
    if os.path.exists('/kaggle/input'):
        print("📍 Entorno: Kaggle Notebook")
        BASE_PATH = '/kaggle/input/geology-forecast-challenge-open/'

    else:
        print("📍 Entorno: Local / Codespaces")
        BASE_PATH = kagglehub.competition_download(
            'geology-forecast-challenge-open'
        )
        print("Download path:", BASE_PATH)

    # ==============================
    # 2. EXPLORAR ARCHIVOS
    # ==============================
    print("\n📂 Archivos encontrados:")

    for root, dirs, files in os.walk(BASE_PATH):
        for f in files:
            path = os.path.join(root, f)
            size_mb = os.path.getsize(path) / (1024**2)
            print(f"   {path} ({size_mb:.2f} MB)")

    # ==============================
    # 3. CARGAR DATASETS
    # ==============================
    train_path = os.path.join(BASE_PATH, 'train.csv')
    test_path  = os.path.join(BASE_PATH, 'test.csv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"No encuentro train.csv en {BASE_PATH}")

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    print(f"\n✅ train: {df_train.shape}")
    print(f"✅ test : {df_test.shape}")

    return df_train, df_test