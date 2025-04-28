import scipy.io
import pandas as pd
import numpy as np

def convert_mat_to_csv(mat_file_path, csv_file_path):
    """
    Converte tutte le variabili numeriche 2D da un file .mat a un file .csv.
    Se ci sono più variabili, verranno concatenate in un unico DataFrame.

    Args:
        mat_file_path (str): Percorso del file .mat di input.
        csv_file_path (str): Percorso del file .csv di output.
    """
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        dfs = []
        for var_name, var_value in mat_data.items():
            if not var_name.startswith('__') and isinstance(var_value, (pd.DataFrame, pd.Series, np.ndarray)):
                if isinstance(var_value, np.ndarray) and var_value.ndim == 2:
                    df = pd.DataFrame(var_value)
                    dfs.append(df)
                elif isinstance(var_value, pd.Series):
                    df = pd.DataFrame(var_value)
                    dfs.append(df)
                elif isinstance(var_value, pd.DataFrame):
                    dfs.append(var_value)

        if dfs:
            combined_df = pd.concat(dfs, axis=1, ignore_index=True)
            combined_df.to_csv(csv_file_path, index=False)
            print(f"Variabili numeriche 2D da '{mat_file_path}' convertite con successo in '{csv_file_path}'.")
        else:
            print(f"Avviso: Nessuna variabile numerica 2D trovata nel file '{mat_file_path}' da convertire in CSV.")

    except FileNotFoundError:
        print(f"Errore: File .mat non trovato al percorso: {mat_file_path}")
    except Exception as e:
        print(f"Errore durante la conversione: {e}")

if __name__ == '__main__':
    # Sostituisci con il percorso del tuo file .mat
    name = 'cars_train_annos'
    mat_file = name + '.mat'
    csv_file = name + '.csv'

    convert_mat_to_csv(mat_file, csv_file)