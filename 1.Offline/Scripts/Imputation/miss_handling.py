import pandas as pd
import numpy as np
import os
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
from river import stats, linear_model, preprocessing, neighbors, neural_net, optim, tree
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def tree_input(df):

    df = df.copy()

    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['second'] = df['datetime'].dt.second
    df['day'] = df['datetime'].dt.day

    model = (
        preprocessing.StandardScaler() |
        tree.HoeffdingTreeRegressor(
            grace_period=100,
            model_selector_decay=0.9
        )
    )

    heartrate_imputed = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Imputing with tree"):
        x = {"day": row['day'], "hour": row['hour'], "minute": row['minute'], "second": row['second']}
        y = row['heartrate']

        if np.isnan(y):
            y_pred = model.predict_one(x) or -1
            heartrate_imputed.append(round(y_pred, 1))
            # model.learn_one(x, round(y_pred, 1))

        else:
            model.learn_one(x, y)
            heartrate_imputed.append(y)

    df['heartrate_imputed'] = heartrate_imputed
    rmse = root_mean_squared_error(df['target'], df['heartrate_imputed'])
    return rmse


def mlp_input(df):
    df = df.copy()
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['second'] = df['datetime'].dt.second

    model = preprocessing.StandardScaler() | neural_net.MLPRegressor(
        hidden_dims=(3,),
        activations=(
            neural_net.activations.ReLU,
            neural_net.activations.ReLU,
            neural_net.activations.Identity
        ),
        optimizer=optim.SGD(0.01),
        seed=1
    )

    heartrate_imputed = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Imputing with MLP"):
        x = {"hour": row['hour'], "minute": row['minute'], "second": row['second']}
        y = row['heartrate']

        if np.isnan(y):
            y_pred = model.predict_one(x) or -1
            heartrate_imputed.append(round(y_pred, 1))
        else:
            model.learn_one(x, y)
            heartrate_imputed.append(y)

    df['heartrate_imputed'] = heartrate_imputed
    rmse = root_mean_squared_error(df['target'], df['heartrate_imputed'])
    return rmse

def reg_input(df):

    df = df.copy()

    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['second'] = df['datetime'].dt.second
    # df['day'] = df['datetime'].dt.day

    model = preprocessing.StandardScaler() | linear_model.LinearRegression()

    heartrate_imputed = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Imputing with regression"):
        x = {"hour": row['hour'], "minute": row['minute'], "second": row['second']}
        y = row['heartrate']

        if np.isnan(y):
            y_pred = model.predict_one(x) or -1
            heartrate_imputed.append(round(y_pred, 1))
            # model.learn_one(x, round(y_pred, 1))

        else:
            model.learn_one(x, y)
            heartrate_imputed.append(y)

    df['heartrate_imputed'] = heartrate_imputed
    rmse = root_mean_squared_error(df['target'], df['heartrate_imputed'])
    return rmse


def mean_input(df):

    df = df.copy()

    model = stats.Mean()

    heartrate_imputed = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Imputing with mean"):
        y = row['heartrate']

        if np.isnan(y):
            y_pred = model.get() or -1
            heartrate_imputed.append(round(y_pred, 1))
        else:
            model.update(y)
            heartrate_imputed.append(y)

    df['heartrate_imputed'] = heartrate_imputed
    rmse = root_mean_squared_error(df['target'], df['heartrate_imputed'])
    return rmse

def knn_input(df):

    df = df.copy()

    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['second'] = df['datetime'].dt.second

    model = neighbors.KNNRegressor(n_neighbors=5)

    heartrate_imputed = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Imputing with KNN"):
        x = {"hour": row['hour'], "minute": row['minute'], "second": row['second']}
        y = row['heartrate']

        if np.isnan(y):
            y_pred = model.predict_one(x) or -1
            heartrate_imputed.append(round(y_pred, 1))
            # model.learn_one(x, round(y_pred, 1))

        else:
            model.learn_one(x, y)
            heartrate_imputed.append(y)

    df['heartrate_imputed'] = heartrate_imputed

    return root_mean_squared_error(df['target'], df['heartrate_imputed'])

def process_mechanism(mech, num_datasets, mrs):
    local_results = {mech: {}}

    folder_path_m_base = os.path.join(
        os.path.dirname(__file__), 
        f"../../Data/COVID-19-Wearables-Missing/{mech}/"
    )

    patients = [
        os.path.join(folder_path_m_base, name)
        for name in os.listdir(folder_path_m_base)
        if os.path.isdir(os.path.join(folder_path_m_base, name))
    ]
    patients = patients[:1]  # Limitar ao primeiro paciente

    for i in range(1, num_datasets + 1):
        for pat in tqdm(patients, desc=f"Processing mechanism {mech}"):
            for mr in tqdm(mrs, desc=f"Processing missing rates for {mech}", leave=False):
                
                if mr not in local_results[mech]:
                    local_results[mech][mr] = {f"{imp}": 0 for imp in ['mean', 'reg', 'knn', 'mlp', 'tree']}
                    local_results[mech][mr].update({f"t_{imp}": 0 for imp in ['mean', 'reg', 'knn', 'mlp', 'tree']})

                folder_path_m = f"{pat}/{i}"
                files_hr = [
                    f for f in os.listdir(folder_path_m)
                    if f.endswith(f"_hr_{mech}_{i}_{mr}.csv")
                ]

                df = pd.read_csv(os.path.join(folder_path_m, files_hr[0]))
                df = df.iloc[:1000]
                df['datetime'] = pd.to_datetime(df['datetime'])

                # --- Chamada da sua função de imputação ---
                for imp_name, imp_func in [
                    ('mean', mean_input),
                    ('reg', reg_input),
                    ('knn', knn_input),
                    ('mlp', mlp_input),
                    ('tree', tree_input)
                ]:
                    start_time = time.time()
                    local_results[mech][mr][imp_name] += imp_func(df)
                    local_results[mech][mr][f't_{imp_name}'] = time.time() - start_time

    # --- Normalização ---
    for mr in local_results[mech]:
        for imp in ['mean', 'reg', 'knn', 'mlp', 'tree']:
            local_results[mech][mr][imp] /= len(patients) * num_datasets
            local_results[mech][mr][f't_{imp}'] /= len(patients) * num_datasets

    return local_results


# ---- Execução principal ---- #
if __name__ == "__main__":
    mechanisms = ["MCAR", "MAR_l", "MAR_m", "MAR_h", "MNAR_l", "MNAR_m", "MNAR_h"]
    # mechanisms = ["MCAR"]
    mrs = ["05", "10", "15", "20", "25"]
    num_datasets = 1

    combined_results = {}

    with tqdm(total=len(mechanisms), desc="Progresso geral", position=0) as global_pbar:
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(process_mechanism, mech, num_datasets, mrs): mech
                for mech in mechanisms
            }
            for future in as_completed(futures):
                mech_name = futures[future]
                try:
                    result = future.result()
                    combined_results.update(result)
                except Exception as e:
                    print(f"\n❌ Erro ao processar {mech_name}: {e}")
                finally:
                    global_pbar.update(1)

    # ---- Gera DataFrame final ---- #
    records = []
    for mech in combined_results:
        for mr in combined_results[mech]:
            for imp in ['mean', 'reg', 'knn', 'mlp', 'tree']:
                records.append({
                    'mechanism': mech,
                    'missing_rate': mr,
                    'imputer': imp,
                    'rmse': combined_results[mech][mr][imp],
                    'time': combined_results[mech][mr][f't_{imp}']
                })

    results_df = pd.DataFrame(records)
    results_df.to_csv('imputation_results.csv', index=False)
    
    critical = []
    for clf in ['mean', 'reg', 'knn', 'mlp', 'tree']:
        for mech in combined_results:
            for mr in combined_results[mech]:
                critical.append({
                    'classifier_name': clf,
                    'dataset_name': f"{mech}_{mr}",
                    'accuracy': combined_results[mech][mr][clf]
                })
    critical_df = pd.DataFrame(critical)
    critical_df.to_csv('imputation_critical.csv', index=False)

    print("\n✅ Resultados salvos em 'imputation_results.csv'")








