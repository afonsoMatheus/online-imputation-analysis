import pandas as pd
import numpy as np
import os
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
from river import stats, linear_model, preprocessing, neighbors, neural_net, optim, tree
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

imputers = ['mean', 'reg', 'mlp', 'tree']
# imputers = ['mlp']    

def tree_input(df):

    df_inputed = df[['datetime', 'heartrate', 'target']].copy()

    model = (
        preprocessing.StandardScaler() |
        tree.HoeffdingTreeRegressor()
    )

    target = df.loc[df['heartrate'].isna(), 'target'].tolist()
    heartrate_imputed = []
    # for _, row in tqdm(df.iterrows(), total=len(df), desc="Imputing with tree"):
    for _, row in df.iterrows():
        x = {"hour": row['hour'], "minute": row['minute'], "second": row['second']}
        y = row['heartrate']

        if np.isnan(y):
            y_pred = model.predict_one(x) or -1
            heartrate_imputed.append(round(y_pred, 0))
            # model.learn_one(x, round(y_pred, 1))

        else:
            model.learn_one(x, y)

    rmse = root_mean_squared_error(target, heartrate_imputed)
    df_inputed.loc[df_inputed['heartrate'].isna(), 'heartrate'] = heartrate_imputed
    return rmse, df_inputed


def mlp_input(df):

    df_inputed = df[['datetime', 'heartrate', 'target']].copy()

    model = preprocessing.StandardScaler() | neural_net.MLPRegressor(
        hidden_dims=(5,),
        activations=(
            neural_net.activations.ReLU,
            neural_net.activations.ReLU,
            neural_net.activations.Identity
        ),
        # optimizer=optim.SGD(0.001),
        optimizer=optim.Adam(0.1),
        seed=1
    )
    
    target = df.loc[df['heartrate'].isna(), 'target'].tolist()
    heartrate_imputed = []

    # for _, row in tqdm(df.iterrows(), total=len(df), desc="Imputing with MLP"):
    for _, row in df.iterrows():
        x = {"hour": row['hour'], "minute": row['minute'], "second": row['second']}
        y = row['heartrate']

        if np.isnan(y):
            y_pred = model.predict_one(x) or -1
            heartrate_imputed.append(round(y_pred, 0))
        else:
            model.learn_one(x, y)

    df_inputed.loc[df_inputed['heartrate'].isna(), 'heartrate'] = heartrate_imputed
    rmse = root_mean_squared_error(target, heartrate_imputed)
    return rmse, df_inputed

def reg_input(df):

    df_inputed = df[['datetime', 'heartrate', 'target']].copy()

    model = preprocessing.StandardScaler() | linear_model.LinearRegression()

    target = df.loc[df['heartrate'].isna(), 'target'].tolist()
    heartrate_imputed = []

    # for _, row in tqdm(df.iterrows(), total=len(df), desc="Imputing with regression"):
    for _, row in df.iterrows():
        x = {"hour": row['hour'], "minute": row['minute'], "second": row['second']}
        y = row['heartrate']

        if np.isnan(y):
            y_pred = model.predict_one(x) or -1
            heartrate_imputed.append(round(y_pred, 0))
            # model.learn_one(x, round(y_pred, 1))

        else:
            model.learn_one(x, y)

    rmse = root_mean_squared_error(target, heartrate_imputed)
    df_inputed.loc[df_inputed['heartrate'].isna(), 'heartrate'] = heartrate_imputed
    return rmse, df_inputed


def mean_input(df):

    df_inputed = df[['datetime', 'heartrate', 'target']].copy()

    model = stats.Mean()

    heartrate_imputed = []
    target = df.loc[df['heartrate'].isna(), 'target'].tolist()
    # for _, row in tqdm(df.iterrows(), total=len(df), desc="Imputing with mean"):
    for _, row in df.iterrows():
        y = row['heartrate']

        if np.isnan(y):
            y_pred = model.get() or -1
            heartrate_imputed.append(round(y_pred, 0))
        else:
            model.update(y)
            # heartrate_imputed.append(y)

    rmse = root_mean_squared_error(target, heartrate_imputed)
    df_inputed.loc[df_inputed['heartrate'].isna(), 'heartrate'] = heartrate_imputed
    return rmse, df_inputed

def knn_input(df):

    df_inputed = df[['datetime', 'heartrate', 'target']].copy()

    # capture the true targets for rows where heartrate is missing
    target = df.loc[df['heartrate'].isna(), 'target'].tolist()

    model = neighbors.KNNRegressor(n_neighbors=5)

    heartrate_imputed = []
    # for _, row in tqdm(df.iterrows(), total=len(df), desc="Imputing with KNN"):
    for _, row in df.iterrows():
        x = {"hour": row['hour'], "minute": row['minute'], "second": row['second']}
        y = row['heartrate']

        if np.isnan(y):
            y_pred = model.predict_one(x) or -1
            heartrate_imputed.append(round(y_pred, 0))
            # model.learn_one(x, round(y_pred, 1))

        else:
            model.learn_one(x, y)

    df_inputed.loc[df_inputed['heartrate'].isna(), 'heartrate'] = heartrate_imputed
    rmse = root_mean_squared_error(target, heartrate_imputed)
    return rmse, df_inputed

def run_imputer(imp_name, imp_func, df):
    import time
    start_time = time.time()
    rmse, imp_df = imp_func(df)
    elapsed = time.time() - start_time
    # print(f"Imputer {imp_name} completed in {elapsed:.2f}")
    return imp_name, rmse, elapsed, imp_df

def process_single_mr(mech, mr, i, pat, imputers, folder_path_m, folder_path_imputed):
    """Função auxiliar que processa um único MR em um processo separado."""
    local_result = {mr: {f"{imp}": 0 for imp in imputers}}
    local_result[mr].update({f"t_{imp}": 0 for imp in imputers})

    try:
        files_hr = [
            f for f in os.listdir(folder_path_m)
            if f.endswith(f"_hr_{mech}_{i}_{mr}.csv")
        ]
        if not files_hr:
            print(f"⚠️ Nenhum arquivo encontrado para {mr} em {pat}")
            return local_result

        df = pd.read_csv(os.path.join(folder_path_m, files_hr[0]))
        first_valid_idx = df['heartrate'].first_valid_index()
        df = df.loc[first_valid_idx:].reset_index(drop=True)
        # df = df.iloc[:20000]

        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['second'] = df['datetime'].dt.second

        imputers_list = [
            ('mean', mean_input),
            ('reg', reg_input),
            ('mlp', mlp_input),
            ('tree', tree_input)
        ]

        for name, func in imputers_list:
            try:
                imp_name, rmse, elapsed, imp_df = run_imputer(name, func, df.copy())

                local_result[mr][imp_name] += rmse
                local_result[mr][f"t_{imp_name}"] += elapsed

                path_imp = os.path.join(folder_path_imputed, f"{imp_name}")
                os.makedirs(path_imp, exist_ok=True)

                imp_df.to_csv(
                    os.path.join(
                        path_imp,
                        f"{pat.rstrip('/').split('/')[-1]}_hr_{mech}_{i}_{mr}_{imp_name}.csv"
                    ),
                    index=False
                )

                print(f"✅ Processed {pat.rstrip('/').split('/')[-1]} | Mechanism: {mech} | MR: {mr} | Dataset: {i} | Imputer: {imp_name}")

            except Exception as e:
                print(f"❌ Error running imputer {name} on {pat} | {mech} | {mr} | Dataset {i}: {e}")

    except Exception as e:
        print(f"❌ Erro ao processar {pat.rstrip('/').split('/')[-1]} | Mechanism: {mech} | MR: {mr} | Dataset: {i} | Error: {e}")

    return local_result


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
    patients = patients[:3]  # Limitar ao primeiro paciente

    for i in range(1, num_datasets + 1):
        for pat in tqdm(patients, desc=f"Processing mechanism {mech}"):
            folder_path_imputed = os.path.join(
                os.path.dirname(__file__), 
                f"../../Data/COVID-19-Wearables-Input/{mech}/{pat.rstrip('/').split('/')[-1]}/{i}/"
            )
            os.makedirs(folder_path_imputed, exist_ok=True)

            folder_path_m = f"{pat}/{i}"

            # ⚡ Executa cada MR em paralelo
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        process_single_mr, mech, mr, i, pat, imputers, folder_path_m, folder_path_imputed
                    ): mr for mr in mrs
                }

                # for future in tqdm(as_completed(futures), total=len(futures), desc="MRs completed", leave=False):
                for future in as_completed(futures):
                    mr = futures[future]
                    try:
                        result = future.result()
                        for mr_key, mr_dict in result.items():
                            if mr_key not in local_results[mech]:
                                local_results[mech][mr_key] = mr_dict.copy()
                            else:
                                for k, v in mr_dict.items():
                                    local_results[mech][mr_key][k] = local_results[mech][mr_key].get(k, 0) + v

                    except Exception as e:
                        print(f"❌ Falha no MR {mr}: {e}")

    # --- Normalização ---
    for mr in local_results[mech]:
        for imp in imputers:
            # print(f"Acummulated {mech} | {mr} | {imp}: {local_results[mech][mr][imp]}")
            local_results[mech][mr][imp] /= len(patients) * num_datasets
            # print(f"Normalized {mech} | {mr} | {imp}: {local_results[mech][mr][imp]}")
            local_results[mech][mr][f"t_{imp}"] /= len(patients) * num_datasets
            # print()

    return local_results

# ---- Execução principal ---- #
if __name__ == "__main__":
    mechanisms_batchs = [["MNAR_l", "MNAR_m","MNAR_h"],["MAR_l", "MAR_m", "MAR_h"],["MCAR"]]
    # mechanisms_batchs = [["MCAR"]]
    mrs = ["05", "10", "15", "20", "25"]
    num_datasets = 1

    combined_results = {}

    total_time_start = time.time()

    for mechanisms in mechanisms_batchs:
        time_start = time.time()
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

            time_elapsed = time.time() - time_start
            hours = time_elapsed / 3600
            print(f"\n⏱️ Tempo de execução para mecanismos {mechanisms}: {hours:.2f} horas")

    # ---- Gera DataFrame final ---- #
            records = []
            for mech in combined_results:
                for mr in combined_results[mech]:
                    for imp in imputers:
                        records.append({
                            'mechanism': mech,
                            'missing_rate': mr,
                            'imputer': imp,
                            'rmse': round(combined_results[mech][mr][imp], 2),
                            'time': round(combined_results[mech][mr][f't_{imp}'], 2)
                        })

            results_df = pd.DataFrame(records)
            result_path = f'Analysis/imputation_results.csv'
            if os.path.exists(result_path):
                os.remove(result_path)
            results_df.to_csv(result_path, index=False)
            
            critical = []
            for imp in imputers:
                for mech in combined_results:
                    for mr in combined_results[mech]:
                        critical.append({
                            'classifier_name': imp,
                            'dataset_name': f"{mech}_{mr}",
                            'accuracy': combined_results[mech][mr][imp]
                        })

            critical_df = pd.DataFrame(critical)
            critical_path = f'CD_Diagram/imputation_critical.csv'
            if os.path.exists(critical_path):
                os.remove(critical_path)
            critical_df.to_csv(critical_path, index=False)

    total_time_elapsed = time.time() - total_time_start
    hours = total_time_elapsed / 3600
    print(f"\n⏱️ Tempo total de execução: {hours:.2f} horas")







