import sys
from mlflow.tracking import MlflowClient

def get_best_metric(experiment_name, run_name, metric_name="testing_ma_605", tracking_uri="./mlruns"):
    client = MlflowClient(tracking_uri=tracking_uri)
    
    # 1. Buscar el experimento por su nombre exacto
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"{experiment_name},{run_name},ERROR: Experimento no encontrado")
        sys.exit(1)
        
    exp_id = experiment.experiment_id
    
    # 2. Buscar el run por su nombre SOLO dentro de ese experimento
    query = f"tags.`mlflow.runName` = '{run_name}'"
    runs = client.search_runs(experiment_ids=[exp_id], filter_string=query)
    
    if not runs:
        print(f"{experiment_name},{run_name},ERROR: Run no encontrado en este exp.")
        sys.exit(1)
        
    # Si por casualidad lanzaste el mismo run dos veces, coge el más reciente
    run_id = runs[0].info.run_id
    
    # 3. Extraer el historial y sacar el máximo
    try:
        historial = client.get_metric_history(run_id, metric_name)
        if not historial:
            print(f"{experiment_name},{run_name},ERROR: Métrica '{metric_name}' vacía")
            sys.exit(1)
            
        max_valor = max([m.value for m in historial])
        
        # Salida limpia en formato CSV
        print(f"{experiment_name},{run_name},{max_valor:.4f}")
        
    except Exception as e:
        print(f"{experiment_name},{run_name},ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ahora pedimos 2 argumentos obligatorios en lugar de 1
    if len(sys.argv) < 3:
        print("Uso: python get_best_ma.py <nombre_experimento> <nombre_run> [nombre_metrica]")
        sys.exit(1)
        
    e_name = sys.argv[1]
    r_name = sys.argv[2]
    # El tercer argumento (opcional) es la métrica
    m_name = sys.argv[3] if len(sys.argv) > 3 else "testing_ma_605"
    
    get_best_metric(e_name, r_name, m_name)