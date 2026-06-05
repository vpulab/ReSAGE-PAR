import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import ast
import os

def add_jitter(arr, amount=0.03):
    """
    Jitter aumentado al 3% para esparcir más las "nubes" de puntos
    """
    return arr + np.random.uniform(-amount, amount, size=len(arr))

def plot_blip_2d_separability_debug(excel_path, json_path, output_dir="plots_2d"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_path, 'r', encoding='utf-8') as f:
        prompts_dict = json.load(f)
        attributes = list(prompts_dict.keys())

    df_prompting = pd.read_excel(excel_path, sheet_name='prompting')
    df_sanity = pd.read_excel(excel_path, sheet_name='sanity_check')

    df_prompting['pos'] = df_prompting['pos'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_prompting['neg'] = df_prompting['neg'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    pos_matrix = np.array(df_prompting['pos'].tolist())
    neg_matrix = np.array(df_prompting['neg'].tolist())

    num_images = pos_matrix.shape[0]
    print(f"✅ Excel cargado. Total de filas (imágenes) detectadas: {num_images}")
    print("-" * 50)

    for i, attr in enumerate(attributes):
        if i >= pos_matrix.shape[1]:
            continue

        pos_scores = pos_matrix[:, i]
        neg_scores = neg_matrix[:, i]

        gt_col_name = f"{attr}_pos"
        if gt_col_name not in df_sanity.columns:
            continue

        # Forzamos a que trate todo como float para cazar posibles NaNs
        gt_labels = df_sanity[gt_col_name].fillna(-1).values 
        
        mask_present = (gt_labels == 1)
        mask_absent = (gt_labels == 0)

        # ====== CHIVATO DE DEBUG ======
        num_verdes = np.sum(mask_present)
        num_rojos = np.sum(mask_absent)
        total_puntos = num_verdes + num_rojos
        print(f"Atributo '{attr}': Dibujando {num_verdes} Verdes y {num_rojos} Rojos. Total = {total_puntos}")
        
        if total_puntos != num_images:
            print(f"  ⚠️ CUIDADO: Faltan {num_images - total_puntos} muestras. Revisa si hay celdas vacías en la columna '{gt_col_name}'.")
        # ==============================

        fig, ax = plt.subplots(figsize=(8, 8))

        # Aumentamos Jitter (0.03)
        x_present_jittered = add_jitter(neg_scores[mask_present], amount=0.03)
        y_present_jittered = add_jitter(pos_scores[mask_present], amount=0.03)
        
        x_absent_jittered = add_jitter(neg_scores[mask_absent], amount=0.03)
        y_absent_jittered = add_jitter(pos_scores[mask_absent], amount=0.03)

        # Reducimos tamaño (s=25) y bajamos opacidad (alpha=0.4) 
        # para que 10 puntos superpuestos se vean como una mancha intensa, pero separada
        ax.scatter(x_present_jittered, y_present_jittered, 
                   color='#2ca02c', alpha=0.4, s=25, edgecolors='white', linewidth=0.3, label=f'ESTÁ (GT=1) [{num_verdes}]')
        
        ax.scatter(x_absent_jittered, y_absent_jittered, 
                   color='#d62728', alpha=0.4, s=25, edgecolors='white', linewidth=0.3, label=f'NO ESTÁ (GT=0) [{num_rojos}]')

        #ax.plot([-0.1, 1.1], [-0.1, 1.1], 'k--', alpha=0.3, label='Línea de Incertidumbre')

        ax.set_title(f'Separabilidad 2D: {attr}', fontsize=15, fontweight='bold', pad=15)
        ax.set_xlabel('Negative Probe Score ("Not In")', fontsize=12)
        ax.set_ylabel('Positive Probe Score ("In")', fontsize=12)
        
        # Ampliamos un poco más los límites para que el jitter no se corte
        #ax.set_xlim(-0.1, 1.1)
        #ax.set_ylim(-0.1, 1.1)
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', fontsize=10)

        safe_attr_name = attr.replace("/", "_").replace("\\", "_")
        plot_filename = os.path.join(output_dir, f"{safe_attr_name}_2d.png")
        
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300)
        plt.close()


# ==========================================
# EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    EXCEL_PATH = "/mnt/rhome/paa/pedestrian/synthetic-pseudolabeling/rapzs_lora_4_with_transform_256_192_loss_train_model_21/RAPzs_fixed-rule_blip_gemini_scores/scores_train.xlsx" # <-- CAMBIA ESTO
    JSON_PATH = "/mnt/rhome/paa/pedestrian/synthetic-pseudolabeling/src/stage_b_scoring/prompting/rapzs_gemini_negative.json" # <-- CAMBIA ESTO
    
    plot_blip_2d_separability_debug(excel_path=EXCEL_PATH, json_path=JSON_PATH)