import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. DEFINICIÓN DEL MLP
# ==========================================
class AttrCalibratorMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(AttrCalibratorMLP, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Salida de Logits (sin sigmoide)
            nn.Linear(hidden_dim // 2, output_dim) 
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. CARGA DE DATOS (Sirve para Train y Test)
# ==========================================
def load_data_for_mlp(excel_path, attributes):
    print(f"Cargando datos desde: {excel_path}...")
    df_prompt = pd.read_excel(excel_path, sheet_name='prompting')
    df_sanity = pd.read_excel(excel_path, sheet_name='sanity_check')

    df_prompt['pos'] = df_prompt['pos'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_prompt['neg'] = df_prompt['neg'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    pos_matrix = np.array(df_prompt['pos'].tolist())
    neg_matrix = np.array(df_prompt['neg'].tolist())
    
    # Concatenamos positivos y negativos
    X = np.hstack([pos_matrix, neg_matrix])
    
    Y_list = []
    for attr in attributes:
        col_name = f"{attr}_pos"
        if col_name in df_sanity.columns:
            Y_list.append(df_sanity[col_name].fillna(0).values)
        else:
            Y_list.append(np.zeros(len(df_sanity)))
            
    Y = np.array(Y_list).T 
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# ==========================================
# 3. ENTRENAMIENTO
# ==========================================
def train_mlp(X, Y, epochs=200, batch_size=32, lr=0.001):
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = AttrCalibratorMLP(input_dim, output_dim).cuda()
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_Y in loader:
            batch_X, batch_Y = batch_X.cuda(), batch_Y.cuda()
            
            optimizer.zero_grad()
            logits = model(batch_X)       
            loss = criterion(logits, batch_Y) 
            loss.backward()               
            optimizer.step()              
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f}")
            
    print("✅ Entrenamiento completado.\n")
    return model

# ==========================================
# 4. EVALUACIÓN (TESTING)
# ==========================================
def evaluate_mlp(model, X_test, Y_test, attributes):
    print("Iniciando evaluación en el conjunto de Test...")
    
    # 1. Ponemos el modelo en modo evaluación (desactiva el Dropout)
    model.eval()
    
    # 2. Apagamos el cálculo de gradientes (ahorra memoria y es más rápido)
    with torch.no_grad():
        X_test = X_test.cuda()
        
        # 3. Pasamos los datos por el modelo
        logits = model(X_test)
        
        # 4. Transformamos los logits a probabilidades (0 a 1)
        probabilidades = torch.sigmoid(logits)
        
        # 5. Aplicamos el umbral de 0.5 para decidir si es 1 o 0
        predicciones = (probabilidades > 0.5).int().cpu().numpy()
        
    Y_real = Y_test.numpy()
    
    # 6. Calculamos y mostramos el reporte de métricas
    print("\n" + "="*50)
    print("📊 REPORTE DE MÉTRICAS MULTI-ETIQUETA")
    print("="*50)
    
    # Exactitud global estricta (Match exacto de todos los atributos)
    exact_match_acc = accuracy_score(Y_real, predicciones)
    print(f"Exact Match Ratio (Todas las etiquetas correctas a la vez): {exact_match_acc*100:.2f}%\n")
    
    # Reporte detallado por atributo (zero_division=0 evita warnings si un atributo no aparece nunca en el test)
    report = classification_report(Y_real, predicciones, target_names=attributes, zero_division=0)
    print(report)

import json
    
# ==========================================
# EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    # Sustituye por tu lista de atributos real
    mis_atributos = ["Female", "AgeLess18", "Hat"] 
    TRAIN_EXCEL = "/mnt/rhome/paa/pedestrian/synthetic-pseudolabeling/rapzs_lora_4_with_transform_256_192_loss_train_model_21/RAPzs_fixed-rule_blip_gemini_scores/scores_train.xlsx" # <-- CAMBIA ESTO
    JSON_PATH = "/mnt/rhome/paa/pedestrian/synthetic-pseudolabeling/src/stage_b_scoring/prompting/rapzs_gemini_negative.json" # <-- CAMBIA ESTO
    TEST_EXCEL = "/mnt/rhome/paa/pedestrian/synthetic-pseudolabeling/rapzs_lora_4_with_transform_256_192_loss_train_model_21/RAPzs_fixed-rule_blip_gemini_scores/scores_test.xlsx"     # El excel nuevo

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        prompts_dict = json.load(f)
        mis_atributos = list(prompts_dict.keys())

    # 1. Entrenar
    X_train, Y_train = load_data_for_mlp(TRAIN_EXCEL, mis_atributos)
    modelo_entrenado = train_mlp(X_train, Y_train, epochs=200)
    
    # 2. Testear
    X_test, Y_test = load_data_for_mlp(TEST_EXCEL, mis_atributos)
    evaluate_mlp(modelo_entrenado, X_test, Y_test, mis_atributos)
    
    # 3. (Opcional) Guardar el modelo si los resultados son buenos
    torch.save(modelo_entrenado.state_dict(), "mlp_calibrador_final.pth")