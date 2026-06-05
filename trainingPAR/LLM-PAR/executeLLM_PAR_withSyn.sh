

#!/bin/bash

# 1. Cargar el entorno correctamente
# En scripts .sh, a veces 'conda activate' falla si no inicializas shell
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate llmpar

# 2. Definir variables de entorno
export CSIC_REAL="/home/ubuntu/data/CSIC_REAL"
export CSIC_SYN="/home/ubuntu/data/CSIC_SYN"

# 3. Ruta base del experimento sintético (para no repetir el churro de nombre)
# OJO: Verifica si tras el unzip la carpeta se llama EXACTAMENTE así
SYN_BASE_DIR="$CSIC_SYN/peta_lora_32_with_transform_256_192_loss_train_model_21_reb_forlabeling_por2"

echo "=== Iniciando Entrenamiento PETA con Datos Sintéticos ==="

time python train.py \
    --dataset PETA \
    --use_synthetic \
    --exp PETA_syn \
    --synthetic_img_dir "$SYN_BASE_DIR/generatedImgs/" \
    --pseudolabels_csv "$SYN_BASE_DIR/generated.csv" \
    --prompts_csv "$SYN_BASE_DIR/prompts.csv"

echo "=== Entrenamiento finalizado ==="

#python train.py --dataset PETA --exp PETA_baseline
