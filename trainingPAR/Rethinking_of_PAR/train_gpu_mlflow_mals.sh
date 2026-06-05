#!/bin/bash

# ==========================================
# RETHINKING BASELINE: RESNET50
# ==========================================
echo "Iniciando bloque ResNet50..."

mlflow run . --experiment-name=rapzs --run-name=resnet50_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/rap_zs_github_baseline_Synx2_MALS.yaml
sleep 5

mlflow run . --experiment-name=rapv2 --run-name=resnet50_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/rap_v2_github_baseline_Synx2_MALS.yaml
sleep 5

mlflow run . --experiment-name=rapv1 --run-name=resnet50_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/rap_v1_github_baseline_Synx2_MALS.yaml
sleep 5

mlflow run . --experiment-name=petazs --run-name=resnet50_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/petazs_github_baseline_Synx2_MALS.yaml
sleep 5

mlflow run . --experiment-name=peta --run-name=resnet50_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/peta_github_baseline_Synx2_MALS.yaml
sleep 5

mlflow run . --experiment-name=pa100k --run-name=resnet50_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/pa100k_github_baseline_Synx2_MALS.yaml
sleep 5


# ==========================================
# RETHINKING BASELINE: BNINCEPTION
# ==========================================
echo "Iniciando bloque BNInception..."

mlflow run . --experiment-name=rapzs --run-name=bninception_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/rap_zs_bninception_baseline_Synx2_MALS.yaml
sleep 5

mlflow run . --experiment-name=petazs --run-name=bninception_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/petazs_bninception_baseline_Synx2_MALS.yaml
sleep 5

mlflow run . --experiment-name=peta --run-name=bninception_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/peta_bninception_baseline_Synx2_MALS.yaml
sleep 5

mlflow run . --experiment-name=rapv1 --run-name=bninception_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/rap_v1_bninception_baseline_Synx2_MALS.yaml
sleep 5

mlflow run . --experiment-name=rapv2 --run-name=bninception_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/rap_v2_bninception_baseline_Synx2_MALS.yaml
sleep 5

mlflow run . --experiment-name=pa100k --run-name=bninception_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/pa100k_bninception_baseline_Synx2_MALS.yaml
sleep 5


# ==========================================
# RETHINKING BASELINE: SWIN-S
# ==========================================
echo "Iniciando bloque Swin-S..."

mlflow run . --experiment-name=rapzs --run-name=swin_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/rap_zs_swin_s_baseline_Synx2_MALS.yaml
sleep 5

mlflow run . --experiment-name=petazs --run-name=swin_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/petazs_swin_s_baseline_Synx2_MALS.yaml
sleep 5

mlflow run . --experiment-name=peta --run-name=swin_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/peta_swin_s_baseline_Synx2_MALS.yaml
sleep 5

mlflow run . --experiment-name=rapv1 --run-name=swin_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/rap_v1_swin_s_baseline_Synx2_MALS.yaml
sleep 5

mlflow run . --experiment-name=rapv2 --run-name=swin_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/rap_v2_swin_s_baseline_Synx2_MALS.yaml
sleep 5

mlflow run . --experiment-name=pa100k --run-name=swin_mals_x2 --env-manager=local \
  -P cfg=./configs/pedes_baseline/pa100k_swin_s_baseline_Synx2_MALS.yaml

echo "¡Todos los experimentos han finalizado con éxito!"