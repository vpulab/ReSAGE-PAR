# 🚶‍♂️ Pedestrian Attribute Recognition (PAR) Framework

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)

A robust and modular framework for Pedestrian Attribute Recognition (PAR). This repository implements state-of-the-art methods and novel diffusion-based data augmentation techniques.

> **Note:** This codebase is built upon the strong foundation of [Rethinking_of_PAR](https://github.com/valencebond/Rethinking_of_PAR). We extend it with synthetic data support and additional features.

---

## 🚀 Key Features

* **Multi-Dataset Support:** Compatible with PA100k, PETA, RAPv1, RAPv2, etc.
* **Flexible Backbones:** Support for ResNet50, BNInception, Swin Transformer.
* **Comprehensive Evaluation:** mA, F1-score, and attribute-level analysis.
* **🆕 Synthetic Data Support:** Hybrid training with generated images and verified pseudo-labels.

---

## 🛠️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/PAyuso/Rethinking_of_PAR_paa.git](https://github.com/PAyuso/Rethinking_of_PAR_paa.git)
    cd Rethinking_of_PAR_paa
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare the Datasets:**
    Create a `data/` directory in the root of the project (if it doesn't exist) and place your downloaded datasets inside. The folder structure must exactly match the following layout for the data loaders to work correctly:

    ```text
    trainingPAR/
    ├── data/
    │   ├── PA100k/
    │   ├── PETA/
    │   ├── RAP/
    │   └── RAP2/
    ├── configs/
    ├── models/
    └── ...
    ```
    *(Note: You only need to include the folders for the specific datasets you intend to train or evaluate on).*
---

## 🎨 Synthetic Data Usage (New)

We have integrated a new functionality that allows augmenting the training set using **synthetic images** (e.g., generated via Stable Diffusion + LoRA) and their corresponding verified **pseudo-labels**.

This feature enables mixing real and synthetic data to improve model generalization, particularly for tail classes.

### ⚙️ Configuration in `config.yaml`

To enable synthetic data usage, add or modify the `SYNTHETIC` block in your `.yaml` configuration file.

#### Parameters:
* **USE:** (`True`/`False`) Activates the synthetic data loader.
* **PERCENTAGE:** (`float`) The ratio of synthetic data relative to the real dataset size (e.g., `1.0` for 1:1 ratio, `0.5` for 1:0.5).
* **PSEUDOLABELS_CSV:** Path to the CSV file containing the verified attributes (0, 1, or -1 for masked).
* **PATH:** Path to the folder containing the generated images.

#### Example Configuration:

```yaml
SYNTHETIC:
  USE: True
  PERCENTAGE: 1.0
  PSEUDOLABELS_CSV: "/mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/pa100k_lora_32_with_transform_256_192_loss_train_model_21_reb_forlabeling_por2/pseudolabels_with-1.csv"
  PATH: "/mnt/rhome/paa/pedestrian/dataAugmentationMethods/LORA/diffusers/examples/text_to_image/pa100k_lora_32_with_transform_256_192_loss_train_model_21_reb_forlabeling_por2/generatedImgs/"
```

### 📊 Pseudo-label Structure
The PSEUDOLABELS_CSV file must follow the structure expected by the dataloader. Ensure columns match the target dataset attributes.

* 1: Attribute present.

* 0: Attribute absent.

* -1: Attribute ignored/masked (used to filter out generative hallucinations or uncertain predictions).

---

## 📊 MLflow Integration

Este repositorio utiliza **MLflow** para registrar y monitorizar experimentos. Sigue estos pasos para configurar y ejecutar entrenamientos con MLflow.

### 1) Crear carpeta para runs

Crea un directorio donde se guardarán las ejecuciones de MLflow:

```bash
mkdir mlruns
```

### 2) Lanzar el servidor de MLflow

En una terminal separada, ejecuta el servidor MLflow apuntando a la carpeta de runs:

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns
```

**Explicación:**
- `--host 0.0.0.0`: Escucha en todas las interfaces de red
- `--port 5000`: Puerto donde corre la UI
- `--backend-store-uri ./mlruns`: Directorio donde se guardan los datos de runs

Accede a la UI en `http://localhost:5000` o `http://<IP_DEL_SERVIDOR>:5000`

### 3) Configurar el cliente

En otra terminal (donde ejecutarás los entrenamientos), configura la variable de entorno:

```bash
export MLFLOW_TRACKING_URI=http://0.0.0.0:5000
```

O si MLflow corre en otro servidor:
```bash
export MLFLOW_TRACKING_URI=http://<IP_DEL_SERVIDOR>:5000
```

### 4) Ejecutar un experimento con MLflow

El archivo `MLproject` ya está configurado para ejecutar entrenamientos. Usa el siguiente comando:

```bash
mlflow run . --experiment-name=rapzs --run-name=baseline_bin --env-manager=local \
  -P cfg=./configs/pedes_baseline/rap_zsgithub_baseline.yaml
```

**Significado de cada parámetro:**
- `mlflow run .`: Ejecuta el proyecto MLflow del directorio actual
- `--experiment-name=rapzs`: Nombre del experimento para agrupar runs
- `--run-name=baseline_bin`: Nombre único del run (visible en la UI)
- `--env-manager=local`: Usa el entorno local actual (sin crear uno nuevo)
- `-P cfg=...`: Parámetro del proyecto; ruta al archivo de configuración YAML

### 5) Crear entornos conda

Hay dos definiciones de entornos en la carpeta `conda/`:

- `environment.yaml`: Entorno base para entrenamientos normales
- `environment_dataaug.yaml`: Entorno con dependencias adicionales para MixUp/CutMix

Para crear los entornos:

```bash
conda env create -f conda/environment.yaml
conda env create -f conda/environment_dataaug.yaml
```

Activa el entorno según tus necesidades:
```bash
conda activate rethinking          # Para entrenamientos base
conda activate rethinking_dataaug   # Para entrenamientos con MixUp/CutMix
```

### 6) Data Augmentation Avanzada (MixUp/CutMix)

Si deseas usar **MixUp** o **CutMix** en tu configuración:

1. **Asegúrate de estar en el entorno adecuado:**
   ```bash
   conda activate rethinking_dataaug
   ```

2. **Configura tu archivo YAML** con `ENABLE: true` bajo `TRAIN.DATAAUG`:
   ```yaml
   TRAIN:
     DATAAUG:
       TYPE: 'mixup'  # o 'cutmix'
       ENABLE: true
       NUMCLASSES: 53  # Número de atributos del dataset
   ```

3. **Ejecuta con MLflow como habitualmente:**
   ```bash
   export MLFLOW_TRACKING_URI=http://<IP_DEL_SERVIDOR>:5000
   mlflow run . --experiment-name=rapzs --run-name=baseline_mixup --env-manager=local \
     -P cfg=./configs/pedes_baseline/rebuttal/rap_zsgithubLRsave_BIN_mixup.yaml
   ```

**Nota:** Sin `ENABLE: true` en la configuración, MixUp/CutMix no se activarán aunque esté configurado el TYPE.

### 7) Monitorizando Experimentos

Una vez que los entrenamientos están en marcha:

1. Abre la UI de MLflow en `http://<IP_DEL_SERVIDOR>:5000`
2. Selecciona el experiment (ej: `rapzs`)
3. Visualiza métricas, parámetros y artefactos de cada run
4. Compara múltiples runs lado a lado
