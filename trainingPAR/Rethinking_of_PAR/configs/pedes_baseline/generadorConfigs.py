import os

# La plantilla base con los huecos listos para rellenar
template = """NAME: '{backbone}.base.adam'

SYNTHETIC:
  USE_MALS: True
  PERCENTAGE: 1.0
  MALS_PATH: '/mnt/rhome/paa/pedestrian/dataAugmentationMethods/datasets/realOnes/MALS/'

DATASET:
  TYPE: 'pedes'
  NAME: '{real_dataset_name}'
  TRAIN_SPLIT: 'trainval'
  VAL_SPLIT: 'test'
  ZERO_SHOT: {zero_shot}
  LABEL: 'eval'
  HEIGHT: 256
  WIDTH: 192
  ROOT: '{root_path}'

RELOAD:
  TYPE: False
  NAME: 'backbone'

BACKBONE:
  TYPE: '{backbone}'

CLASSIFIER:
  NAME: 'linear'
  POOLING: 'avg'
  SCALE: 1
  BN: False

LOSS:
  TYPE: 'bceloss'
  LOSS_WEIGHT: [1]
  SAMPLE_WEIGHT: 'weight'

TRAIN:
  CLIP_GRAD: True
  BATCH_SIZE: 64
  OPTIMIZER:
    TYPE: 'adam'
    WEIGHT_DECAY: 5e-4
  LR_SCHEDULER:
    TYPE: 'plateau'
    LR_FT: 0.0001
    LR_NEW: 0.0001
  EMA:
    ENABLE: False

METRIC:
  TYPE: 'pedestrian'

VIS:
  CAM: 'valid'
"""

# Tus datasets y sus rutas exactas
datasets = [
    # (Prefijo_archivo, DATASET.NAME, ZERO_SHOT, ROOT)
    ("peta", "PETA", "False", "/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/PETA/images/"),
    ("petazs", "PETA", "True", "/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/PETA/images/"),
    ("rap_v1", "RAP", "False", "/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/RAP/RAP_dataset/"),
    ("rap_v2", "RAP2", "False", "/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/RAP2/RAP_dataset/"),
    ("rap_zs", "RAP2", "True", "/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/RAP2/RAP_dataset/"),
    ("pa100k", "PA100k", "False", "/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/PA100k/data/")
]

backbones = ["resnet50", "bninception", "swin_s"]

contador = 0

print("Generando archivos de configuración...")
print("-" * 40)

for prefijo, ds_name, zs, root in datasets:
    for bb in backbones:
        # Saltamos el resnet50 de PA100k porque ya lo tienes creado
        if prefijo == "pa100k" and bb == "resnet50":
            continue
            
        # Nombramos el archivo (para resnet50 mantengo tu nomenclatura "github")
        nombre_bb = "github" if bb == "resnet50" else bb
        file_name = f"{prefijo}_{nombre_bb}_baseline_Synx2_MALS.yaml"
            
        # Rellenamos la plantilla
        contenido = template.format(
            backbone=bb,
            real_dataset_name=ds_name,
            zero_shot=zs,
            root_path=root
        )
        
        # Guardamos el archivo
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(contenido)
            
        print(f"✅ Creado: {file_name}")
        contador += 1

print("-" * 40)
print(f"¡Listo! Se han generado {contador} archivos YAML en esta carpeta.")