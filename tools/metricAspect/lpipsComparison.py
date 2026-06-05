import os
import argparse
import torch
import lpips
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def calculate_average_lpips(dir_real, dir_syn, dataset_name):
    print(f"Cargando modelo LPIPS (VGG) para el dataset: {dataset_name}...")
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_vgg.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    valid_ext = ('.png', '.jpg', '.jpeg')
    real_files = [f for f in os.listdir(dir_real) if f.lower().endswith(valid_ext)]
    
    total_lpips = 0.0
    count = 0
    
    print(f"Se encontraron {len(real_files)} imágenes reales en 'cond'. Calculando...")
    
    # Hemos añadido el nombre del dataset a la barra de carga
    for filename in tqdm(real_files, desc=f"Calculando LPIPS [{dataset_name}]"):
        img_real_path = os.path.join(dir_real, filename)
        img_syn_path = os.path.join(dir_syn, filename)
        
        if not os.path.exists(img_syn_path):
            continue
            
        img_real = Image.open(img_real_path).convert('RGB')
        img_syn = Image.open(img_syn_path).convert('RGB')
        
        tensor_real = transform(img_real).unsqueeze(0).to(device)
        tensor_syn = transform(img_syn).unsqueeze(0).to(device)
        
        with torch.no_grad():
            d = loss_fn_vgg(tensor_real, tensor_syn)
            
        total_lpips += d.item()
        count += 1
        
    if count == 0:
        print(f"¡Error! No se encontraron parejas de imágenes válidas para {dataset_name}.")
        return
        
    avg_lpips = total_lpips / count
    
    print("\n" + "="*50)
    print(f"📊 RESULTADOS PARA: {dataset_name.upper()}")
    print("-" * 50)
    print(f"Total de pares evaluados: {count}")
    print(f"LPIPS Promedio:           {avg_lpips:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description="Calcular LPIPS entre carpetas 'cond' y 'generated'.")
    parser.add_argument("dataset_path", type=str, help="Ruta base del dataset (debe contener 'cond' y 'generated')")
    
    args = parser.parse_args()
    
    # Extraer el nombre del dataset de la ruta (magia negra para limpiar barras finales)
    dataset_name = os.path.basename(os.path.normpath(args.dataset_path))
    
    # Construir las rutas a las subcarpetas
    CARPETA_REAL = os.path.join(args.dataset_path, "condImgs")
    CARPETA_SINTETICA = os.path.join(args.dataset_path, "generatedImgs")
    
    # Validar que las carpetas existan antes de empezar
    if not os.path.exists(CARPETA_REAL):
        print(f"Error: No se encontró la carpeta de imágenes reales en {CARPETA_REAL}")
        exit(1)
        
    if not os.path.exists(CARPETA_SINTETICA):
        print(f"Error: No se encontró la carpeta de imágenes sintéticas en {CARPETA_SINTETICA}")
        exit(1)
        
    calculate_average_lpips(CARPETA_REAL, CARPETA_SINTETICA, dataset_name)