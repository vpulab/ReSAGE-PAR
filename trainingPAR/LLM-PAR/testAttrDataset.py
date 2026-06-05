import os
import torch
from dataset.AttrDataset import MultiModalAttrDataset, get_transform

# 1. Creamos unos "falsos argumentos" para engañar al Dataset
class MockArgs:
    def __init__(self):
        self.dataset = 'PA100k'  # Cambia esto a PA100k o PETA si usas otro
        self.height = 256
        self.width = 128
        
        # Parámetros sintéticos que acabamos de añadir
        self.use_synthetic = True
        self.pseudolabels_csv = '$CSIC_SYN/pa100k_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4791613219522997356/pseudolabels.csv'
        self.synthetic_img_dir = '$CSIC_SYN/pa100k_lora_32_with_transform_256_192_loss_train_model_21_forlabeling_4791613219522997356/generatedImgs/'

        self.pseudolabels_csv = os.path.expandvars(self.pseudolabels_csv)
        self.synthetic_img_dir = os.path.expandvars(self.synthetic_img_dir)
    
def main():
    args = MockArgs()
    
    print("Cargando transformaciones...")
    train_transform, _ = get_transform(args)
    
    print(f"Inicializando MultiModalAttrDataset para {args.dataset}...")
    dataset = MultiModalAttrDataset(split='train', args=args, transform=train_transform)
    
    total_imgs = len(dataset)
    print(f"\n✅ Dataset cargado con éxito. Total de imágenes: {total_imgs}")
    
    if total_imgs == 0:
        print("Error: El dataset está vacío.")
        return

    # 2. Vamos a sacar una imagen REAL (suele estar al principio, ej: índice 0)
    print("\n" + "="*50)
    print("🕵️‍♂️ PRUEBA 1: IMAGEN REAL (Índice 0)")
    print("="*50)
    img_real, label_real, name_real, sent_real, _ = dataset[0]
    
    print(f"Nombre/Ruta: {name_real}")
    print(f"Tensor de Imagen: {img_real.shape} | Tipo: {img_real.dtype}")
    print(f"Tensor de Etiquetas: {label_real.shape} | Suma (1s): {label_real.sum()}")
    print(f"Texto (Sentence): {sent_real}")
    
    # 3. Vamos a sacar una imagen SINTÉTICA (como las añadimos con hstack/vstack, están al final)
    print("\n" + "="*50)
    print("🤖 PRUEBA 2: IMAGEN SINTÉTICA (Último índice)")
    print("="*50)
    
    # Cogemos la última imagen del dataset
    last_idx = total_imgs - 1
    img_syn, label_syn, name_syn, sent_syn, _ = dataset[last_idx]
    
    print(f"Nombre/Ruta Absoluta: {name_syn}")
    print(f"Tensor de Imagen: {img_syn.shape} | Tipo: {img_syn.dtype}")
    print(f"Tensor de Etiquetas: {label_syn.shape} | Suma (1s): {label_syn.sum()}")
    print(f"Tensor de Etiquetas: {label_syn} ")
    print(f"Texto (Sentence): {sent_syn}")

    print("\n🎉 ¡Si ves los dos tensores de imagen como [3, 256, 128] y las frases tienen sentido, el código funciona perfecto!")

if __name__ == '__main__':
    main()