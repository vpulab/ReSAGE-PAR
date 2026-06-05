import os
import pickle
import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image, UnidentifiedImageError
from local import get_pkl_rootpath
import torchvision.transforms as T
import random

class MultiModalAttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):
        assert args.dataset in ['PA100k', 'RAPv1','RAPv2','PETA','WIDER','RAPzs','PETAzs','MSP','MSPCD'], \
            f'dataset name {args.dataset} is not exist, The legal name is PA100k,RAPV1,RAPV2,PETA,WIDER'
        
        pkl_path, root_path = get_pkl_rootpath(args.dataset)
        dataset_info = pickle.load(open(pkl_path, 'rb+'))
        
        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = root_path
        self.attr_id = dataset_info.attributes
        self.attr_num = len(self.attr_id)

        print(self.attr_num )

        self.attributes = dataset_info.attr_name
        self.img_idx = dataset_info.partition[split]
        self.sentences = dataset_info.sentences
        self.max_length = dataset_info.max_length
        self.max_length = (self.max_length // 5 + 1 ) * 5
        self.limit_words = dataset_info.limit_word
        
        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]
            
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]

        # ====================================================================
        # AÑADIDO: Carga limpia de datos sintéticos (solo en entrenamiento)
        # ====================================================================
        # Usamos un flag general en args, por ejemplo args.use_synthetic
        if (split=='train' or split=='trainval') and args.use_synthetic:
            # Rutas desde args
            pseudo_csv_path = args.pseudolabels_csv
            pseudo_csv_path = os.path.expandvars(pseudo_csv_path)
            
            # Ahora el CSV de prompts se pasa explícitamente por argumento
            generated_csv_path = args.prompts_csv 
            generated_csv_path = os.path.expandvars(generated_csv_path)
            
            synthetic_img_dir = args.synthetic_img_dir
            synthetic_img_dir = os.path.expandvars(synthetic_img_dir)

            if os.path.isfile(pseudo_csv_path):
                dfPL = pd.read_csv(pseudo_csv_path, index_col=False)
                
                # --- CARGAR TEXTO (PROMPTS) CON LÓGICA DE RAIZ ---
                dict_syn_sentences = {}
                if os.path.isfile(generated_csv_path):
                    dfGen = pd.read_csv(generated_csv_path, index_col=False)
                    text_col = 'prompt' if 'prompt' in dfGen.columns else 'sentence'
                    
                    if 'genImg' in dfGen.columns and text_col in dfGen.columns:
                        # Guardamos el diccionario tal cual está en el CSV de prompts
                        dict_syn_sentences = dict(zip(dfGen['genImg'].astype(str), dfGen[text_col]))
                    else:
                        print(f"⚠️ Aviso: No se encontraron las columnas 'genImg' o '{text_col}' en {generated_csv_path}")

                # --- IDENTIFICAR COLUMNAS DE ATRIBUTOS ---
                if "img_name" in dfPL.columns:
                    img_idx = dfPL.columns.get_loc("img_name")
                    attr_cols_csv = list(dfPL.columns[img_idx + 1:])
                else:
                    raise ValueError("El CSV no contiene la columna 'img_name'")
                
                full_name_to_idx = {a: i for i, a in enumerate(self.attributes)}
                syn_img_ids = []
                syn_vectors = []
                
                for _, r in dfPL.iterrows():
                    img_filename = str(r["img_name"]) # Ej: "img-0_2.png"
                    syn_path = os.path.join(synthetic_img_dir, img_filename)
                    
                    # --- LÓGICA DE BÚSQUEDA DE PROMPT (RAIZ) ---
                    # 1. Separamos por el último "_"
                    # 2. Si el nombre es "img-0_2.png", obtenemos "img-0" y ".png"
                    if "_" in img_filename:
                        base_part = img_filename.rsplit('_', 1)[0] # "img-0"
                        ext_part = img_filename.rsplit('.', 1)[-1]  # "png"
                        root_filename = f"{base_part}.{ext_part}"   # "img-0.png"
                    else:
                        root_filename = img_filename

                    # Buscamos en el diccionario usando el root_filename
                    prompt_text = dict_syn_sentences.get(root_filename, "A person walking.")
                    self.sentences[syn_path] = prompt_text
                    
                    # --- PROCESAR VECTOR DE ATRIBUTOS ---
                    vec = [-1] * len(self.attributes)
                    for c in attr_cols_csv:
                        if c in full_name_to_idx:
                            val = r[c]
                            if pd.notna(val):
                                try:
                                    vec[full_name_to_idx[c]] = int(round(float(val)))
                                except ValueError:
                                    vec[full_name_to_idx[c]] = -1
                    
                    syn_img_ids.append(syn_path)
                    syn_vectors.append(vec)
                
                # --- FUSIÓN FINAL ---
                self.img_id = np.hstack([self.img_id, np.array(syn_img_ids)])
                self.label = np.vstack([self.label, np.array(syn_vectors, dtype=np.int32)])
                self.img_num = len(self.img_id)
                
                print(f"[SYN-PL] Fusionadas {len(syn_img_ids)} imágenes usando prompts de {generated_csv_path}")
            else:
                print(f"⚠️ Aviso: No se encontró el CSV de pseudolabels en {pseudo_csv_path}")
        # ====================================================================

        self.label_all = self.label
        
        # Reconstruir las listas de sentences con TODAS las imágenes (reales + sintéticas)
        self.all_sentence = []
        for img_name_or_path in self.img_id:
            self.all_sentence.append(self.sentences[img_name_or_path])
            
        # Refrescar random sentences
        self.random_sentences = [self.all_sentence[i] for i in random.sample(range(len(self.all_sentence)), len(self.all_sentence))]   

        #self.img_id = self.img_id[:10]
        #self.label = self.label[:10]
        #self.random_sentences = self.random_sentences[:10]

    def __getitem__(self, index):
        imgname, gt_label = self.img_id[index], self.label[index]

        sentences = self.sentences[imgname]
        
        # Resolución inteligente de la ruta: 
        # Si es absoluta (datos sintéticos), úsala tal cual. Si es relativa (datos reales), añádele el root_path.
        if isinstance(imgname, str) and os.path.isabs(imgname):
            imgpath = imgname
        else:
            imgpath = os.path.join(self.root_path, imgname)
            
        #img_pil = Image.open(imgpath).convert("RGB")
        try:
            img_pil = Image.open(imgpath).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            print(f"--- ERROR detectado en imagen {imgpath}. Reintentando o saltando... ---")
            # Opción A: Reintentar una vez tras un pequeño sueño (útil en Lustre)
            import time
            time.sleep(0.5)
            try:
                img_pil = Image.open(imgpath).convert("RGB")
            except:
                # Opción B: Si sigue fallando, cargar una imagen negra de 224x224 
                # para que el entrenamiento no se detenga por una sola foto corrupta
                #from PIL import Image
                img_pil = Image.new('RGB', (224, 224), (0, 0, 0))


        if self.transform is not None:
            img_pil = self.transform(img_pil)

        gt_label = gt_label.astype(np.float32)
        
        if self.target_transform is not None:
            gt_label = self.transform(gt_label)
        
        return img_pil, gt_label, imgname, sentences, self.random_sentences[index]

    def __len__(self):
        return len(self.img_id)

def get_transform(args):
    # (El código del get_transform se queda exactamente igual)
    height = args.height
    width = args.width
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    random_erasing = T.RandomErasing(
        p=0.5,               # Probability of applying the transform
        scale=(0.02, 0.33),  # Proportion of the erased area against input image
        ratio=(0.3, 3.3),    # Aspect ratio of the erased area
        value='random',      # Fill erased area with random values
        inplace=False        # Apply the transform out-of-place
    )
    train_transform = T.Compose([
                T.Resize((height, width), interpolation=3),
                T.RandomHorizontalFlip(0.5),
                T.Pad(10),
                T.RandomCrop((height, width)),
                T.ToTensor(),
                normalize,
                random_erasing,
            ])
    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform