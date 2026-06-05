import os
import json
import numpy as np

class MALSDataset:
    def _align_label_PA100k(self, raw_label):
            """Traduce el vector MALS al formato PA100k (26 enteros)."""
            pa100k_label = [0] * 26
            if raw_label[0] != -1: pa100k_label[0] = 1 if raw_label[0] == 0 else 0
            if raw_label[1] != -1: 
                if raw_label[1] == 3: pa100k_label[1] = 1
                elif raw_label[1] == 2: pa100k_label[2] = 1
                elif raw_label[1] in [0, 1]: pa100k_label[3] = 1
            if raw_label[3] != -1: pa100k_label[7] = 1 if raw_label[3] == 0 else 0 
            if raw_label[5] != -1: pa100k_label[9] = 1 if raw_label[5] == 0 else 0  
            if raw_label[6] != -1: pa100k_label[10] = 1 if raw_label[6] == 0 else 0 
            if raw_label[4] != -1: pa100k_label[11] = 1 if raw_label[4] == 0 else 0 
            if raw_label[7] != -1:
                if raw_label[7] == 1: pa100k_label[13] = 1
                elif raw_label[7] == 0: pa100k_label[14] = 1
            len_lower, type_lower = raw_label[8], raw_label[9]
            if type_lower != -1:
                if type_lower == 0: pa100k_label[24] = 1
                elif type_lower == 1:
                    if len_lower == 1: pa100k_label[23] = 1
                    elif len_lower == 0: pa100k_label[22] = 1
            return pa100k_label

    def _align_label_RAP(self, raw_label):
        """Traduce el vector MALS (27 elementos) al formato RAPv2 (119 enteros)."""
        rap_label = [0] * 119
        
        # --- 1. GÉNERO ---
        # RAP solo tiene 'Femal' (índice 39). Si es hombre, se queda en 0.
        if raw_label[0] != -1:
            if raw_label[0] == 0: rap_label[39] = 1 # Female
                
        # --- 2. EDAD ---
        # 35:'AgeLess16', 36:'Age17-30', 37:'Age31-45', 38:'Age46-60', 96:'AgeBiger60'
        if raw_label[1] != -1:
            if raw_label[1] == 0: rap_label[35] = 1    # young
            elif raw_label[1] == 1: rap_label[36] = 1  # teen
            elif raw_label[1] == 2: rap_label[37] = 1  # adult
            elif raw_label[1] == 3: rap_label[38] = 1  # old (Lo mapeamos a 46-60)
                
        # --- 3. PELO ---
        # RAP solo tiene 'hs-LongHair' (índice 1). Si es corto, se queda en 0.
        if raw_label[2] != -1:
            if raw_label[2] == 1: rap_label[1] = 1 # LongHair
                
        # --- 4. ACCESORIOS (Recuerda: en MALS 0 es SÍ) ---
        if raw_label[3] != -1 and raw_label[3] == 0: rap_label[3] = 1   # Hat
        if raw_label[4] != -1 and raw_label[4] == 0: rap_label[27] = 1  # Backpack
        if raw_label[5] != -1 and raw_label[5] == 0: rap_label[29] = 1  # HandBag
        if raw_label[6] != -1 and raw_label[6] == 0: rap_label[28] = 1  # ShoulderBag
            
        # --- 5. ROPA SUPERIOR ---
        # RAP solo tiene 'ub-ShortSleeve' (índice 13)
        if raw_label[7] != -1:
            if raw_label[7] == 1: rap_label[13] = 1 # ShortSleeve
                
        # --- 6. ROPA INFERIOR ---
        len_lower = raw_label[8]
        type_lower = raw_label[9]
        if type_lower != -1 and len_lower != -1:
            if type_lower == 0: # Skirt / Dress
                rap_label[16] = 1 # General 'lb-Skirt'
                rap_label[18] = 1 # General 'lb-Dress'
                if len_lower == 0: rap_label[103] = 1  # 'lb-LongSkirt'
                elif len_lower == 1: rap_label[17] = 1 # 'lb-ShortSkirt'
            elif type_lower == 1: # Pants
                if len_lower == 0: rap_label[15] = 1   # 'lb-LongTrousers'
                elif len_lower == 1: rap_label[102] = 1 # 'lb-Shorts'
                    
        # --- 7. COLORES ROPA SUPERIOR ---
        # Mapeo a los índices exactos de RAP (fíjate que el rojo en tu lista tiene un typo 'up-ColorRed' en el 57)
        upper_colors_map = {
            10: 54, # Black
            11: 55, # White
            12: 57, # Red
            13: 63, # Purple
            14: 61, # Yellow
            15: 59, # Blue
            16: 58, # Green
            17: 56  # Gray
        }
        for mals_idx, rap_idx in upper_colors_map.items():
            if raw_label[mals_idx] != -1 and raw_label[mals_idx] == 0: # 0 es SÍ
                rap_label[rap_idx] = 1
                
        # --- 8. COLORES ROPA INFERIOR ---
        lower_colors_map = {
            18: 68, # Black
            19: 69, # White
            20: 77, # Purple
            21: 75, # Yellow
            22: 73, # Blue
            23: 72, # Green
            24: 78, # Pink
            25: 70, # Gray
            26: 76  # Brown
        }
        for mals_idx, rap_idx in lower_colors_map.items():
            if raw_label[mals_idx] != -1 and raw_label[mals_idx] == 0: # 0 es SÍ
                rap_label[rap_idx] = 1
                
        return rap_label


    def _align_label_PETA(self, raw_label):
        """Traduce el vector MALS (27 elementos) al formato PETA (105 enteros)."""
        peta_label = [0] * 105
        
        # --- 1. GÉNERO ---
        # MALS [0]: 0=Female, 1=Male
        # PETA: 87='personalFemale', 16='personalMale'
        if raw_label[0] != -1:
            if raw_label[0] == 0: peta_label[87] = 1
            elif raw_label[0] == 1: peta_label[16] = 1
                
        # --- 2. EDAD ---
        # MALS [1]: 0=young, 1=teenager, 2=adult, 3=old
        # PETA: 80='personalLess15', 0='personalLess30', 1='personalLess45', 3='personalLarger60'
        if raw_label[1] != -1:
            if raw_label[1] == 0: peta_label[80] = 1    # Young -> <15
            elif raw_label[1] == 1: peta_label[0] = 1   # Teenager -> <30
            elif raw_label[1] == 2: peta_label[1] = 1   # Adult -> <45
            elif raw_label[1] == 3: peta_label[3] = 1   # Old -> >60
                
        # --- 3. PELO (¡Nuevo! PETA sí lo tiene) ---
        # MALS [2]: 0=short, 1=long
        # PETA: 98='hairShort', 15='hairLong'
        if raw_label[2] != -1:
            if raw_label[2] == 0: peta_label[98] = 1
            elif raw_label[2] == 1: peta_label[15] = 1
                
        # --- 4. ACCESORIOS (Recuerda: en MALS 0 es SÍ) ---
        if raw_label[3] != -1 and raw_label[3] == 0: peta_label[10] = 1 # Hat -> 'accessoryHat'
        if raw_label[4] != -1 and raw_label[4] == 0: peta_label[4] = 1  # Backpack -> 'carryingBackpack'
        if raw_label[5] != -1 and raw_label[5] == 0: peta_label[5] = 1  # Handbag -> 'carryingOther'
        if raw_label[6] != -1 and raw_label[6] == 0: peta_label[17] = 1 # Bag -> 'carryingMessengerBag'
            
        # --- 5. ROPA SUPERIOR ---
        # MALS [7]: sleeve (0=long, 1=short)
        # PETA: 93='upperBodyLongSleeve', 26='upperBodyShortSleeve'
        if raw_label[7] != -1:
            if raw_label[7] == 0: peta_label[93] = 1
            elif raw_label[7] == 1: peta_label[26] = 1
                
        # --- 6. ROPA INFERIOR ---
        # MALS [8]: length, MALS [9]: type
        len_lower = raw_label[8]
        type_lower = raw_label[9]
        if type_lower != -1 and len_lower != -1:
            if type_lower == 0: # Dress / Skirt
                if len_lower == 0: peta_label[92] = 1   # 'lowerBodyLongSkirt'
                elif len_lower == 1: peta_label[27] = 1 # 'lowerBodyShortSkirt'
            elif type_lower == 1: # Pants
                if len_lower == 0: peta_label[31] = 1   # 'lowerBodyTrousers'
                elif len_lower == 1: peta_label[25] = 1 # 'lowerBodyShorts'
                    
        # --- 7. COLORES ROPA SUPERIOR ---
        # Mapeamos los índices de MALS a los índices exactos de PETA
        # 35:Black, 44:White, 43:Red, 42:Purple, 45:Yellow, 36:Blue, 38:Green, 39:Grey
        upper_colors_map = {
            10: 35, 11: 44, 12: 43, 13: 42, 
            14: 45, 15: 36, 16: 38, 17: 39
        }
        for mals_idx, peta_idx in upper_colors_map.items():
            if raw_label[mals_idx] != -1 and raw_label[mals_idx] == 0: # 0 es SÍ
                peta_label[peta_idx] = 1
                
        # --- 8. COLORES ROPA INFERIOR ---
        # 46:Black, 55:White, 53:Purple, 56:Yellow, 47:Blue, 49:Green, 52:Pink, 50:Grey, 48:Brown
        lower_colors_map = {
            18: 46, 19: 55, 20: 53, 21: 56, 
            22: 47, 23: 49, 24: 52, 25: 50, 26: 48
        }
        for mals_idx, peta_idx in lower_colors_map.items():
            if raw_label[mals_idx] != -1 and raw_label[mals_idx] == 0: # 0 es SÍ
                peta_label[peta_idx] = 1
                
        return peta_label

    def _align_label_RAPv1(self, raw_label):
        """Traduce el vector MALS (27 elementos) al formato RAPv1 (92 enteros)."""
        rap1_label = [0] * 92
        
        # --- 1. GÉNERO ---
        # 0: 'Female'
        if raw_label[0] != -1:
            if raw_label[0] == 0: rap1_label[0] = 1 
                
        # --- 2. EDAD ---
        # 1: 'AgeLess16', 2: 'Age17-30', 3: 'Age31-45'
        # Nota: La lista que me has pasado de RAPv1 no tiene edad > 45, así que si MALS dice 'old' (3), lo ignoramos para no falsear datos.
        if raw_label[1] != -1:
            if raw_label[1] == 0: rap1_label[1] = 1    # young -> <16
            elif raw_label[1] == 1: rap1_label[2] = 1  # teen -> 17-30
            elif raw_label[1] == 2: rap1_label[3] = 1  # adult -> 31-45
                
        # --- 3. PELO ---
        # 10: 'hs-LongHair'
        if raw_label[2] != -1:
            if raw_label[2] == 1: rap1_label[10] = 1 
                
        # --- 4. ACCESORIOS (Recuerda: en MALS 0 es SÍ) ---
        if raw_label[3] != -1 and raw_label[3] == 0: rap1_label[12] = 1 # 'hs-Hat'
        if raw_label[4] != -1 and raw_label[4] == 0: rap1_label[35] = 1 # 'attach-Backpack'
        if raw_label[5] != -1 and raw_label[5] == 0: rap1_label[37] = 1 # 'attach-HandBag'
        if raw_label[6] != -1 and raw_label[6] == 0: rap1_label[36] = 1 # 'attach-SingleShoulderBag'
            
        # --- 5. ROPA SUPERIOR ---
        # 23: 'ub-ShortSleeve'
        if raw_label[7] != -1:
            if raw_label[7] == 1: rap1_label[23] = 1 
                
        # --- 6. ROPA INFERIOR ---
        len_lower = raw_label[8]
        type_lower = raw_label[9]
        if type_lower != -1 and len_lower != -1:
            if type_lower == 0: # Skirt / Dress
                rap1_label[25] = 1 # General 'lb-Skirt'
                rap1_label[27] = 1 # General 'lb-Dress'
                if len_lower == 1: rap1_label[26] = 1 # 'lb-ShortSkirt'
            elif type_lower == 1: # Pants
                if len_lower == 0: rap1_label[24] = 1 # 'lb-LongTrousers'
                # RAPv1 en tu lista no tiene 'lb-Shorts', así que si el pantalón es corto, no marcamos nada.
                    
        # --- 7. COLORES ROPA SUPERIOR ---
        # 63:'up-Black', 64:'up-White', 65:'up-Gray', 66:'up-Red', 67:'up-Green', 68:'up-Blue', 69:'up-Yellow', 71:'up-Purple'
        upper_colors_map = {
            10: 63, # Black
            11: 64, # White
            12: 66, # Red
            13: 71, # Purple
            14: 69, # Yellow
            15: 68, # Blue
            16: 67, # Green
            17: 65  # Gray
        }
        for mals_idx, rap_idx in upper_colors_map.items():
            if raw_label[mals_idx] != -1 and raw_label[mals_idx] == 0: # 0 es SÍ
                rap1_label[rap_idx] = 1
                
        # --- 8. COLORES ROPA INFERIOR ---
        # 75:'low-Black', 76:'low-White', 77:'low-Gray', 79:'low-Green', 80:'low-Blue', 81:'low-Yellow'
        # Nota: RAPv1 no tiene Purple, Pink ni Brown en la ropa inferior, así que esos colores de MALS no se mapean a nada.
        lower_colors_map = {
            18: 75, # Black
            19: 76, # White
            21: 81, # Yellow
            22: 80, # Blue
            23: 79, # Green
            25: 77  # Gray
        }
        for mals_idx, rap_idx in lower_colors_map.items():
            if raw_label[mals_idx] != -1 and raw_label[mals_idx] == 0: # 0 es SÍ
                rap1_label[rap_idx] = 1
                
        return rap1_label

    def __init__(self, mals_base_path, num_real_images, target_dataset="PA100k", active_attr_indices=None):
        """
        mals_base_path: Ruta a la carpeta principal de MALS.
        num_real_images: Cantidad de imágenes del dataset original.
        target_dataset: Nombre del dataset (PA100k, PETA, RAP...).
        active_attr_indices: Lista o array de NumPy con los índices de los atributos que realmente se usan.
        """
        self.mals_base_path = mals_base_path
        self.gene_attrs_path = os.path.join(mals_base_path, 'gene_attrs')
        
        self.target_dataset = target_dataset
        self.num_real_images = num_real_images
        self.active_attr_indices = active_attr_indices  # <--- GUARDAMOS LOS ÍNDICES
        
        # 1. Cargar el 100% de la base de datos MALS
        all_ids, all_labels = self._load_all_synthetic_data()
        
        # 2. Crear el Pool (x2) adaptado a este dataset
        self.pool_ids, self.pool_labels = self._create_adapted_pool(all_ids, all_labels)

    def _load_all_synthetic_data(self):
        """Lee TODOS los JSONs de MALS y aplica el recorte de atributos."""
        ids_list, labels_list = [], []
        
        if not os.path.exists(self.gene_attrs_path):
            raise FileNotFoundError(f"Ruta no encontrada: {self.gene_attrs_path}")
            
        json_files = [f for f in os.listdir(self.gene_attrs_path) if f.endswith('_attrs.json')]
        
        for json_file in json_files:
            split_name = json_file.replace('_attrs.json', '')
            if split_name.startswith('g_'):
                split_name = split_name[2:]
            
            split_folder_path = os.path.join(self.mals_base_path, split_name)
            if not os.path.isdir(split_folder_path):
                continue
            
            json_path = os.path.join(self.gene_attrs_path, json_file)
            
            with open(json_path, 'r') as f:
                split_data = json.load(f)
                
            for item in split_data:
                img_name_raw = item.get("image")
                raw_label = item.get("label")
                
                if img_name_raw and raw_label is not None:
                    img_name_clean = os.path.basename(img_name_raw)
                    final_path = os.path.join(self.mals_base_path, split_name, img_name_clean)
                    
                    ids_list.append(final_path)
                    
                    if self.target_dataset == "PA100k":
                        aligned_label = self._align_label_PA100k(raw_label)
                    elif self.target_dataset in ["PETA", "PETAzs"]:
                        aligned_label = self._align_label_PETA(raw_label)
                    elif self.target_dataset in ["RAP2"]: # <--- AÑADIDO AQUI
                        aligned_label = self._align_label_RAP(raw_label)
                    elif self.target_dataset in ["RAP"]: # <- Si le pasas "RAP" a secas o "RAPv1"
                        aligned_label = self._align_label_RAPv1(raw_label)
                    else:
                        raise ValueError(f"Dataset '{self.target_dataset}' no implementado.")
                        
                    labels_list.append(aligned_label)

        # 1. Convertimos a NumPy (Matriz completa)
        ids_np = np.array(ids_list)
        labels_np = np.array(labels_list, dtype=np.int32)

        # ========================================================
        # 2. POST-PROCESADO: RECORTAR SOLO LOS ATRIBUTOS ACTIVOS
        # ========================================================
        if self.active_attr_indices is not None and len(labels_np) > 0:
            # Seleccionamos solo las columnas indicadas en la lista
            labels_np = labels_np[:, self.active_attr_indices]

        return ids_np, labels_np

        

    def _create_adapted_pool(self, all_ids, all_labels):
        """Paso 2: Genera un subset aleatorio de tamaño (num_real_images * 2)"""
        target_pool_size = self.num_real_images * 2
        total_mals = len(all_ids)
        
        # Limitamos por si PA100k pide un x2 más grande que el propio MALS entero
        pool_size = min(target_pool_size, total_mals)
        
        print(f"[MALS Setup] Dataset Real: {self.num_real_images} img | "
              f"Creando Pool Sintético (x2): {pool_size} img")
        
        if pool_size > 0:
            # Hacemos random del conjunto grande
            indices = np.random.choice(total_mals, pool_size, replace=False)
            return all_ids[indices], all_labels[indices]
        return np.array([]), np.empty((0, all_labels.shape[1]), dtype=np.int32)

    def get_balanced_data(self, percentage):
        """
        Paso 3: Devuelve un porcentaje exacto SOBRE EL POOL (x2).
        """
        if percentage <= 0:
            return np.array([]), np.empty((0, self.pool_labels.shape[1]), dtype=np.int32)

        # Ajustamos el porcentaje por si lo pasas como 50 en lugar de 0.5
        p = percentage
        if p > 1.0 and isinstance(p, (int, float)):
             p = p / 100.0

        # Calculamos basándonos en el tamaño del Pool actual
        pool_size = len(self.pool_ids)
        final_amount = int(pool_size * p)
        
        print(f"[MALS Balance] Aplicando {p*100}% sobre el Pool de {pool_size} -> "
              f"Extrayendo {final_amount} imágenes finales.")

        if final_amount > 0:
            # Hacemos random del pool para extraer ese porcentaje
            indices = np.random.choice(pool_size, final_amount, replace=False)
            return self.pool_ids[indices], self.pool_labels[indices]
            
        return np.array([]), np.empty((0, self.pool_labels.shape[1]), dtype=np.int32)