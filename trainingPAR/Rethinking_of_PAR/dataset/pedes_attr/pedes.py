import os
import pickle
import ast

import numpy as np
import torch.utils.data as data
from PIL import Image
import pandas as pd
from tools.function import get_pkl_rootpath


# ---- legacy constants kept (unused in pseudolabel path) ----
labelsSyntheticFile   = "labels.txt"
promptsSyntheticFile  = "promptsGenerated.txt"
pandasFile            = "syntheticDataset_labels_-1.csv"
pandasFile0           = "syntheticDataset_labels_0.csv"
syntheticNameFile     = "aug"       # legacy marker for old flows
syntheticExtensionFile = ".png"
# ------------------------------------------------------------

# RAPv2 attributes list (original code)
listAttributesRAPv2 = ['hs-BaldHead', 'hs-LongHair', 'hs-BlackHair', 'hs-Hat', 'hs-Glasses','ub-Shirt','ub-Sweater','ub-Vest','ub-TShirt','ub-Cotton','ub-Jacket','ub-SuitUp','ub-Tight','ub-ShortSleeve','ub-Others','lb-LongTrousers','lb-Skirt','lb-ShortSkirt','lb-Dress','lb-Jeans','lb-TightTrousers','shoes-Leather', 'shoes-Sports', 'shoes-Boots', 'shoes-Cloth', 'shoes-Casual', 'shoes-Other','attachment-Backpack','attachment-ShoulderBag','attachment-HandBag','attachment-Box','attachment-PlasticBag','attachment-PaperBag','attachment-HandTrunk','attachment-Other','AgeLess16', 'Age17-30', 'Age31-45', 'Age46-60','Femal','BodyFat','BodyNormal','BodyThin','Customer','Employee','action-Calling','action-Talking','action-Gathering','action-Holding','action-Pushing','action-Pulling','action-CarryingByArm','action-CarryingByHand','action-Other']
# from preprocess rethinking
listAttributesRAPv2.remove('Age46-60')
listAttributes = listAttributesRAPv2

from dataset.pedes_attr.MALS import MALSDataset

class PedesAttr(data.Dataset):
    """
    Dataset that can optionally append synthetic images from a *pseudolabels CSV*.

    CSV format (dynamic attributes):
        img_name, score, prob, decision, <attr_1>, <attr_2>, ..., <attr_M>

    - The columns after 'decision' are treated as the attribute vector.
    - Those columns' names must match dataset attribute names. Missing attrs are filled with -1.
    - We keep the same index order as the original `attr_labels` by building a full-length vector
      in the dataset's attribute order, then subsetting according to cfg.DATASET.LABEL (all/eval/color).
    """

    def __init__(self, cfg, split, transform=None, target_transform=None, idx=None, train=None):

        assert cfg.DATASET.NAME in ['PETA', 'PA100k', 'RAP', 'RAP2'], \
            f'dataset name {cfg.DATASET.NAME} is not exist'

        data_path = get_pkl_rootpath(cfg.DATASET.NAME, cfg.DATASET.ZERO_SHOT)
        print("which pickle", data_path)
        dataset_info = pickle.load(open(data_path, 'rb+'))

        # Keep FULL attribute list for alignment
        self.attr_id_full = list(dataset_info.attr_name)
        full_name_to_idx = {a: i for i, a in enumerate(self.attr_id_full)}

        img_id = dataset_info.image_name
        attr_label = dataset_info.label
        attr_label[attr_label == 2] = 0

        # Active (possibly subset) attribute list
        self.attr_id = list(dataset_info.attr_name)  # may be overridden by LABEL selection
        self.attr_num = len(self.attr_id)

        # Indices for eval/color selection
        self.eval_attr_idx = None
        self.color_attr_idx = None

        if 'label_idx' not in dataset_info.keys():
            print(' this is for zero shot split')
            assert cfg.DATASET.ZERO_SHOT
            self.eval_attr_num = self.attr_num
            self.active_attr_indices = list(range(len(self.attr_id_full)))  # all
        else:
            self.eval_attr_idx = dataset_info.label_idx.eval
            
            self.eval_attr_num = len(self.eval_attr_idx)

            assert cfg.DATASET.LABEL in ['all', 'eval', 'color'], f'key word {cfg.DATASET.LABEL} error'
            if cfg.DATASET.LABEL == 'eval':
                attr_label = attr_label[:, self.eval_attr_idx]
                self.attr_id  = [self.attr_id[i] for i in self.eval_attr_idx]
                self.attr_num = len(self.attr_id)
                self.active_attr_indices = list(self.eval_attr_idx)
            elif cfg.DATASET.LABEL == 'color':
                self.color_attr_idx = dataset_info.label_idx.color
                active = self.eval_attr_idx + self.color_attr_idx
                attr_label = attr_label[:, active]
                self.attr_id  = [self.attr_id[i] for i in active]
                self.attr_num = len(self.attr_id)
                self.active_attr_indices = list(active)
            else:
                # 'all'
                self.active_attr_indices = list(range(len(self.attr_id_full)))

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = cfg.DATASET.NAME
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = cfg.DATASET.ROOT

        if self.target_transform:
            self.attr_num = len(self.target_transform)
            print(f'{split} target_label: {self.target_transform}')
        else:
            self.attr_num = len(self.attr_id)
            print(f'{split} target_label: all')

        self.img_idx = dataset_info.partition[split]
        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]  # default partition 0
        if idx is not None:
            self.img_idx = idx

        # ---------- base (real) split ----------
        if train==False and cfg.SYNTHETICTEST.USE == True:
            if os.path.isdir(cfg.SYNTHETICTEST.PATH):
                self.synthetic_path = cfg.SYNTHETICTEST.PATH
                dfWithSynImages = pd.read_csv(cfg.SYNTHETICTEST.PATH + pandasFile, index_col=False)
                dfConfigured = dfWithSynImages.sample(frac=cfg.SYNTHETICTEST.PERCENTAGE)
                listSyntheticIDs = list(dfConfigured['imgName'])
                dfsyntheticLabels = dfConfigured.drop(['imgName'], axis=1)
                listSyntheticLabels = dfsyntheticLabels.values.tolist()

                syntheticIDsNP   = np.array(listSyntheticIDs)
                syntheticLabelsNP = np.array(listSyntheticLabels)

                self.img_id = syntheticIDsNP
                self.label  = syntheticLabelsNP
                self.img_num = self.img_id.shape[0]
                print(self.img_num)
            else:
                print("Error loading synthetic data at test")
        else:
            self.img_num = self.img_idx.shape[0]
            self.img_id = [img_id[i] for i in self.img_idx]
            self.label  = attr_label[self.img_idx]

        print(self.img_num)

        if cfg.SYNTHETIC.USE_MALS == True and train==True:
 
        
            # 1. Al inicializar, le pasas las imágenes reales. 
            # Automáticamente se genera un "pool_ids" que es un numpy array con el tamaño x2 aleatorio.
            mals_dataset = MALSDataset(
                mals_base_path=cfg.SYNTHETIC.MALS_PATH, 
                num_real_images=self.img_num,  # PA100k, PETA, etc.
                target_dataset=cfg.DATASET.NAME,
                active_attr_indices=self.active_attr_indices
            )

            # 2. Cuando pides los datos, le pasas el % (C.SYNTHETIC.PERCENTAGE)
            # Esto cogerá ese porcentaje de la reserva x2.
            syntheticIDsNP, syntheticLabelsNP = mals_dataset.get_balanced_data(percentage=cfg.SYNTHETIC.PERCENTAGE)

            # 3. Apilas los datos al original
            if len(syntheticIDsNP) > 0:
                self.img_id = np.hstack([self.img_id, syntheticIDsNP])
                self.label  = np.vstack([self.label, syntheticLabelsNP])
                self.img_num = self.img_id.shape[0]

            print(f"Dataset combinado con éxito. Nuevo total: {self.img_num}")
        
        # ==================== Add synthetic-for-training from pseudolabels CSV ====================
        # CSV columns: img_name, score, prob, decision, <attr_1>, <attr_2>, ..., <attr_M>
        if cfg.SYNTHETIC.USE == True and train==True:
            use_pseudo = hasattr(cfg.SYNTHETIC, "PSEUDOLABELS_CSV") and \
                         (cfg.SYNTHETIC.PSEUDOLABELS_CSV is not None) and \
                         os.path.isfile(cfg.SYNTHETIC.PSEUDOLABELS_CSV)

            if not os.path.isdir(cfg.SYNTHETIC.PATH):
                print("Error loading synthetic data at train: invalid SYNTHETIC.PATH")
            else:
                self.synthetic_path = cfg.SYNTHETIC.PATH

                if use_pseudo:
                    dfPL = pd.read_csv(cfg.SYNTHETIC.PSEUDOLABELS_CSV, index_col=False)

                    #for req ["score", "prob", "decision"] in dfPL.columns
                    saltarDec=False
                    # required fixed columns
                    for req in ["img_name", "score", "prob", "decision"]:
                        if req not in dfPL.columns:
                            saltarDec=True

                            #raise ValueError(f"PSEUDOLABELS_CSV must contain column '{req}'.")
                    if saltarDec == True:
                        
                        # dynamic attribute columns = all columns after 'decision'
                        dec_idx = dfPL.columns.get_loc("img_name")
                        attr_cols_csv = list(dfPL.columns[dec_idx + 1:])
                    else:
                        # dynamic attribute columns = all columns after 'decision'
                        dec_idx = dfPL.columns.get_loc("decision")
                        attr_cols_csv = list(dfPL.columns[dec_idx + 1:])
                        if not attr_cols_csv:
                            raise ValueError("No attribute columns found after 'decision' in the pseudolabels CSV.")

                    # Sample fraction if desired
                    frac = getattr(cfg.SYNTHETIC, "PERCENTAGE", 1.0)
                    if 0 < frac < 1.0:
                        dfPL = dfPL.sample(frac=frac, random_state=getattr(cfg.SYNTHETIC, "SEED", None))

                    # Optional subdir like "generatedImgs"
                    syn_subdir = getattr(cfg.SYNTHETIC, "IMG_SUBDIR", "")

                    def _resolve_syn_path(v: str) -> str:
                        v = str(v)
                        if os.path.isabs(v):
                            return v
                        # If relative name, resolve under SYNTHETIC.PATH (and subdir if configured and CSV only has basename)
                        if syn_subdir and not os.path.dirname(v):
                            return os.path.join(self.synthetic_path, syn_subdir, v)
                        return os.path.join(self.synthetic_path, v)

                    syn_img_ids   = []
                    syn_vectors   = []

                    for _, r in dfPL.iterrows():
                        # 1) resolve image path (no 'aug' prefix)
                        syn_path = _resolve_syn_path(r["img_name"])

                        # 2) build FULL-length vector in dataset order (fill -1)
                        vec_full = [-1] * len(self.attr_id_full)
                        for c in attr_cols_csv:
                            if c not in full_name_to_idx:
                                # Unknown attribute name -> ignore silently (or print once if you prefer)
                                continue
                            try:
                                v = r[c]
                                v = int(round(float(v))) if pd.notna(v) else -1
                            except Exception:
                                v = -1
                            vec_full[full_name_to_idx[c]] = v

                        # 3) subset to the current LABEL selection (all/eval/color)
                        vec_active = [vec_full[i] for i in self.active_attr_indices]

                        syn_img_ids.append(syn_path)
                        syn_vectors.append(vec_active)

                    syntheticIDsNP    = np.array(syn_img_ids)
                    syntheticLabelsNP = np.array(syn_vectors, dtype=np.int32)
                    print("lenght syn vector "+str(len(syn_vectors[0])))
                    # merge with real data
                    self.img_id = np.hstack([self.img_id, syntheticIDsNP])
                    self.label  = np.vstack([self.label, syntheticLabelsNP])
                    self.img_num = self.img_id.shape[0]
                    print(f"[SYN-PL] Merged synthetic (pseudolabels CSV): +{len(syn_img_ids)} => total {self.img_num}")

                else:
                    # ---------------- Legacy (non-pseudolabel) CSV flow: unchanged ----------------
                    if cfg.SYNTHETIC.PANDASFILE is None:
                        if cfg.SYNTHETIC.VECTOR == -1:
                            print("---------------->Menos 1!")
                            dfWithSynImages = pd.read_csv(cfg.SYNTHETIC.PATH + pandasFile, index_col=False)
                        elif cfg.SYNTHETIC.VECTOR == 0:
                            dfWithSynImages = pd.read_csv(cfg.SYNTHETIC.PATH + pandasFile0, index_col=False)
                    else:  # MALS
                        dfWithSynImages = pd.read_csv(cfg.SYNTHETIC.PANDASFILE, index_col=False)

                    dfConfigured = dfWithSynImages.sample(frac=cfg.SYNTHETIC.PERCENTAGE)
                    listSyntheticIDs = list(dfConfigured['imgName'])
                    if cfg.SYNTHETIC.PANDASFILE is not None:
                        listSyntheticIDs = [syntheticNameFile + imageName for imageName in listSyntheticIDs]
                    print(listSyntheticIDs)
                    dfsyntheticLabels = dfConfigured.drop(['imgName'], axis=1)
                    listSyntheticLabels = dfsyntheticLabels.values.tolist()

                    syntheticIDsNP    = np.array(listSyntheticIDs)
                    syntheticLabelsNP = np.array(listSyntheticLabels)

                    self.img_id = np.hstack([self.img_id, syntheticIDsNP])
                    self.label  = np.vstack([self.label, syntheticLabelsNP])
                    self.img_num = self.img_id.shape[0]
                    print(self.img_num)
        # ==================== END pseudolabel-CSV block ====================

        # Optional class weights (unchanged logic)
        if cfg.DATASET.NAME == "RAP2" and cfg.DATASET.ZERO_SHOT == True and cfg.LOSS.TYPE=='weightedbceloss':
            listSamplesForAttribute = []
            totalSamples = len(self.label)
            for attribute in listAttributes:
                idxToAttribute = listAttributes.index(attribute)
                labelsWithAttribute = [label for label in self.label if label[idxToAttribute] == 1]
                samplesAttribute = len(labelsWithAttribute)
                listSamplesForAttribute.append(samplesAttribute)

            weightsSamples = [ (round((totalSamples / (x * len(listAttributes))) , 2) ) for x in listSamplesForAttribute]
            self.pos_weigth = weightsSamples
            print(weightsSamples)

        self.cfg = cfg

    def __getitem__(self, index):
        imgname, gt_label = self.img_id[index], self.label[index]

        if self.cfg.SYNTHETIC.USE_MALS and "MALS" in str(imgname):
            imgpath = os.path.join(self.cfg.SYNTHETIC.MALS_PATH, imgname)
            #print(f"Loading image from MALS path: {imgpath}")
        # Pseudolabeled synthetic: absolute/resolved path stored directly
        elif isinstance(imgname, str) and os.path.isabs(imgname):
            imgpath = imgname
        elif syntheticNameFile in str(imgname):
            # Legacy synthetic flow (kept for backward compatibility)
            if self.cfg.SYNTHETIC.PANDASFILE is not None:
                imgpath = os.path.join(self.synthetic_path, str(imgname).replace(syntheticNameFile, ""))
            else:
                syn_subdir = getattr(self.cfg.SYNTHETIC, "IMG_SUBDIR", "")
                if syn_subdir:
                    imgpath = os.path.join(self.synthetic_path, syn_subdir, str(imgname).replace(syntheticNameFile, ""))
                else:
                    imgpath = os.path.join(self.synthetic_path, imgname)
        else:
            # Real image
            imgpath = os.path.join(self.root_path, imgname)

        img = Image.open(imgpath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        gt_label = np.asarray(gt_label).astype(np.float32)
        if self.target_transform:
            gt_label = gt_label[self.target_transform]

        return img, gt_label, imgname,

    def __len__(self):
        return len(self.img_id)
