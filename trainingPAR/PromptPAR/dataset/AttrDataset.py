import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2

from tools.function import get_pkl_rootpath
import torchvision.transforms as T
class MultiModalAttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):

        assert args.dataset in ['PA100k', 'RAPV1','RAPV2','PETA','WIDER','RAPzs','PETAzs','UPAR','YCJC',], \
            f'dataset name {args.dataset} is not exist,The legal name is PA100k,RAPV1,RAPV2,PETA,WIDER'
            
        dataset_dir='/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/' #Set this to the directory where the dataset is located
        if args.dataset!='RAPzs' and args.dataset!='PETAzs':
            

            if args.dataset=='PETA':
                self.root_path = dataset_dir+args.dataset+"/images/"
                dataset_info = pickle.load(open(dataset_dir+args.dataset+"/pad.pkl", 'rb+'))
            if args.dataset=='RAPV1':
                self.root_path = dataset_dir+"RAP/RAP_dataset/"
                dataset_info = pickle.load(open(dataset_dir+"RAP/expand_pad.pkl", 'rb+'))
            if args.dataset=='RAPV2':
                self.root_path = dataset_dir+"RAP2/RAP_dataset/"
                dataset_info = pickle.load(open(dataset_dir+"RAP2/pad.pkl", 'rb+'))
            if args.dataset=='PA100k':
                self.root_path = dataset_dir+args.dataset+"/data/"
                dataset_info = pickle.load(open(dataset_dir+args.dataset+"/pad.pkl", 'rb+'))


        elif args.dataset=='RAPzs':
            dataset_info = pickle.load(open(dataset_dir+"RAP2/dataset_zs_pad.pkl", 'rb+'))
            self.root_path = dataset_dir+"RAP2/RAP_dataset/"

        elif args.dataset=='PETAzs':
            dataset_info = pickle.load(open(dataset_dir+"PETA/dataset_zs_pad.pkl", 'rb+'))
            self.root_path = dataset_dir+"PETA/images/"

        #dataset_info = pickle.load(open('/data/jinjiandong/datasets/yichangjiance/pad.pkl', 'rb+'))

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform
        
        self.attr_id = dataset_info.attributes
        self.attr_num = len(self.attr_id)
        self.attributes=dataset_info.attributes
        
        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]


        self.attr_id_full = list(dataset_info.attr_name)
        full_name_to_idx = {a: i for i, a in enumerate(self.attr_id_full)}

        if 'zs' in args.dataset or args.dataset == 'RAPV1' or args.dataset == 'PA100k':
            # en este pkl tienen capados ya los atributos
            print(' this is for zero shot split or RAPv1/PA100k, use all attributes for evaluation')
            
            self.eval_attr_num = self.attr_num
            self.active_attr_indices = list(range(len(self.attr_id_full)))  # all
        else:
            self.eval_attr_idx = dataset_info.label_idx.eval
            print(attr_label.shape)
            print(len(self.eval_attr_idx))
            idx = self.eval_attr_idx
            print("min/max:", min(idx), max(idx))
            print("tail:", sorted(idx)[-10:])
            print("shape:", attr_label.shape)
            self.eval_attr_num = len(self.eval_attr_idx)

            attr_label = attr_label[:, self.eval_attr_idx]
            self.attr_id  = [self.attr_id[i] for i in self.eval_attr_idx]
            self.attr_num = len(self.attr_id)
            self.active_attr_indices = list(self.eval_attr_idx)
            
            

        self.label = attr_label[self.img_idx]
        
                # ==================== Add synthetic-for-training from pseudolabels CSV ====================
        # CSV columns: img_name, score, prob, decision, <attr_1>, <attr_2>, ..., <attr_M>
 
        if args.syn_use == True and (split=='train' or split=='trainval'):
            self.attr_id_full = list(dataset_info.attr_name)
            full_name_to_idx = {a: i for i, a in enumerate(self.attr_id_full)}
            self.synthetic_path = args.synthetic_path
            print(f"[SYN-PL] Loading synthetic pseudolabels from CSV: {args.syn_pseudolabels_csv}")
            import pandas as pd
            dfPL = pd.read_csv(self.synthetic_path+args.syn_pseudolabels_csv, index_col=False)

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

            print(f" {dfPL.head(5)}")

            # Sample fraction if desired
            frac = args.syn_use_frac  # e.g., 0.5 for 50%
            if 0 < frac < 1.0:
                dfPL = dfPL.sample(frac=frac, random_state=args.seed)

            # Optional subdir like "generatedImgs"
            syn_subdir = self.synthetic_path+args.syn_img_subdir

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
            print("Processing synthetic pseudolabels CSV entries... dfPL length:"+str(len(dfPL)))
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

        
        
        
        self.label_all = self.label

    def __getitem__(self, index):
        imgname, gt_label = self.img_id[index], self.label[index]

        # Pseudolabeled synthetic: absolute/resolved path stored directly
        if isinstance(imgname, str) and os.path.isabs(imgname):
            imgpath = imgname
            #print(f"Loading synthetic image from absolute path: {imgpath}")
        else:
            # Real image
            imgpath = os.path.join(self.root_path, imgname)

        img = Image.open(imgpath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        gt_label = np.asarray(gt_label).astype(np.float32)
        if self.target_transform:
            gt_label = gt_label[self.target_transform]

        return img, gt_label, imgname

    def __len__(self):
        return len(self.img_id)

def get_transform(args):
    height = args.height
    width = args.width
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform
