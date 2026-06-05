import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

# Ensure local third_party folder is on sys.path so downloaded metric helpers are importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY = os.path.join(SCRIPT_DIR, "third_party")
if os.path.isdir(THIRD_PARTY) and THIRD_PARTY not in sys.path:
    sys.path.insert(0, THIRD_PARTY)

# Also add FD-DINOv2 from third_party to sys.path if it exists
FD_DINOV2_PATH = os.path.join(THIRD_PARTY, "FD-DINOv2")
if os.path.isdir(FD_DINOV2_PATH) and FD_DINOV2_PATH not in sys.path:
    sys.path.insert(0, FD_DINOV2_PATH)

# Optional: Import evaluation metrics (assume external implementations)
from pytorch_fid import fid_score  # Requires pytorch-fid package
# For CMMD, import helper modules placed in `third_party` by prepareGithubs.sh
try:
    import distance
    import embedding
    import io_util
except Exception as e:
    raise ImportError(f"Could not import third-party metric helpers from {THIRD_PARTY}: {e}")

import subprocess

from pytorch_fid.inception import InceptionV3
from fid_score import compute_statistics_of_path


from fid_score import calculate_frechet_distance

import glob
import random

import math 
import re
import clip
import pandas as pd
import argparse
import importlib.util
from pathlib import Path

# Helper function to import dataset classes from src/lora_training/customDatasets
def _import_dataset_class(module_name: str, class_name: str):
    """Import a dataset class from src/lora_training/customDatasets with fallback to filesystem loading."""
    candidates = [
        f"lora_training.customDatasets.{module_name}",
        f"src.lora_training.customDatasets.{module_name}",
        f"customDatasets.{module_name}",
        module_name,
    ]
    for cand in candidates:
        try:
            mod = importlib.import_module(cand)
            return getattr(mod, class_name)
        except Exception:
            continue
    
    # Fallback: try to load the module directly from the repository src/lora_training/customDatasets path
    try:
        # Navigate from tools/metricAspect up to repo root, then to src/lora_training/customDatasets
        current_script_dir = Path(__file__).resolve().parent  # tools/metricAspect
        repo_root = current_script_dir.parents[1]  # go up 2 levels to repo root
        module_file = repo_root / "src" / "lora_training" / "customDatasets" / f"{module_name}.py"
        if module_file.exists():
            spec = importlib.util.spec_from_file_location(module_name, str(module_file))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            return getattr(mod, class_name)
    except Exception:
        pass
    
    raise ImportError(f"Could not import {class_name} from any of: {candidates} or filesystem path")


RAPzsDatasetAll = _import_dataset_class("RAPzsAll", "RAPzsDatasetAll")
PETAzsDatasetAll = _import_dataset_class("PETAzsAll", "PETAzsDatasetAll")
PETADatasetAll = _import_dataset_class("PETAAll", "PETADatasetAll")
RAPv1DatasetAll = _import_dataset_class("RAPv1All", "RAPv1DatasetAll")
RAPv2DatasetAll = _import_dataset_class("RAPv2All", "RAPv2DatasetAll")
PA100kDatasetAll = _import_dataset_class("PA100kAll", "PA100kDatasetAll")

from fid_score import ImagePathDataset
import torchvision.transforms as TF

import tqdm

from fid_score import extract_features
from fid_score import calculate_frechet_distance_fd

class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        self.image_files = [f for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)



def filter_prompts_by_string(df, query):
    # Escape special regex characters to treat query as a literal string
    escaped_query = re.escape(query)
    return df[df["prompt"].str.contains(escaped_query, case=False, na=False)]

class CFIDDataset(Dataset):
    def __init__(self, path, reshape_to, max_count=-1, syn=True, dataset=None):
        
        df = pd.read_csv(path+"generated.csv")
        
        if syn:
            self.path = path+"/generatedImgs/"
        else:
            self.path = path+"/condImgs/"

        self.reshape_to = reshape_to

        self.max_count = max_count

        assert max_count > len(dataset.listAttributePrompt)
        
        samplesPerAttribute=int(math.ceil(max_count/len(dataset.listAttributePrompt)))
        
        self.dictImagesByAttribute = {}

        for attribute in dataset.listAttributePrompt:
            print(attribute)
            matches = filter_prompts_by_string(df, attribute)
            #df["prompt_clean"] = df["prompt"].str.replace(r"\s+", " ", regex=True).str.strip()
            #matches =df[df["prompt_clean"].str.contains(attribute, case=False, na=False)]
            
            if syn:
                matches = list(matches['genImg'].iloc)
            else:
                matches = list(matches['condImg'].iloc)

            matches = matches[:samplesPerAttribute]
            listImgs=[self.path+img for img in matches]
            if len(matches) > 0:
                print("got")
                self.dictImagesByAttribute[dataset.listAttributes[dataset.listAttributePrompt.index(attribute)]]=listImgs

    #def __len__(self):
    #    return len(self.dictImagesByAttribute)

    def __len__(self, attribute):
        return len(self.dictImagesByAttribute[attribute])

    def _center_crop_and_resize(self, im, size):
        w, h = im.size
        l = min(w, h)
        top = (h - l) // 2
        left = (w - l) // 2
        box = (left, top, left + l, top + l)
        im = im.crop(box)
        # Note that the following performs anti-aliasing as well.
        return im.resize((size, size), resample=Image.BICUBIC)  # pytype: disable=module-attr

    def _read_image(self, path, size):
        im = Image.open(path)
        if size > 0:
            im = self._center_crop_and_resize(im, size)
        return np.asarray(im).astype(np.float32)

    def __getitem__(self, idx, attribute):
        listaImgs = self.dictImagesByAttribute[attribute]

        img_path = self.path+listaImgs[idx]

        x = self._read_image(img_path, self.reshape_to)
        if x.ndim == 3:
            return x
        elif x.ndim == 2:
            # Convert grayscale to RGB by duplicating the channel dimension.
            return np.tile(x[Ellipsis, np.newaxis], (1, 1, 3))

class CMMDDataset(Dataset):
    def __init__(self, path, reshape_to, max_count=-1, sorted=False):
        self.path = path
        self.reshape_to = reshape_to
        self.sorted = sorted

        self.max_count = max_count
        img_path_list = self._get_image_list()
        if max_count > 0:
            img_path_list = img_path_list[:max_count]
        self.img_path_list = img_path_list

    def __len__(self):
        return len(self.img_path_list)

    def _get_image_list(self):
        ext_list = ["png", "jpg", "jpeg"]
        image_list = []
        for ext in ext_list:
            image_list.extend(glob.glob(f"{self.path}/*{ext}"))
            image_list.extend(glob.glob(f"{self.path}/*.{ext.upper()}"))
        # Sort the list to ensure a deterministic output.
        if self.sorted:
            image_list.sort()
        else:
            random.shuffle(image_list)

        return image_list

    def _center_crop_and_resize(self, im, size):
        w, h = im.size
        l = min(w, h)
        top = (h - l) // 2
        left = (w - l) // 2
        box = (left, top, left + l, top + l)
        im = im.crop(box)
        # Note that the following performs anti-aliasing as well.
        return im.resize((size, size), resample=Image.BICUBIC)  # pytype: disable=module-attr

    def _read_image(self, path, size):
        im = Image.open(path)
        if size > 0:
            im = self._center_crop_and_resize(im, size)
        return np.asarray(im).astype(np.float32)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]

        x = self._read_image(img_path, self.reshape_to)
        if x.ndim == 3:
            return x
        elif x.ndim == 2:
            # Convert grayscale to RGB by duplicating the channel dimension.
            return np.tile(x[Ellipsis, np.newaxis], (1, 1, 3))



def compute_embeddings_for_dataset(
    dataset,
    embedding_model,
    batch_size
):
    """Computes embeddings for the images in the given directory.

    This drops the remainder of the images after batching with the provided
    batch_size to enable efficient computation on TPUs. This usually does not
    affect results assuming we have a large number of images in the directory.

    Args:
      img_dir: Directory containing .jpg or .png image files.
      embedding_model: The embedding model to use.
      batch_size: Batch size for the embedding model inference.
      max_count: Max number of images in the directory to use.

    Returns:
      Computed embeddings of shape (num_images, embedding_dim).
    """
    
    dataloader = DataLoader(dataset, batch_size=batch_size)
    count=len(dataloader)
    print(f"Calculating embeddings for {len(dataset)} images.")

    all_embs = []
    for batch in tqdm.tqdm(dataloader, total=count // batch_size):
        image_batch = batch.numpy()
        
        # Normalize to the [0, 1] range.
        image_batch = image_batch / 255.0

        if np.min(image_batch) < 0 or np.max(image_batch) > 1:
            raise ValueError(
                "Image values are expected to be in [0, 1]. Found:" f" [{np.min(image_batch)}, {np.max(image_batch)}]."
            )

        # Compute the embeddings using a pmapped function.
        embs = np.asarray(
            embedding_model.embed(image_batch)
        )  # The output has shape (num_devices, batch_size, embedding_dim).
        all_embs.append(embs)

    all_embs = np.concatenate(all_embs, axis=0)

    return all_embs


def computeCMMD_for_dataloader(dataset, batch_size=32):
    global embedding_model
    embs = compute_embeddings_for_dataset(dataset, embedding_model, batch_size).astype(
        "float32"
    )
    return embs

def computeCMMD(path, batch_size=32, max_count=-1):
    global embedding_model
    embs = io_util.compute_embeddings_for_dir(path, embedding_model, batch_size, max_count).astype(
        "float32"
    )
    return embs

def compute_cmmd(ref_embs, eval_embs):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    
    val = distance.mmd(ref_embs, eval_embs)
    return val.numpy()

modelFID = None
dims=2048
#dims=1024
device="cuda"

embedding_model = None


from fid_score import calculate_activation_statistics

def setModelCMMD():
    global embedding_model
    embedding_model = embedding.ClipEmbeddingModel()
    #embedding_model.to(device)

try:
    from FDDINOv2.utils.load_encoder import load_encoder
except ImportError:
    # Fallback if FD-DINOv2 is not in third_party
    try:
        from utils.load_encoder import load_encoder
    except ImportError:
        load_encoder = None

#from FDDINOv2.src.load_encoder import load_encoder
#check "vit_small_patch16_224_dino"
import timm

def setModelFDDINO():
    global modelFDDINO
    #modelFDDINO = load_encoder("dinov2", device, ckpt = None, arch = None,
    #                        clean_resize = False,
    #                        sinception = False,
    #                        depth = 0)
    modelFDDINO = timm.create_model("vit_small_patch16_224_dino", pretrained=True)
    modelFDDINO.eval()
    modelFDDINO.cuda()

def setModelFID():
    global modelFID
    global dims
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    modelFID = InceptionV3([block_idx]).to(device)


def getDistributionFDDINO_for_images_paths_list(files, batch_size=32, num_workers=1):
    global modelFDDINO
    
    m1, s1 = calculate_activation_statistics(
        files, modelFDDINO, batch_size=batch_size, dims=2048, device="cuda", num_workers=1
    )
    
    return m1, s1

def getDistributionFID_for_images_paths_list(files, batch_size=32, num_workers=1):
    global modelFID
    
    m1, s1 = calculate_activation_statistics(
        files, modelFID, batch_size=batch_size, dims=2048, device="cuda", num_workers=1
    )
    
    return m1, s1



def getDistributionFID(path, batch_size=32, num_workers=1):
    global modelFID
    m1, s1 = compute_statistics_of_path(
        path, modelFID, batch_size, dims, device, num_workers
    )

    return m1, s1

from fid_score import IMAGE_EXTENSIONS, get_representations_fd, compute_statistics_fd
import pathlib
import random 
def getDistributionFDDino(path, batch_size=32, num_workers=1, numImages=0):
    global modelFDDINO

    path = pathlib.Path(path)
    files = sorted(
        [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
    )
    
    if numImages > 0:
        files = random.choices(files, k=numImages)
    
    dataset = ImagePathDataset(files,
        transforms=TF.Compose([
            TF.Resize((224,224)),
            TF.ToTensor(),
    ]))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    reps = get_representations_fd(modelFDDINO, dataloader, "cuda")
    mu, sigma = compute_statistics_fd(reps)
    return mu, sigma

def calculateFID(mReal, sReal, mSyn, sSyn):

    fid_value = calculate_frechet_distance(mReal, sReal, mSyn, sSyn)

    return fid_value


def getScoreCMMD(fake_image_dir, n,  emb_real_cmmd, batch_size):
   
    # create new dataset
    fake_dataset = CMMDDataset(fake_image_dir, reshape_to=512, max_count=n)

    # Compute CMMD (use real features)
    emb_syn_cmmd = computeCMMD_for_dataloader(fake_dataset, batch_size=batch_size)
    cmmd = float(compute_cmmd(emb_real_cmmd, emb_syn_cmmd))

    return cmmd

def getScoreFID(fake_image_dir, n, mReal, sReal, batch_size):

    # create new dataset
    fake_dataset = CMMDDataset(fake_image_dir, reshape_to=512, max_count=n)

    # Compute FID
    mSyn, sSyn = getDistributionFID_for_images_paths_list(fake_dataset.img_path_list, batch_size=batch_size, num_workers=1)
    fid = calculateFID(mReal, sReal, mSyn, sSyn)

    return fid



def getScoreFDDino(fake_image_dir, n, mReal, sReal,batch_size, numImages=0):

    
    # Compute FDDINO
    fdDINOmSyn, fdDINOsSyn = getDistributionFDDino(fake_image_dir, batch_size=batch_size, num_workers=1, numImages=numImages)
    
    
    fdDINO = calculate_frechet_distance_fd(mReal, sReal, fdDINOmSyn, fdDINOsSyn)

    return fdDINO



def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="RAPzs", type=str)
    parser.add_argument("--nMuestreo", default=5, type=int)
    parser.add_argument("--batchSize", default=32, type=int)
    parser.add_argument("--pathGen", default="", type=str)
    parser.add_argument("--pathReal", default="", type=str)
    parser.add_argument("--pathSyn", default="", type=str)
    parser.add_argument("--pathGraphic", default="", type=str)
    parser.add_argument(
        "--sampleSizes",
        nargs="+",
        type=str,
        default=None,
        help=(
            "List of sample sizes to evaluate (e.g., --sampleSizes 100 500 1000 or include 'all' to use full dataset). "
            "If not provided, defaults to [100, 500, 1000, 5000, 10000, 15000, dataset_length]."
        ),
    )
    
    parser.add_argument("--FID", action='store_true')
    parser.add_argument("--CFID", action='store_true')
    parser.add_argument("--CMMD", action='store_true')
    parser.add_argument("--FD", action='store_true')

    args = parser.parse_args()
    return args


def _parse_sample_sizes(tokens, full_size):
    """Parse a list of tokens from --sampleSizes into integer sizes.

    - Accepts integers as strings (e.g., "100").
    - Accepts the literal 'all' (case-insensitive) to mean the full dataset size.
    - Returns a list in the given order, allowing duplicates if provided.
    """
    sizes = []
    for tok in tokens:
        if isinstance(tok, str) and tok.lower() == "all":
            sizes.append(int(full_size))
        else:
            try:
                sizes.append(int(tok))
            except Exception:
                raise ValueError(f"Invalid sample size token: {tok}")
    return sizes




def calculate_cfid(y_true, y_pred, x_true, x_true_pred):
    """
    Computes Conditional Fréchet Inception Distance (CFID) in PyTorch.
    """
    # Center the data
    m_y_true = y_true.mean(dim=0)
    m_y_pred = y_pred.mean(dim=0)

    m_x_true = x_true.mean(dim=0)
    m_x_true_pred = x_true_pred.mean(dim=0)

    y_true_centered = y_true - m_y_true
    y_pred_centered = y_pred - m_y_pred
    x_centered = x_true - m_x_true
    x_centered_pred = x_true_pred - m_x_true_pred

    # Covariances
    C_yy = covariance(y_true_centered)
    C_ypyp = covariance(y_pred_centered)
    C_yx = covariance(y_true_centered, x_centered)
    C_ypx = covariance(y_pred_centered, x_centered_pred)
    C_xy = covariance(x_centered, y_true_centered)
    C_xyp = covariance(x_centered_pred, y_pred_centered)
    C_xx = covariance(x_centered)

    C_xx_pred = covariance(x_centered_pred)

    # Inverse of C_xx
    C_xx_inv = torch.linalg.pinv(C_xx)
    C_xx_inv_pred = torch.linalg.pinv(C_xx_pred)

    # Conditional covariances
    C_yy_given_x = C_yy - C_yx @ C_xx_inv @ C_xy
    C_ypyp_given_x = C_ypyp - C_ypx @ C_xx_inv_pred @ C_xyp

    # Mean difference term
    mean_diff = torch.sum((m_y_true - m_y_pred) ** 2)

    # Cross-covariance alignment term
    cross_cov_diff = C_yx - C_ypx
    cross_cov_term = torch.trace(cross_cov_diff @ C_xx_inv @ cross_cov_diff.T)

    # Conditional covariance alignment term
    cov_term = torch.trace(C_yy_given_x + C_ypyp_given_x) - 2 * trace_sqrt_product(C_yy_given_x, C_ypyp_given_x)

    return mean_diff + cross_cov_term + cov_term



def trace_sqrt_product(A, B, eps=1e-10):
    """
    Computes Tr((A^{1/2} B A^{1/2})^{1/2}) using symmetric matrix square root.
    """
    sqrt_A = matrix_sqrt(A + eps * torch.eye(A.size(0)))
    inner = sqrt_A @ B @ sqrt_A
    sqrt_inner = matrix_sqrt(inner + eps * torch.eye(inner.size(0)))
    return torch.trace(sqrt_inner)

def matrix_sqrt(matrix):
    """
    Computes the matrix square root using eigen decomposition.
    """
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=0))
    return eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T


def covariance(x, y=None):
    """
    Computes the covariance matrix between x and y.
    """
    if y is None:
        y = x
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    return x.T @ y / (x.size(0) - 1)




def getEmbeddingsFID(files, batch_size, num_workers=1):
    
    datasetTF = ImagePathDataset(files,
        transforms=TF.Compose([
            TF.Resize((299,299)),
            TF.ToTensor(),
    ]))
    dataloaderTF = torch.utils.data.DataLoader(
        datasetTF,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm.tqdm(dataloaderTF):
        batch = batch.to(device)

        with torch.no_grad():
            pred = modelFID(batch)[0]

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]
    
    pred_torch = torch.from_numpy(pred_arr).float() # .float() ensures it's in float32

    return pred_torch

def getEmbeddingText(prompt):
    global modelEmbeddingTxt
    global projection
    text = clip.tokenize(prompt).to(device)  # or any prompt
    with torch.no_grad():
        text_embedding = modelEmbeddingTxt.encode_text(text)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

    
    
    text_2048 = projection(text_embedding.float())


    return text_2048

modelEmbeddingTxt=None
modelNameTxt="RN50x64"
projection=None

def setModelTxt():
    global modelEmbeddingTxt
    global projection
    modelEmbeddingTxt, preprocess = clip.load(modelNameTxt, device=device)
    projection = torch.nn.Linear(1024, 2048)
    projection.float()
    projection.to(device)


def setDataset(datasetName):
    
    if datasetName == "RAPzs":
        dataset=RAPzsDatasetAll(split="train")
    
    if datasetName == "PA100k":
        dataset=PA100kDatasetAll(split="train")

    # 50957
    if datasetName == "PA100k_reduced":
        dataset=PA100kDatasetAll(split="train")

    if datasetName == "PETAzs":
        dataset=PETAzsDatasetAll(split="train")

    if datasetName == "PETA":
        dataset=PETADatasetAll(split="train")

    if datasetName == "RAPv1":
        dataset=RAPv1DatasetAll(split="train")

    if datasetName == "RAPv2":
        dataset=RAPv2DatasetAll(split="train")

    return dataset

def calculateCFID(args):

    setModelFID()
    setModelTxt()

    pathGen=args.pathGen
    nMuestreo=args.nMuestreo

    dataset=setDataset(args.dataset)

    
    originalCFIDDataset = CFIDDataset(pathGen, reshape_to=512, max_count=len(dataset), syn=False, dataset=dataset)

    # Use provided sample sizes or default. If 'all' is present it expands to len(dataset).
    if args.sampleSizes is not None:
        sample_sizes = _parse_sample_sizes(args.sampleSizes, len(dataset))
    else:
        sample_sizes = [100, 500, 1000, 5000, 10000, 15000, len(dataset)]
    #sample_sizes = [len(dataset)]
    #sample_sizes=[100]

    #originalCFIDDataset
    print("Getting CFID real dataset")
    resultsReal = {}
    resultsRealTxt = {}
    for attr in originalCFIDDataset.dictImagesByAttribute:
        print("getting CFID attribute "+attr)
        real_imgs = originalCFIDDataset.dictImagesByAttribute[attr]
        embeddings = getEmbeddingsFID(real_imgs, batch_size=args.batchSize, num_workers=1)
        prompt="A photo of "+dataset.listAttributePrompt[dataset.listAttributes.index(attr)]  

        embeddingTxt=getEmbeddingText(prompt)
        embeddingTxt=embeddingTxt.to("cpu")
        tensorsembeddingsTxt=embeddingTxt.repeat(len(real_imgs), 1)
        resultsRealTxt[attr]=tensorsembeddingsTxt
        resultsReal[attr] = embeddings

    results={}
    CFIDMedium=0
    for n in sample_sizes:
        print("Sample size "+str(n))
        firstTime=True
        # store max CFID from each attribute in n samples
        CFIDMax={}
        CFIDMin={}

        # store list of values in the n samples and then make the median
        CFID_scoresMed = {}
        for i in range(nMuestreo):

            synCFIDDataset = CFIDDataset(pathGen, reshape_to=512, max_count=n, syn=True, dataset=dataset)
            
            for attr in synCFIDDataset.dictImagesByAttribute:
                prompt="A photo of "+dataset.listAttributePrompt[dataset.listAttributes.index(attr)]  

                embeddingTxt=getEmbeddingText(prompt)
                embeddingsReal=resultsReal[attr]

                if attr not in CFID_scoresMed.keys():
                    CFID_scoresMed[attr]=0
                
                syn_imgs = synCFIDDataset.dictImagesByAttribute[attr]
                embeddingsSyn = getEmbeddingsFID(syn_imgs, batch_size=args.batchSize, num_workers=1)
                #CFIDmSyn, CFIDsSyn = getDistributionFID_for_images_paths_list(syn_imgs, batch_size=args.batchSize, num_workers=1)
                embeddingsReal=embeddingsReal.to("cpu")
                embeddingsSyn=embeddingsSyn.to("cpu")
                embeddingTxt=embeddingTxt.to("cpu")

                tensorsembeddingsTxtSyn=embeddingTxt.repeat(len(syn_imgs), 1)
                tensorsembeddingsTxtReal = resultsRealTxt[attr]

                print("Embedding img real"+str(embeddingsReal.shape))
                print("Embedding img syn"+str(embeddingsSyn.shape))
                print("Embedding txt real"+str(tensorsembeddingsTxtReal.shape))
                print("Embedding txt syn"+str(tensorsembeddingsTxtSyn.shape))
                CFID = calculate_cfid(embeddingsReal, embeddingsSyn, tensorsembeddingsTxtReal, tensorsembeddingsTxtSyn)
                CFID=CFID.item()
                if firstTime:
                    CFIDMax[attr]=CFID
                    CFIDMin[attr]=CFID

                else:
                    if CFIDMax[attr] < CFID:
                        CFIDMax[attr]=CFID
                    if CFIDMin[attr] > CFID:
                        CFIDMin[attr]=CFID

                CFID_scoresMed[attr]+=CFID

        
        results[n]={}
        for attr in synCFIDDataset.dictImagesByAttribute:
            
            results[n][attr]={}
            results[n][attr]['med']=float(CFID_scoresMed[attr]/nMuestreo)
            results[n][attr]['max']=CFIDMax[attr]
            results[n][attr]['min']=CFIDMin[attr]
            CFIDMedium+=float(CFID_scoresMed[attr]/nMuestreo)
        
        CFIDMedium=CFIDMedium/len(synCFIDDataset.dictImagesByAttribute)
        results[n]['average']={}
        results[n]['average']['med']=CFIDMedium
        results[n]['average']['max']=CFIDMedium
        results[n]['average']['min']=CFIDMedium
        
        createPandasMetricByCategory_new('CFID', args, results)

    return


def createPandasMetric(metric, args, dictionary):
    
    #print(dictionary)
    os.makedirs(args.pathGraphic, exist_ok=True)
    with pd.ExcelWriter(args.pathGraphic+'/'+metric+'Scores_output_all.xlsx', engine='openpyxl') as writer:
        for metric_name, sample_data in dictionary.items():
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(sample_data, orient="index")
            df.index.name = 'sample_size'
            df.reset_index(inplace=True)  # <- this makes sample_size a column

            sheet_name = str(metric)[:31].replace("/", "_").replace("\\", "_")
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Excel file created: "+args.pathGraphic+'/'+metric+'Scores_output_all.xlsx')
        

    return

def createPandasMetric_new(metric, args, dictionary):
    """
    Converts dictionary into rows and writes to <metric>Scores_output_all.csv
    - If file exists: append without header
    - If file does not exist: create with header
    Columns: MetricName, sample_size, <all keys from sample_data>
    """
    os.makedirs(args.pathGraphic, exist_ok=True)
    out_csv = os.path.join(args.pathGraphic, f"{metric}Scores_output_all.csv")

    # Build rows for all metric_name entries
    frames = []
    for metric_name, sample_data in dictionary.items():
        df = pd.DataFrame.from_dict(sample_data, orient="index")
        df.index.name = "sample_size"
        df.reset_index(inplace=True)
        df.insert(0, "MetricName", metric_name)  # prepend column
        frames.append(df)

    if not frames:
        print("⚠️ No data to write")
        return

    result = pd.concat(frames, ignore_index=True)

    # Append or create
    file_exists = os.path.exists(out_csv)
    result.to_csv(out_csv, mode="a" if file_exists else "w",
                  header=not file_exists, index=False)

    print(("Appended to: " if file_exists else "Created: ") + out_csv)

def createPandasMetricByCategory(metric, args, dictionary):
    
    categories = set()
    for sample_data in dictionary.values():
        categories.update(sample_data.keys())

    os.makedirs(args.pathGraphic, exist_ok=True)
    with pd.ExcelWriter(args.pathGraphic+'/'+metric+'Scores_output_all.xlsx', engine='openpyxl') as writer:
        for category in categories:
            rows = []
            for sample_name, sample_data in dictionary.items():
                if category in sample_data:
                    row = {
                        "Sample": sample_name,
                        "Min": sample_data[category].get("min", np.nan),
                        "Med": sample_data[category].get("med", np.nan),
                        "Max": sample_data[category].get("max", np.nan),
                    }
                    rows.append(row)

            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=str(category), index=False)

    print("Excel file created: "+args.pathGraphic+'/'+metric+'Scores_output_all.xlsx')
        

    return


def createPandasMetricByCategory_new(metric, args, dictionary):
    """
    Build rows with columns:
      Sample, Category, Min, Med, Max
    Write to <args.pathGraphic>/<metric>Scores_output_all.csv
      - If the CSV exists: append without header
      - If it doesn't: create with header
    """
    # Collect unique categories
    categories = set()
    for sample_data in dictionary.values():
        categories.update(sample_data.keys())

    # Build rows
    rows = []
    for category in categories:
        for sample_name, sample_data in dictionary.items():
            if category in sample_data:
                rows.append({
                    "Sample": sample_name,
                    "Category": str(category),
                    "Min": sample_data[category].get("min", np.nan),
                    "Med": sample_data[category].get("med", np.nan),
                    "Max": sample_data[category].get("max", np.nan),
                })

    # Make sure output dir exists
    os.makedirs(args.pathGraphic, exist_ok=True)
    out_csv = os.path.join(args.pathGraphic, f"{metric}Scores_output_all.csv")

    # Create or append
    file_exists = os.path.exists(out_csv)
    df = pd.DataFrame(rows, columns=["Sample", "Category", "Min", "Med", "Max"])
    df.to_csv(out_csv, mode="a" if file_exists else "w",
              header=not file_exists, index=False)

    print(("Appended to: " if file_exists else "Created: ") + out_csv)


def calculateCMMD(args):
    real_image_dir=args.pathReal
    fake_image_dir=args.pathSyn
    nMuestreo=args.nMuestreo
    batch_size=args.batchSize
    setModelCMMD()
    # Create subset datasets (simulated with slicing for now)
    #rapv2 max = 50957
    
    if args.dataset=="PA100k_reduced":
        max_countRapv2=50957
        fake_dataset = CMMDDataset(fake_image_dir, reshape_to=512, max_count=max_countRapv2)

    else:
        fake_dataset = CMMDDataset(fake_image_dir, reshape_to=512, max_count=-1)

    # Use provided sample sizes or default. If 'all' is present it expands to len(fake_dataset).
    if args.sampleSizes is not None:
        sample_sizes = _parse_sample_sizes(args.sampleSizes, len(fake_dataset))
    else:
        sample_sizes = [100, 500, 1000, 5000, 10000, 15000, len(fake_dataset)]
    #sample_sizes = [len(fake_dataset)]

    # get embedding for Real
    if args.dataset=="PA100k_reduced":
        max_countRapv2=50957
        emb_real_cmmd = computeCMMD(real_image_dir, batch_size=batch_size, max_count=max_countRapv2)
    else:
        emb_real_cmmd = computeCMMD(real_image_dir, batch_size=batch_size, max_count=-1)
    
    metric='CMDD'
    results={}
    results[metric]={}

    for n in sample_sizes:
        print("Sample size "+str(n))
        if n < len(fake_dataset):
            firstTime=True
            cmmdSum=0

            for i in range(nMuestreo):

                cmmd = getScoreCMMD(fake_image_dir, n, emb_real_cmmd, batch_size)

                if firstTime:
                    maxCMMD=cmmd
                    minCMMD=cmmd
                    
                    firstTime=False
                    
                else:
                    
                    if cmmd < minCMMD:
                        minCMMD=cmmd
                    if cmmd > maxCMMD:
                        maxCMMD=cmmd

                cmmdSum+=cmmd
            # calculate median scores
            cmmdMed=float(cmmdSum/nMuestreo)
        else:
            cmmd = getScoreCMMD(fake_image_dir, n, emb_real_cmmd, batch_size)

            cmmdMed=float(cmmd)
            minCMMD=float(cmmd)
            maxCMMD=float(cmmd)

        results[metric][n]={}
        
        results[metric][n]={}
        results[metric][n]['med']=cmmdMed
        results[metric][n]['max']=maxCMMD
        results[metric][n]['min']=minCMMD


        createPandasMetric_new(metric, args, results)
    
    emb_real_cmmd = None
    fake_dataset = None
    results = None


    return


def calculateFIDmetric(args):
    real_image_dir=args.pathReal
    fake_image_dir=args.pathSyn
    nMuestreo=args.nMuestreo
    batch_size=args.batchSize
    setModelFID()
    # Create subset datasets (simulated with slicing for now)
    #real_dataset = CMMDDataset(real_image_dir)

    if args.dataset=="PA100k_reduced":
        max_countRapv2=50957
        fake_dataset = CMMDDataset(fake_image_dir, reshape_to=512, max_count=max_countRapv2)
    else:
        fake_dataset = CMMDDataset(fake_image_dir, reshape_to=512, max_count=-1)

    # Use provided sample sizes or default. If 'all' is present it expands to len(fake_dataset).
    if args.sampleSizes is not None:
        sample_sizes = _parse_sample_sizes(args.sampleSizes, len(fake_dataset))
    else:
        sample_sizes = [100, 500, 1000, 5000, 10000, 15000, len(fake_dataset)]
    #sample_sizes = [len(fake_dataset)]

    # get embedding for Real
    mReal, sReal = getDistributionFID(real_image_dir, batch_size=batch_size, num_workers=1)
    

    metric='FID'
    results={}
    results[metric]={}
    for n in sample_sizes:
        print("Sample size "+str(n))
        if n < len(fake_dataset):
            firstTime=True

            fidSum=0

            for i in range(nMuestreo):

                fid = getScoreFID(fake_image_dir, n, mReal, sReal, batch_size)

                if firstTime:
                    maxFID=fid
                    minFID=fid
                    
                    firstTime=False
                    
                else:
                    if fid < minFID:
                        minFID=fid
                    if fid > maxFID:
                        maxFID=fid
                    
                        
                fidSum+=fid
            # calculate median scores
            fidMed=float(fidSum/nMuestreo)
        else:
            fid = getScoreFID(fake_image_dir, n, mReal, sReal, batch_size)
            fidMed=float(fid)
            minFID=float(fid)
            maxFID=float(fid)


        results[metric][n]={}
        
        results[metric][n]={}
        results[metric][n]['med']=fidMed
        results[metric][n]['max']=maxFID
        results[metric][n]['min']=minFID


        createPandasMetric_new(metric, args, results)

    mReal = None
    sReal = None
    fake_dataset = None
    results = None

    return 



def calculateFDDino(args):
    real_image_dir=args.pathReal
    fake_image_dir=args.pathSyn
    nMuestreo=args.nMuestreo
    batch_size=args.batchSize
    # set model as x
    setModelFDDINO()
    
    if args.dataset=="PA100k_reduced":
        max_countRapv2=50957
        fake_dataset = CMMDDataset(fake_image_dir, reshape_to=512, max_count=max_countRapv2)
    else:
        fake_dataset = CMMDDataset(fake_image_dir, reshape_to=512, max_count=-1)

    # Use provided sample sizes or default. If 'all' is present it expands to len(fake_dataset).
    if args.sampleSizes is not None:
        sample_sizes = _parse_sample_sizes(args.sampleSizes, len(fake_dataset))
    else:
        sample_sizes = [100, 500, 1000, 5000, 10000, 15000, len(fake_dataset)]
    #sample_sizes = [len(fake_dataset)]
    #sample_sizes = [100, 500]
    # get embedding for Real
    #cambiar para sacarlo con dino
    
    mReal, sReal = getDistributionFDDino(real_image_dir, batch_size=batch_size, num_workers=1, numImages=0)
    
    metric='FDDINO'
    results={}
    results[metric]={}

    for n in sample_sizes:
        print("Sample size "+str(n))
        if n < len(fake_dataset):
            print("less than n")
            firstTime=True
            fdDINOSum=0
            for i in range(nMuestreo):

                fdDINO = getScoreFDDino(fake_image_dir, n, mReal, sReal, batch_size, numImages=n)
                print(fdDINO)
                if firstTime:
                    maxfdDINO=fdDINO
                    minfdDINO=fdDINO
                    
                    firstTime=False
                    
                else:

                    if fdDINO < minfdDINO:
                        minfdDINO=fdDINO
                    if fdDINO > maxfdDINO:
                        maxfdDINO=fdDINO
  
                fdDINOSum+=fdDINO
            # calculate median scores
            fdDINOMed=float(fdDINOSum/nMuestreo)
        else:
            fdDINO = getScoreFDDino(fake_image_dir, n, mReal, sReal, batch_size, numImages=n)
            fdDINOMed=float(fdDINO)
            minfdDINO=float(fdDINO)
            maxfdDINO=float(fdDINO)


        
        results[metric][n]={}
        results[metric][n]['med']=fdDINOMed
        results[metric][n]['max']=maxfdDINO
        results[metric][n]['min']=minfdDINO

    

        createPandasMetric_new(metric, args, results)

    mReal = None
    sReal = None
    fake_dataset = None
    results = None

    return 


def main():
    args = parseArgs()

    if args.FID:
        calculateFIDmetric(args)

    if args.FD:
        calculateFDDino(args)
    
    if args.CFID:
        calculateCFID(args)
    
    if args.CMMD:
        calculateCMMD(args)


if __name__ == "__main__":
    main()
