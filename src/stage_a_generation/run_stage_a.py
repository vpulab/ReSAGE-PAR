#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import importlib

def _import_dataset_class(module_name: str, class_name: str):
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
    # Fallback: try to load the module directly from the repository `src/lora_training/customDatasets` path
    try:
        import importlib.util
        src_dir = Path(__file__).resolve().parents[1]
        module_file = src_dir / "lora_training" / "customDatasets" / f"{module_name}.py"
        if module_file.exists():
            spec = importlib.util.spec_from_file_location(module_name, str(module_file))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            return getattr(mod, class_name)
    except Exception:
        pass

    raise ImportError(f"Could not import {class_name} from any of: {candidates} or filesystem path {module_file}")


RAPv1DatasetAll   = _import_dataset_class("datasetRAPv1All", "RAPv1DatasetAll")
RAPv2DatasetAll   = _import_dataset_class("datasetRAPv2All", "RAPv2DatasetAll")
RAPzsDatasetAll   = _import_dataset_class("datasetRAPzsAll", "RAPzsDatasetAll")
PETAzsDatasetAll  = _import_dataset_class("datasetPETAzsAll", "PETAzsDatasetAll")
PETADatasetAll    = _import_dataset_class("datasetPETAAll", "PETADatasetAll")
PA100kDatasetAll  = _import_dataset_class("datasetPA100kAll", "PA100kDatasetAll")

try:
    from .attribute_editing import AttributeEditing
    from .generation_prompt_formatting import PromptGenerator
except ImportError:
    from attribute_editing import AttributeEditing
    from generation_prompt_formatting import PromptGenerator

if is_wandb_available():
    import wandb

import mlflow

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__, log_level="INFO")


proxy = "http://192.168.22.3:8080"
# If needed: proxy = "http://user:pass@your.proxy.com:port"

os.environ["HTTP_PROXY"] = proxy
os.environ["HTTPS_PROXY"] = proxy
os.environ["http_proxy"] = proxy
os.environ["https_proxy"] = proxy


from io import BytesIO
import base64

import pandas as pd

def generateFinalVal(args, dataset, pipeline, generator, attribute_editor, prompt_generator, seed):
    """
    Extended: if batch contains 'vector', save it to CSV with one column per attribute.
    Attribute names come from:
      1) DATASETS_MAPPING_VECTORS[dataset_key] -> 'attr_name' from the pickle,
      2) fallback: attr_0..attr_{D-1} inferred from the vector length.
    If the CSV already exists with fewer columns, it is upgraded to include the full attribute list.
    Applies attribute editing and prompt generation when vectors are present.
    """
    import pickle

    print("Final validation")
    out_dir = f"{args.path_syn}_{seed}" if seed is not None else args.path_syn
    os.makedirs(out_dir, exist_ok=True)

    patToSaveImgsCond = os.path.join(out_dir, "condImgs")
    patToSaveImgsGen  = os.path.join(out_dir, "generatedImgs")
    os.makedirs(patToSaveImgsCond, exist_ok=True)
    os.makedirs(patToSaveImgsGen,  exist_ok=True)

    csv_path = os.path.join(out_dir, "generated.csv")
    pipeline.set_progress_bar_config(disable=True)

    # Load attribute names from dataset class
    attribute_names = None
    if args.dataset_name in DATASET_CLASSES:
        print(f"Loading attribute names from dataset class: {args.dataset_name}")
        try:
            # Instantiate the dataset class to access listAllAttrib
            dataset_class = DATASET_CLASSES[args.dataset_name]
            # Most dataset classes expect a 'split' argument
            dataset_instance = dataset_class(split="train")
            if hasattr(dataset_instance, 'listAttributes'):
                attribute_names = [str(a).strip() for a in dataset_instance.listAttributes]
                print(f"Loaded {len(attribute_names)} attribute names from class: {attribute_names[:5]}...")
            else:
                print(f"Warning: Dataset class has no 'listAttributes' attribute")
        except Exception as e:
            print(f"Warning: Could not load attribute names from dataset class: {e}")
            import traceback
            traceback.print_exc()
            attribute_names = None
    else:
        print(f"Warning: Dataset {args.dataset_name} not in DATASET_CLASSES")

    base_columns = ["condImg", "genImg", "prompt"]
    columns = None
    columns_gt = None
    columns_sel = None

    # Prepare dataset object
    ds = dataset["train"] if hasattr(dataset, "keys") and "train" in dataset else dataset
    n = len(ds)

    buffer = []
    batch_size = 96
    batch_size_save = args.batch_size_memory
    cont = 0

    # Helper to (re)create or upgrade CSV header to include all columns
    def ensure_csv_header(csv_path, desired_cols):
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            pd.DataFrame(columns=desired_cols).to_csv(csv_path, index=False)
            print(f"Created new CSV with columns: {desired_cols[:10]}..." if len(desired_cols) > 10 else f"Created new CSV with columns: {desired_cols}")
            return

        # If exists, check and upgrade header if needed
        existing_cols = list(pd.read_csv(csv_path, nrows=0).columns)
        if existing_cols != desired_cols:
            print("Upgrading existing CSV to include full attribute columns...")
            df_old = pd.read_csv(csv_path)
            # Find missing columns
            missing_cols = [c for c in desired_cols if c not in df_old.columns]
            if missing_cols:
                # Create DataFrame with missing columns (all NaN) - avoids fragmentation warning
                missing_df = pd.DataFrame(np.nan, index=df_old.index, columns=missing_cols)
                df_old = pd.concat([df_old, missing_df], axis=1)
            # Reorder to match desired order
            df_old = df_old[desired_cols]
            df_old.to_csv(csv_path, index=False)
            print(f"CSV header upgraded. Added {len(missing_cols)} new columns.")

    # Iterate by batch
    for start in tqdm(range(0, n, batch_size), desc="Batches"):
        end = min(start + batch_size, n)
        batch = ds[start:end]  # dict-of-lists: {'image': [...], 'text': [...], 'vector': [[...], ...], ...}

        # Debug: Check what keys are in batch
        if start == 0:
            print(f"DEBUG: Batch keys: {batch.keys()}")
            print(f"DEBUG: First item keys if accessing ds[0]: {ds[0].keys() if hasattr(ds[0], 'keys') else 'N/A'}")

        prompts   = list(batch["text"]) if "text" in batch else ["" for _ in range(end - start)]
        imgs_cond = [im.resize((args.width, args.height)) for im in batch["image"]]

        has_vector = "vector" in batch
        vectors_edited = None
        
        # Debug: Report if vectors are found
        if start == 0:
            print(f"DEBUG: has_vector = {has_vector}")
            if has_vector:
                print(f"DEBUG: Number of vectors in batch: {len(batch['vector'])}")
                print(f"DEBUG: First vector shape/length: {np.asarray(batch['vector'][0]).shape}")

        # Apply attribute editing and prompt generation if vectors are present
        if has_vector and attribute_editor and prompt_generator:
            vectors_edited = []
            for i, vec in enumerate(batch["vector"]):
                edited_vec = attribute_editor.produceSelectedAttributes(vec)
                vectors_edited.append(edited_vec)
                # generatePrompt may return (prompt, vector) tuple or just prompt string
                result = prompt_generator.generatePrompt(edited_vec)
                
                if isinstance(result, tuple):
                    prompts[i] = result[0]  # Extract just the prompt string
                else:
                    prompts[i] = result

        # Finalize columns (once), now that we know whether 'vector' is present
        if columns is None:
            if has_vector:
                # Determine attribute_names if still None (fallback to vector length)
                if attribute_names is None:
                    first_vec = np.asarray(batch["vector"][0]).ravel()
                    D = int(first_vec.shape[0])
                    attribute_names = [f"attr_{i}" for i in range(D)]
                    print(f"WARNING: Using fallback attribute names (attr_0, attr_1, ...). Check pickle file.")
                else:
                    print(f"Using {len(attribute_names)} real attribute names from pickle.")
                
                # Create separate column lists for gt and sel
                columns_gt = [f"{name}_gt" for name in attribute_names]
                columns_sel = [f"{name}_sel" for name in attribute_names]
                columns = base_columns + columns_gt + columns_sel
                print(f"CSV will have {len(columns)} columns: {columns[:10]}..." if len(columns) > 10 else f"CSV will have columns: {columns}")
            else:
                columns = base_columns
                print(f"No vectors found - CSV will only have base columns: {columns}")

            ensure_csv_header(csv_path, columns)

        # Run the image pipeline
        out = pipeline(
            prompts,
            image=imgs_cond,
            num_inference_steps=30,
            generator=generator,
            height=args.height,
            width=args.width,
            callback=None,
            callback_steps=None,
        )
        gens = out.images  # list of PIL Images

        # Save images + build CSV rows
        for i, (img_c, img_g, prompt) in enumerate(zip(imgs_cond, gens, prompts)):
            cond_name = f"img-{cont}.png"
            gen_name  = f"img-{cont}.png"

            img_c.save(os.path.join(patToSaveImgsCond, cond_name))
            img_g.save(os.path.join(patToSaveImgsGen,  gen_name))

            row = {"condImg": cond_name, "genImg": gen_name, "prompt": prompt}
            
            if has_vector:
                # Original (ground truth) vector
                vec_gt = np.asarray(batch["vector"][i]).ravel().tolist()
                
                # Edited (selected) vector - use edited if available, otherwise original
                vec_sel = vectors_edited[i] if vectors_edited else batch["vector"][i]
                vec_sel = np.asarray(vec_sel).ravel().tolist()

                # Align length with attribute_names (truncate/pad as needed)
                if len(vec_gt) != len(attribute_names):
                    if len(vec_gt) > len(attribute_names):
                        vec_gt = vec_gt[:len(attribute_names)]
                    else:
                        vec_gt = vec_gt + [np.nan] * (len(attribute_names) - len(vec_gt))
                
                if len(vec_sel) != len(attribute_names):
                    if len(vec_sel) > len(attribute_names):
                        vec_sel = vec_sel[:len(attribute_names)]
                    else:
                        vec_sel = vec_sel + [np.nan] * (len(attribute_names) - len(vec_sel))

                # Map vector elements to their attribute columns (gt and sel)
                row.update({f"{name}_gt": val for name, val in zip(attribute_names, vec_gt)})
                row.update({f"{name}_sel": val for name, val in zip(attribute_names, vec_sel)})

            buffer.append(row)
            cont += 1

        # Flush periodically
        if len(buffer) >= batch_size_save:
            df_chunk = pd.DataFrame(buffer, columns=columns)
            df_chunk.to_csv(csv_path, mode="a", header=False, index=False)
            print(f"Saved batch ending at index {cont}")
            buffer = []

    # Final flush
    if buffer:
        df_chunk = pd.DataFrame(buffer, columns=columns)
        df_chunk.to_csv(csv_path, mode="a", header=False, index=False)
        print(f"Saved final batch (less than {batch_size_save} items)")
    
    return


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (takes precedence over CLI args)",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--testing",
        default=False,
        action="store_true",
        help=(
            "If only inference"
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--num_final_validation_images",
        type=int,
        default=500,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--path_syn",
        type=str,
        default=None,
        help="The directory where syn images should be generated.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--transform",
        default=False,
        action="store_true",
        help=(
            "If apply resolution or not"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--name_exp",
        type=str,
        default="name_exp",
        help=("Exp name."),
    )
    parser.add_argument(
        "--attribute_policy",
        type=str,
        default="identity",
        help="Attribute editing policy (e.g. 'identity', 'top_k:5', 'threshold:0.5', 'random_k:10', 'flip', 'keep:0,2,5', 'set:1,3,4').",
    )
    parser.add_argument(
        "--prompt_dataset",
        type=str,
        default="PA100k",
        help="Dataset to use for prompt generation (e.g. 'PA100k', 'PETA', 'RAPv1', 'RAPv2', 'PETAzs', 'RAPzs').",
    )
    parser.add_argument("--pathDataset", help="Override dataset path if supported by class constructor")
    parser.add_argument("--path-dataset", help="Custom path to dataset images (new parameter, recommended over pathDataset)")
    parser.add_argument("--path-gt", help="Custom path to ground truth pickle file")
    parser.add_argument("--path-gt-img", help="Custom path to ground truth images folder")
    parser.add_argument(
        "--prompt_format_type",
        type=str,
        default="fixed-rule",
        help="The prompt generation formatting type (e.g. 'fixed-rule'). Used to select the PromptGenerator implementation.",
    )
    parser.add_argument(
        "--batch_size_memory",
        type=int,
        default=5000,
        help="Batch size to avoid RAM exhaustion during generation (number of images to process before writing/saving).",
    )
    parser.add_argument(
        "--path_dataset",
        type=str,
        default=None,
        help="Custom path to dataset images (overrides default paths)",
    )
    parser.add_argument(
        "--path_gt",
        type=str,
        default=None,
        help="Custom path to ground truth pickle file",
    )
    parser.add_argument(
        "--path_gt_img",
        type=str,
        default=None,
        help="Custom path to ground truth images folder",
    )
    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config:
        if yaml is None:
            raise ImportError("PyYAML is required to load config files. Install with: pip install pyyaml")
        
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # Extract sections from config
        model_config = config.get("model", {}) or {}
        generation_config = config.get("generation", {}) or {}
        output_config = config.get("output", {}) or {}
        dataset_config = config.get("dataset") or {}
        
        # Apply config values to args - use same names as CLI arguments
        # Model configuration
        if args.pretrained_model_name_or_path is None:
            args.pretrained_model_name_or_path = model_config.get("pretrained_model_name_or_path")
        
        if args.revision is None:
            args.revision = model_config.get("revision")
        
        if args.variant is None:
            args.variant = model_config.get("variant")
        
        # Generation paths
        if args.path_syn is None:
            args.path_syn = generation_config.get("path_syn")
        
        if args.output_dir == "sd-model-finetuned-lora":
            args.output_dir = generation_config.get("output_dir", "sd-model-finetuned-lora")
        
        # Dataset configuration
        if args.dataset_name is None:
            args.dataset_name = generation_config.get("dataset_name")
        
        if args.train_data_dir is None:
            args.train_data_dir = generation_config.get("train_data_dir")
        
        if args.dataset_config_name is None:
            args.dataset_config_name = generation_config.get("dataset_config_name")
        
        # Image parameters
        if args.height == 512:
            args.height = generation_config.get("height", 512)
        
        if args.width == 512:
            args.width = generation_config.get("width", 512)
        
        if not args.transform:
            args.transform = generation_config.get("transform", False)
        
        if not args.center_crop:
            args.center_crop = generation_config.get("center_crop", False)
        
        if not args.random_flip:
            args.random_flip = generation_config.get("random_flip", False)
        
        # Batch size for memory management
        if args.batch_size_memory == 5000:
            args.batch_size_memory = generation_config.get("batch_size_memory", 5000)
        
        # Attribute editing and prompt generation
        if args.attribute_policy == "identity":
            args.attribute_policy = generation_config.get("attribute_policy", "identity")
        
        if args.prompt_dataset == "PA100k":
            args.prompt_dataset = generation_config.get("prompt_dataset", "PA100k")
        
        if args.prompt_format_type == "fixed-rule":
            args.prompt_format_type = generation_config.get("prompt_format_type", "fixed-rule")
        
        # Column names
        if args.image_column == "image":
            args.image_column = generation_config.get("image_column", "image")
        
        if args.caption_column == "text":
            args.caption_column = generation_config.get("caption_column", "text")
        
        # Validation/inference parameters
        if args.validation_prompt is None:
            args.validation_prompt = generation_config.get("validation_prompt")
        
        if args.num_validation_images == 4:
            args.num_validation_images = generation_config.get("num_validation_images", 4)
        
        if args.num_final_validation_images == 500:
            args.num_final_validation_images = generation_config.get("num_final_validation_images", 500)
        
        if args.validation_epochs == 1:
            args.validation_epochs = generation_config.get("validation_epochs", 1)
        
        if args.max_train_samples is None:
            args.max_train_samples = generation_config.get("max_train_samples")
        
        if args.cache_dir is None:
            args.cache_dir = generation_config.get("cache_dir")
        
        if args.seed is None:
            args.seed = generation_config.get("seed")
        
        # Learning rate scheduler
        if args.lr_scheduler == "constant":
            args.lr_scheduler = generation_config.get("lr_scheduler", "constant")
        
        if args.lr_warmup_steps == 500:
            args.lr_warmup_steps = generation_config.get("lr_warmup_steps", 500)
        
        if args.snr_gamma is None:
            args.snr_gamma = generation_config.get("snr_gamma")
        
        # Optimizer parameters
        if args.adam_beta1 == 0.9:
            args.adam_beta1 = float(generation_config.get("adam_beta1", 0.9))
        
        if args.adam_beta2 == 0.999:
            args.adam_beta2 = float(generation_config.get("adam_beta2", 0.999))
        
        if args.adam_weight_decay == 1e-2:
            args.adam_weight_decay = float(generation_config.get("adam_weight_decay", 1e-2))
        
        if args.adam_epsilon == 1e-08:
            args.adam_epsilon = float(generation_config.get("adam_epsilon", 1e-08))
        
        if args.max_grad_norm == 1.0:
            args.max_grad_norm = float(generation_config.get("max_grad_norm", 1.0))
        
        # Other parameters
        if not args.use_8bit_adam:
            args.use_8bit_adam = generation_config.get("use_8bit_adam", False)
        
        if not args.allow_tf32:
            args.allow_tf32 = generation_config.get("allow_tf32", False)
        
        if args.dataloader_num_workers == 0:
            args.dataloader_num_workers = generation_config.get("dataloader_num_workers", 0)
        
        if not args.scale_lr:
            args.scale_lr = generation_config.get("scale_lr", False)
        
        if not args.enable_xformers_memory_efficient_attention:
            args.enable_xformers_memory_efficient_attention = generation_config.get("enable_xformers_memory_efficient_attention", False)
        
        if args.noise_offset == 0:
            args.noise_offset = float(generation_config.get("noise_offset", 0))
        
        if not args.gradient_checkpointing:
            args.gradient_checkpointing = generation_config.get("gradient_checkpointing", False)
        
        if args.gradient_accumulation_steps == 1:
            args.gradient_accumulation_steps = generation_config.get("gradient_accumulation_steps", 1)
        
        # LoRA rank
        if args.rank == 4:
            args.rank = generation_config.get("rank", 4)
        
        # Experiment name
        if args.name_exp == "name_exp":
            args.name_exp = config.get("name", generation_config.get("name_exp", "name_exp"))
        
        # Flags
        if not args.testing:
            args.testing = config.get("testing", False)
        
        # Output configuration
        if args.logging_dir == "logs":
            args.logging_dir = output_config.get("logging_dir", "logs")
        
        if args.mixed_precision is None:
            args.mixed_precision = output_config.get("mixed_precision")
        
        if args.report_to == "tensorboard":
            args.report_to = output_config.get("report_to", "tensorboard")
        
        if not args.push_to_hub:
            args.push_to_hub = output_config.get("push_to_hub", False)
        
        if args.hub_model_id is None:
            args.hub_model_id = output_config.get("hub_model_id")
        
        if args.hub_token is None:
            args.hub_token = output_config.get("hub_token")
        
        if args.prediction_type is None:
            args.prediction_type = output_config.get("prediction_type")
        
        if args.checkpointing_steps == 500:
            args.checkpointing_steps = generation_config.get("checkpointing_steps", 500)
        
        if args.checkpoints_total_limit is None:
            args.checkpoints_total_limit = generation_config.get("checkpoints_total_limit")
        
        if args.resume_from_checkpoint is None:
            args.resume_from_checkpoint = generation_config.get("resume_from_checkpoint")
        
        # Dataset custom paths
        if args.path_dataset is None:
            args.path_dataset = dataset_config.get("path_dataset")
        
        if args.path_gt is None:
            args.path_gt = dataset_config.get("path_gt")
        
        if args.path_gt_img is None:
            args.path_gt_img = dataset_config.get("path_gt_img")

        # Optional dataset path overrides
        if not hasattr(args, "path_dataset") or args.path_dataset is None:
            args.path_dataset = config.get("path_dataset")
        if not hasattr(args, "path_gt") or args.path_gt is None:
            args.path_gt = config.get("path_gt")
        if not hasattr(args, "path_gt_img") or args.path_gt_img is None:
            args.path_gt_img = config.get("path_gt_img")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")
    
    if args.pretrained_model_name_or_path is None:
        raise ValueError("Need a pretrained model name or path. Provide via --config YAML or --pretrained_model_name_or_path CLI arg.")

    return args


DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
    "RAPzs": ("image", "text"),
    "PETAzs": ("image", "text"),
    "PETA": ("image", "text"),
    "PA100k": ("image", "text"),
    "RAPv1": ("image", "text"),
    "RAPv2": ("image", "text"),
}

DATASETS = {
    "RAPzs": "/mnt/rhome/paa/pedestrian/datasetForFID/RAPzs/train/",
    "PA100k": "/mnt/rhome/paa/pedestrian/datasetForFID/PA100k/train/",
    "PETAzs": "/mnt/rhome/paa/pedestrian/datasetForFID/PETAzs/train/",
    "RAPv1": "/mnt/rhome/paa/pedestrian/datasetForFID/RAPv1/trainval/",
    "RAPv2": "/mnt/rhome/paa/pedestrian/datasetForFID/RAPv2/train/",
    "PETA": "/mnt/rhome/paa/pedestrian/datasetForFID/PETA/train/",
}

# Mapping from dataset names to dataset classes
DATASET_CLASSES = {
    "RAPzs": RAPzsDatasetAll,
    "PA100k": PA100kDatasetAll,
    "PETAzs": PETAzsDatasetAll,
    "RAPv1": RAPv1DatasetAll,
    "RAPv2": RAPv2DatasetAll,
    "PETA": PETADatasetAll,
}



if __name__ == "__main__":
    args = parse_args()
    
    # Initialize AttributeEditing and PromptGenerator
    attribute_editor = AttributeEditing(policy=args.attribute_policy, dataset=args.prompt_dataset)
    prompt_generator = PromptGenerator(type=args.prompt_format_type, dataset=args.prompt_dataset)
    
    # Load previous pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=args.mixed_precision,
        safety_checker=None,
    )

    # load attention processors
    pipeline.load_lora_weights(args.output_dir)
    pipeline.to("cuda")
    
    generator = torch.Generator("cuda")
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    # Determine dataset path: use path_dataset override if provided, otherwise use default DATASETS mapping
    if args.path_dataset:
        dataset_path = args.path_dataset
    else:
        dataset_path = DATASETS[args.dataset_name]
    
    data_files = {}
    data_files["train"] = os.path.join(dataset_path, "**")
    
    # Load dataset with imagefolder - this should automatically include all fields from metadata.jsonl
    dataset = load_dataset(
        "imagefolder",
        data_files=data_files,
        cache_dir=args.cache_dir,
    )
    
    # Debug: Check what columns/features the dataset has
    print(f"DEBUG: Dataset features: {dataset['train'].features}")
    print(f"DEBUG: Dataset column names: {dataset['train'].column_names}")
    if len(dataset['train']) > 0:
        print(f"DEBUG: First item keys: {dataset['train'][0].keys()}")
        if 'vector' in dataset['train'][0]:
            print(f"DEBUG: First item has vector with length: {len(dataset['train'][0]['vector'])}")
        else:
            print(f"DEBUG: WARNING - 'vector' field not found in dataset items!")
            print(f"DEBUG: Available fields: {list(dataset['train'][0].keys())}")

    # run inference with attribute editing and prompt generation
    seed_value = args.seed if args.seed is not None else None
    
    generateFinalVal(args, dataset, pipeline, generator, attribute_editor, prompt_generator, seed_value)

