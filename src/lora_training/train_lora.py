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

# Add parent directory to sys.path to enable imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from customDatasets.datasetRAPv1All import RAPv1DatasetAll
from customDatasets.datasetRAPv2All import RAPv2DatasetAll
from customDatasets.datasetRAPzsAll import RAPzsDatasetAll
from customDatasets.datasetPETAzsAll import PETAzsDatasetAll
from customDatasets.datasetPETAAll import PETADatasetAll
from customDatasets.datasetPA100kAll import PA100kDatasetAll



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

def save_model_card(
    repo_id: str,
    images: list = None,
    base_model: str = None,
    dataset_name: str = None,
    repo_folder: str = None,
):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "lora",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))



from io import BytesIO
import base64

def image_to_base64(img):
    """Convert PIL image to base64 string for HTML logging."""
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def log_prompt_table(args, prompts, cond_images, gen_images, table_name="table_default"):
    """
    Log a prompt table as an MLflow artifact (HTML table).
    :param prompts: list of text prompts
    :param cond_images: list of PIL conditioning images
    :param gen_images: list of PIL generated images
    """
    # Build HTML table
    rows = []
    for prompt, cond_img, gen_img in zip(prompts, cond_images, gen_images):
        cond_b64 = image_to_base64(cond_img)
        gen_b64 = image_to_base64(gen_img)
        row = f"""
        <tr>
            <td>{prompt}</td>
            <td><img src="data:image/png;base64,{cond_b64}" width="128"/></td>
            <td><img src="data:image/png;base64,{gen_b64}" width="128"/></td>
        </tr>
        """
        rows.append(row)

    table_html = f"""
    <html><body>
    <table border="1" style="border-collapse: collapse;">
        <tr><th>Prompt</th><th>Conditioning Image</th><th>Generated Image</th></tr>
        {''.join(rows)}
    </table>
    </body></html>
    """
    table_path = f"{args.output_dir}/{table_name}.html"
    # Save HTML file
    with open(table_path, "w") as f:
        f.write(table_html)
    # Log as MLflow artifact (if enabled)
    if args.use_mlflow:
        mlflow.log_artifact(table_path, artifact_path="tables")


def log_validation(
    pipeline,
    args,
    accelerator,
    epoch,
    samples_cond,
    samples_cond_final_val,
    is_final_validation=False,
    
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)


    
    prompts=[]
    imgsCond=[]
    for data in samples_cond:
        prompt=data['text']
        img_cond=data['image']
        imgsCond.append(img_cond.resize((args.width, args.height)))
        prompts.append(prompt)
        genImg=pipeline(prompt, image=img_cond, num_inference_steps=30, generator=generator, height=args.height, width=args.width).images[0]
        print(genImg.height)
        print(genImg.width)
        images.append(genImg)
        
    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            dataWandb=[]
            for prompt, imgCond, genImg in zip(prompts, imgsCond, images):
                dataWandb.append([prompt, wandb.Image(img_cond, caption="Condition"), wandb.Image(genImg, caption="Generated")])
            tracker.log_table(table_name="results_{}".format(epoch), columns=["caption", "condition", "generated"], data=dataWandb)
    
    table_name="table_step_{}".format(epoch)
    log_prompt_table(args, prompts, imgsCond, images, table_name)
    

    if is_final_validation:
        generateFinalVal(args, samples_cond_final_val, pipeline, generator)

    return images

import pandas as pd

def generateFinalVal(args, samples_cond_final_val, pipeline, generator):
    print("Final validation")
    patToSaveImgsCond=args.output_dir+"/condImgs/"
    patToSaveImgsGen=args.output_dir+"/generatedImgs/"
    prompts=[]
    imgsCond=[]
    os.mkdir(patToSaveImgsCond)
    os.mkdir(patToSaveImgsGen)
    columns=['condImg', 'genImg', 'prompt']
    datas=[]
    cont=0
    for data in tqdm(samples_cond_final_val):
        prompt=data['text']
        img_cond=data['image']
        img_cond=img_cond.resize((args.width, args.height))
        
        
        genImg=pipeline(prompt, image=img_cond, num_inference_steps=30, generator=generator, height=args.height, width=args.width).images[0]
        
        img_cond.save(patToSaveImgsCond+'img-{}.png'.format(cont))
        genImg.save(patToSaveImgsGen+'img-{}.png'.format(cont))
        datas.append(['img-{}.png'.format(cont), 'img-{}.png'.format(cont), prompt])
        cont+=1
        #break
    df = pd.DataFrame(data=datas, columns=columns)
    df.to_csv(args.output_dir+"/generated.csv")


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
        "--use_mlflow",
        action="store_true",
        default=False,
        help="Whether to use MLflow for experiment tracking. If not set, training runs without MLflow logging.",
    )

    args = parser.parse_args()
    
    # Load config from YAML if provided
    if args.config:
        if yaml is None:
            raise ImportError("PyYAML is required to load config files. Install with: pip install pyyaml")
        
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # Extract sections from config
        train_config = config.get("training", {})
        model_config = config.get("model", {})
        output_config = config.get("output", {})
        
        # Apply config values to args - use same names as CLI arguments
        # Model configuration
        if args.pretrained_model_name_or_path is None:
            args.pretrained_model_name_or_path = model_config.get("pretrained_model_name_or_path") or model_config.get("base_model")
        
        if args.revision is None:
            args.revision = model_config.get("revision")
        
        if args.variant is None:
            args.variant = model_config.get("variant")
        
        # Dataset configuration
        if args.dataset_name is None and args.train_data_dir is None:
            args.dataset_name = train_config.get("dataset_name")
            args.train_data_dir = train_config.get("train_data_dir")
        
        if args.dataset_config_name is None:
            args.dataset_config_name = train_config.get("dataset_config_name")
        
        if args.image_column is None or args.image_column == "image":
            args.image_column = train_config.get("image_column", "image")
        
        if args.caption_column is None or args.caption_column == "text":
            args.caption_column = train_config.get("caption_column", "text")
        
        # Validation configuration
        if args.validation_prompt is None:
            args.validation_prompt = train_config.get("validation_prompt")
        
        if args.num_validation_images == 4:
            args.num_validation_images = train_config.get("num_validation_images", 4)
        
        if args.num_final_validation_images == 500:
            args.num_final_validation_images = train_config.get("num_final_validation_images", 500)
        
        if args.validation_epochs == 1:
            args.validation_epochs = train_config.get("validation_epochs", 1)
        
        if args.max_train_samples is None:
            args.max_train_samples = train_config.get("max_train_samples")
        
        # Output configuration
        if args.output_dir == "sd-model-finetuned-lora":
            args.output_dir = output_config.get("output_dir", "sd-model-finetuned-lora")
        
        if args.cache_dir is None:
            args.cache_dir = train_config.get("cache_dir")
        
        # Seed
        if args.seed is None:
            args.seed = train_config.get("seed")
        
        # Resolution and dimensions
        if args.height == 512:
            args.height = train_config.get("height", 512)
        
        if args.width == 512:
            args.width = train_config.get("width", 512)
        
        if not args.transform:
            args.transform = train_config.get("transform", False)
        
        if not args.center_crop:
            args.center_crop = train_config.get("center_crop", False)
        
        if not args.random_flip:
            args.random_flip = train_config.get("random_flip", False)
        
        # Training parameters
        if args.train_batch_size == 16:
            args.train_batch_size = train_config.get("train_batch_size", 16)
        
        if args.num_train_epochs == 100:
            args.num_train_epochs = train_config.get("num_train_epochs", 100)
        
        if args.max_train_steps is None:
            args.max_train_steps = train_config.get("max_train_steps")
        
        if args.gradient_accumulation_steps == 1:
            args.gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
        
        if not args.gradient_checkpointing:
            args.gradient_checkpointing = train_config.get("gradient_checkpointing", False)
        
        if args.learning_rate == 1e-4:
            args.learning_rate = float(train_config.get("learning_rate", 1e-4))
        
        if not args.scale_lr:
            args.scale_lr = train_config.get("scale_lr", False)
        
        # Learning rate scheduler
        if args.lr_scheduler == "constant":
            args.lr_scheduler = train_config.get("lr_scheduler", "constant")
        
        if args.lr_warmup_steps == 500:
            args.lr_warmup_steps = train_config.get("lr_warmup_steps", 500)
        
        if args.snr_gamma is None:
            args.snr_gamma = train_config.get("snr_gamma")
        
        # Optimizer parameters
        if not args.use_8bit_adam:
            args.use_8bit_adam = train_config.get("use_8bit_adam", False)
        
        if not args.allow_tf32:
            args.allow_tf32 = train_config.get("allow_tf32", False)
        
        if args.dataloader_num_workers == 0:
            args.dataloader_num_workers = train_config.get("dataloader_num_workers", 0)
        
        if args.adam_beta1 == 0.9:
            args.adam_beta1 = float(train_config.get("adam_beta1", 0.9))
        
        if args.adam_beta2 == 0.999:
            args.adam_beta2 = float(train_config.get("adam_beta2", 0.999))
        
        if args.adam_weight_decay == 1e-2:
            args.adam_weight_decay = float(train_config.get("adam_weight_decay", 1e-2))
        
        if args.adam_epsilon == 1e-08:
            args.adam_epsilon = float(train_config.get("adam_epsilon", 1e-08))
        
        if args.max_grad_norm == 1.0:
            args.max_grad_norm = float(train_config.get("max_grad_norm", 1.0))
        
        # Hub configuration
        if not args.push_to_hub:
            args.push_to_hub = output_config.get("push_to_hub", False)
        
        if args.hub_token is None:
            args.hub_token = output_config.get("hub_token")
        
        if args.prediction_type is None:
            args.prediction_type = train_config.get("prediction_type")
        
        if args.hub_model_id is None:
            args.hub_model_id = output_config.get("hub_model_id")
        
        if args.logging_dir == "logs":
            args.logging_dir = output_config.get("logging_dir", "logs")
        
        if args.mixed_precision is None:
            args.mixed_precision = output_config.get("mixed_precision")
        
        if args.report_to == "tensorboard":
            args.report_to = output_config.get("report_to", "tensorboard")
        
        # Checkpointing
        if args.checkpointing_steps == 500:
            args.checkpointing_steps = train_config.get("checkpointing_steps", 500)
        
        if args.checkpoints_total_limit is None:
            args.checkpoints_total_limit = train_config.get("checkpoints_total_limit")
        
        if args.resume_from_checkpoint is None:
            args.resume_from_checkpoint = train_config.get("resume_from_checkpoint")
        
        # xformers
        if not args.enable_xformers_memory_efficient_attention:
            args.enable_xformers_memory_efficient_attention = train_config.get("enable_xformers_memory_efficient_attention", False)
        
        if args.noise_offset == 0:
            args.noise_offset = float(train_config.get("noise_offset", 0))
        
        # LoRA rank
        if args.rank == 4:
            args.rank = train_config.get("rank", 4)
        
        # Experiment name
        if args.name_exp == "name_exp":
            args.name_exp = config.get("name", train_config.get("name_exp", "name_exp"))
        
        # MLflow
        if not args.use_mlflow:
            args.use_mlflow = output_config.get("use_mlflow", False)
    
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
    "RAPzs": "/mnt/rhome/paa/pedestrian/datasetForFID/RAPzs/train_withmetadata/",
    "PA100k": "/mnt/rhome/paa/pedestrian/datasetForFID/PA100k/train_withmetadata/",
    "PETAzs": "/mnt/rhome/paa/pedestrian/datasetForFID/PETAzs/train_withmetadata/",
    "RAPv1": "/mnt/rhome/paa/pedestrian/datasetForFID/RAPv1/trainval_withmetadata/",
    "RAPv2": "/mnt/rhome/paa/pedestrian/datasetForFID/RAPv2/train_withmetadata/",
    "PETA": "/mnt/rhome/paa/pedestrian/datasetForFID/PETA/train_withmetadata/",
}


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )


    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        log_with=args.report_to
    )

    if accelerator.is_main_process:
        if args.report_to == "wandb":
            wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="pablo-ayuso-universidad-autonoma-de-madrid",
                # Set the wandb project where this run will be logged.
                project="lora_testing",
                name=args.name_exp,
                config={
                    "model": args.pretrained_model_name_or_path,
                    "epochs": args.num_train_epochs,
                    "batch_size": args.train_batch_size,
                    "resolution": args.resolution,
                    "height": args.height,
                    "width": args.width,
                    "transform": args.transform,
                    "dataset": args.dataset_name,
                    "learning_rate": args.learning_rate,
                    "rank": args.rank
                }
                )
        
        if args.use_mlflow:
            descriptionForMLFlow="lora_model_{}_epochs_{}_batch_size{}_resolution_{}_transform_{}_dataset_{}_learningrate_{}_rank_{}".format( 
                args.pretrained_model_name_or_path,args.num_train_epochs,args.train_batch_size,args.resolution,args.transform,args.dataset_name,args.learning_rate,args.rank)
            print("set exp")
            mlflow.set_experiment(args.name_exp)
            print("start_run")
            mlflow.start_run(run_name=descriptionForMLFlow)
            params = {
                    "model": args.pretrained_model_name_or_path,
                    "epochs": args.num_train_epochs,
                    "batch_size": args.train_batch_size,
                    "resolution": args.resolution,
                    "height": args.height,
                    "width": args.width,
                    "transform": args.transform,
                    "dataset": args.dataset_name,
                    "learning_rate": args.learning_rate,
                    "rank": args.rank
            }
            # Log training parameters.
            mlflow.log_params(params)
        if not os.path.isfile(args.output_dir):
            os.mkdir(args.output_dir)

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Freeze the unet parameters before adding adapters
    for param in unet.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        if args.dataset_name in DATASETS.keys():
            data_files = {}
            
            data_files["train"] = os.path.join(DATASETS[args.dataset_name], "**")
            
            print(data_files["train"])
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=args.cache_dir,
            )
            print(dataset["train"])
        else:
        
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                data_dir=args.train_data_dir,
            )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if args.dataset_name in DATASETS.keys():
        column_names = DATASET_NAME_MAPPING[args.dataset_name]
    else:
        column_names = dataset["train"].column_names
    
    print("Training with "+args.dataset_name+" dataset")
    #print(dataset["train"]['text'])
    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def hook_fn(module, input, output):
        print("Hook output shape:", output.shape)    

    # Preprocessing the datasets.
    
    if args.transform:
        train_transforms = transforms.Compose(
            [
                # (h, w),
                transforms.Resize((args.height,args.width), interpolation=transforms.InterpolationMode.BILINEAR),
                #transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    else:
        train_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        # 2. Create a list of dicts from your samples
        data_list = []
  

        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    layer_to_hook = unet.down_blocks[-1].resnets[-1]
    
    #hook_handle = layer_to_hook.register_forward_hook(hook_fn)
    b_print=True
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                img=batch["pixel_values"].to(dtype=weight_dtype)
                if b_print:
                    print(img.shape)
                    
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                #print(latents.shape)



                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if b_print:
                    print(noisy_latents.shape)
                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if b_print:
                    print(model_pred.shape)
                    b_print=False

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                      

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                if args.use_mlflow:
                    mlflow.log_metric("train_loss", f"{train_loss}", step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:

                    if accelerator.is_main_process:

                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        unwrapped_unet = unwrap_model(unet)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )

                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:

            if epoch % args.validation_epochs == 0:

                # create pipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                    safety_checker=None,
                )
                
                indices = random.sample(range(len(dataset['train'])), args.num_validation_images)
                sampled_items = [dataset['train'][i] for i in indices]

                #indices = range(len(dataset['train']))
                #sampled_dataset=[dataset['train'][i] for i in range(len(dataset['train']))]
                sampled_dataset=None
                images = log_validation(pipeline, args, accelerator, epoch, sampled_items, sampled_dataset, is_final_validation=False)

                del pipeline
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

        # Final inference
        # Load previous pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
            safety_checker=None,
        )

        # load attention processors
        pipeline.load_lora_weights(args.output_dir)
        indices = random.sample(range(len(dataset['train'])), args.num_validation_images)
        sampled_items = [dataset['train'][i] for i in indices]
        # run inference
        sampled_dataset=[dataset['train'][i] for i in range(len(dataset['train']))]
        images = log_validation(pipeline, args, accelerator, epoch, sampled_items, sampled_dataset, is_final_validation=False)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                dataset_name=args.dataset_name,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
