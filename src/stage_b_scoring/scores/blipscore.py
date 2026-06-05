from PIL import Image
import requests

import torch

from transformers import BlipProcessor, BlipForQuestionAnswering, BlipForImageTextRetrieval

from transformers import PreTrainedTokenizerFast




modelNameBLIPScore="Salesforce/blip-itm-base-coco"
modelBlipBLIPScore=None
preprocessBlipBLIPScore=None
device=None


import torch.nn.functional as F

# for BLIPV2SCore
def setModelBlipScore():
    global modelBlipBLIPScore, modelNameBLIPScore, preprocessBlipBLIPScore, device
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocessBlipBLIPScore = BlipProcessor.from_pretrained(modelNameBLIPScore)
    
    modelBlipBLIPScore = BlipForImageTextRetrieval.from_pretrained(modelNameBLIPScore)
    modelBlipBLIPScore.to(device)
    return

def getBlipScore(prompt, imgPath):
    global modelBlipBLIPScore, preprocessBlipBLIPScore, device
    
    image = Image.open(imgPath).convert("RGB")
    
    # Generate inputs on CPU first
    inputs = preprocessBlipBLIPScore(images=image, text=prompt, return_tensors="pt")
    
    # Move to device and convert to float16
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = modelBlipBLIPScore(**inputs)
        score = output.itm_score

    match_prob = score[:, 1].item()
    return match_prob
