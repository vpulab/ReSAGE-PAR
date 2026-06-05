# 1. Modelos de HuggingFace (se descargan solos)
google_bert_path = './modelsweights/bert-base-uncased'
vicuna_7b_path = 'lmsys/vicuna-7b-v1.1'

# 2. Pesos locales que acabas de descargar mediante wget
minigpt4_path = './modelsweights/pretrained_minigpt4_7b.pth'
blip2_path = './modelsweights/blip2_pretrained_flant5xxl.pth'
eva_vit_g_path = './modelsweights/eva_vit_g.pth'

import os

def get_pkl_rootpath(dataset):
    if dataset=="RAPv1":
        root_path='$REAL_DATA/RAP/RAP_dataset/'
        pkl_path='./PAR_Template_Scripts/rap1_template.pkl'
    elif dataset=="RAPv2":  
        root_path='$REAL_DATA/RAP2/RAP_dataset/'
        pkl_path='./PAR_Template_Scripts/rap2_template.pkl'
    elif dataset=="PETA": 
        root_path='$REAL_DATA/PETA/images/'
        pkl_path='./PAR_Template_Scripts/peta_template.pkl'
    elif dataset=="PA100k": 
        root_path='$REAL_DATA/PA100k/data/'
        pkl_path='./PAR_Template_Scripts/pa100k_template.pkl'
    elif dataset=="RAPzs": 
        root_path='$REAL_DATA/RAP2/RAP_dataset/'
        pkl_path='./PAR_Template_Scripts/rapzs_template.pkl'
    elif dataset=="PETAzs":
        root_path='$REAL_DATA/PETA/images'
        pkl_path='./PAR_Template_Scripts/petazs_template.pkl'
    elif dataset=="MSP":
        root_path='$REAL_DATA/MSP/images'
        pkl_path='./PAR_Template_Scripts/MSP/msp_random_template.pkl'
    elif dataset=="MSPCD":
        root_path='$REAL_DATA/MSP/images'
        pkl_path='./PAR_Template_Scripts/MSP/msp_cd_template.pkl'
    
    
    #pkl_path = os.path.expandvars(pkl_path)
    root_path = os.path.expandvars(root_path)
    return pkl_path, root_path
