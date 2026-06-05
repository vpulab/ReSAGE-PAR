
from torch.utils.data import Dataset


from typing import Any, Tuple, Dict

import numpy as np
import torchvision.transforms as transforms
import torch
import os

from PIL import Image
from collections import defaultdict
import random
import cv2 as cv
import pickle

import numpy as np
from typing import List
# adapted from https://github.com/kkyuhun94/dalda

TRAIN_FOLDER ="train/"

class PETADatasetAll(Dataset):

    listStartingPrompt=['a', 'there is a']

    listGenderAtDS=['personalMale', 'personalFemale']
    listGenderPrompt=['man', 'woman']

    listUpperBodyAtDS= ['upperBodyCasual','upperBodyFormal','upperBodyJacket','upperBodyLogo', 'upperBodyPlaid',
                    'upperBodyThinStripes',  'upperBodyTshirt', 'upperBodyOther',
                    'upperBodyVNeck']
    listUpperBodyPrompt =['casual', 'formal', 'jacket', 'logo', 'plaid', 'thin stripes', 't shirt', 'other', 'vneck']

    listUpperBodyColorAtDS = ['upperBodyBlack', 'upperBodyBlue', 'upperBodyBrown', 'upperBodyGreen', 'upperBodyGrey', 
                    'upperBodyOrange', 'upperBodyPink', 'upperBodyPurple', 'upperBodyRed', 
                    'upperBodyWhite', 'upperBodyYellow']
    listUpperBodyColorPrompt =['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

    listLowerBodyAtDS=['lowerBodyCasual', 'lowerBodyFormal', 'lowerBodyJeans', 'lowerBodyShorts', 'lowerBodyShortSkirt', 'lowerBodyTrousers', 
                       'lowerBodyCapri', 'lowerBodyHotPants', 'lowerBodyLongSkirt', 'lowerBodyPlaid', 'lowerBodyThinStripes', 'lowerBodySuits']
    listLowerBodyAtPrompt=['casual', 'formal', 'jeans', 'shorts', 'shortskirt', 'trousers', 'capri', 'hotpants', 'long skirt', 'plaid', 'thin stripes', 'suits']
    
    listLowerBodyColorAtDS=['lowerBodyBlack', 'lowerBodyBlue', 
                            'lowerBodyBrown', 'lowerBodyGreen', 'lowerBodyGrey', 'lowerBodyOrange', 
                            'lowerBodyPink', 'lowerBodyPurple', 'lowerBodyRed', 'lowerBodyWhite', 
                            'lowerBodyYellow']
    listLowerBodyColorAtPrompt=['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

    listCarryingAtDS=['carryingBackpack', 'carryingOther', 'carryingMessengerBag', 'carryingNothing', 'carryingPlasticBags', 'carryingBabyBuggy',
                      'carryingShoppingTro', 'carryingUmbrella', 'carryingFolder', 'carryingLuggageCase', 'carryingSuitcase']
    listCarryingAtPrompt=['backpack', 'other', 'messenger bag', 'nothing', 'plastic bags', 'baby buggy', 'shopping tro', 'umbrella', 'folder', 'luggage case', 'suit case']

    listAccesoryAtDS=['accessoryHat','accessoryMuffler','accessoryNothing','accessorySunglasses','accessoryHeadphone','accessoryHairBand','accessoryKerchief']
    listAccesoryAtPrompt=['hat', 'muffler', 'nothing', 'sunglasses', 'headphone', 'hairband', 'kerchief']

    listHairAtDS=['hairLong','hairBald','hairShort']
    listHairAtPrompt=['long hair', 'bald', 'short hair']

    listHairColorAtDS=['hairBlack', 'hairBlue', 'hairBrown', 'hairGreen', 'hairGrey', 
                       'hairOrange', 'hairPink', 'hairPurple', 'hairRed', 'hairWhite', 'hairYellow']
    listHairColorAtPrompt=['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red',
                           'white', 'yellow']

    listFootwearAtDS=['footwearLeatherShoes', 'footwearSandals', 'footwearShoes','footwearSneaker','footwearStocking']
    listFootwearAtPrompt=['leather shoes', 'sandals', 'shoes', 'sneakers', 'stocking']

    listFootwearColorAtDS=['footwearBlack', 'footwearBlue', 'footwearBrown', 'footwearGreen', 
                            'footwearGrey', 'footwearOrange', 'footwearPink', 'footwearPurple', 'footwearRed',
                            'footwearWhite', 'footwearYellow']
    listFootwearColorAtPrompt=['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

    # listas para contar los attributos del prompt
    listOfListUsedToPromptAtDS=[listGenderAtDS, listUpperBodyAtDS, listUpperBodyColorAtDS, listLowerBodyAtDS, listLowerBodyColorAtDS, \
                                 listCarryingAtDS, listAccesoryAtDS,
                                listHairAtDS, listHairColorAtDS, listFootwearAtDS, listFootwearColorAtDS]
    listOfListUsedToPromptAtPrompt=[listGenderPrompt, listUpperBodyPrompt, listUpperBodyColorPrompt, listLowerBodyAtPrompt, listLowerBodyColorAtPrompt, \
                                     listCarryingAtPrompt, listAccesoryAtPrompt, \
                                    listHairAtPrompt, listHairAtPrompt, listFootwearAtPrompt, listFootwearColorAtPrompt]

    listUsedAllPromptAtDS = [attribute for lista in listOfListUsedToPromptAtDS for attribute in lista ]

    listAttributePrompt=['age less 30', 'age less 45', 'age less 60', 'age larger than 60',
                        'backpack', 'other',  'casual', 'casual', 'formal', 'formal',
                        'hat', 'jacket', 'jeans', 'leather shoes', 'logo', 'long hair',
                        'man', 'messenger bag', 'muffler', 'nothing', 'nothing', 'plaid',
                        'plastic bags', 'sandals', 'shoes', 'shorts', 'short sleeve',
                        'short skirt', 'sneakers', 'body thin stripes', 'sunglasses',
                        'trousers', 'tshirt', 'other', 'vneck', 
                        'black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow',
                        'black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow',
                        'black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow',
                        'black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow',
                        'headphone', 'age less 15', 'baby buggy', 'bald', 'boots', 'capri',
                        'shopping tro', 'umbrella', 'woman', 'folder', 'hairband', 'hotpants',
                        'kerchief', 'long skirt', 'long sleeve', 'plaid', 'thin stripes', 
                        'luggage case', 'no sleeve', 'short hair', 'stocking', 
                        'suit', 'suit case', 'suits', 'sweater', 'stripes thick'
                        ]

    listAttributesPETAzs = [
                            'personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60', 
                            'carryingBackpack', 'carryingOther', 'lowerBodyCasual', 'upperBodyCasual', 'lowerBodyFormal', 'upperBodyFormal', 
                            'accessoryHat', 'upperBodyJacket', 'lowerBodyJeans',  'footwearLeatherShoes', 'upperBodyLogo','hairLong',
                            'personalMale', 'carryingMessengerBag', 'accessoryMuffler', 'accessoryNothing', 'carryingNothing', 'upperBodyPlaid', 
                            'carryingPlasticBags', 'footwearSandals', 'footwearShoes', 'lowerBodyShorts', 'upperBodyShortSleeve',
                            'lowerBodyShortSkirt', 'footwearSneaker', 'upperBodyThinStripes', 'accessorySunglasses',  
                            'lowerBodyTrousers', 'upperBodyTshirt', 'upperBodyOther', 'upperBodyVNeck', 
                            'upperBodyBlack', 'upperBodyBlue', 'upperBodyBrown', 'upperBodyGreen', 'upperBodyGrey', 'upperBodyOrange', 'upperBodyPink', 'upperBodyPurple', 'upperBodyRed', 'upperBodyWhite', 'upperBodyYellow',
                            'lowerBodyBlack', 'lowerBodyBlue', 'lowerBodyBrown', 'lowerBodyGreen', 'lowerBodyGrey', 'lowerBodyOrange', 'lowerBodyPink', 'lowerBodyPurple', 'lowerBodyRed', 'lowerBodyWhite', 'lowerBodyYellow',
                            'hairBlack', 'hairBlue', 'hairBrown', 'hairGreen', 'hairGrey', 'hairOrange', 'hairPink', 'hairPurple', 'hairRed', 'hairWhite', 'hairYellow', 
                            'footwearBlack', 'footwearBlue', 'footwearBrown', 'footwearGreen', 'footwearGrey', 'footwearOrange', 'footwearPink', 'footwearPurple', 'footwearRed', 'footwearWhite', 'footwearYellow', 
                            'accessoryHeadphone', 'personalLess15', 'carryingBabyBuggy', 'hairBald', 'footwearBoots', 'lowerBodyCapri',
                            'carryingShoppingTro', 'carryingUmbrella', 'personalFemale', 'carryingFolder', 'accessoryHairBand', 'lowerBodyHotPants', 
                            'accessoryKerchief', 'lowerBodyLongSkirt', 'upperBodyLongSleeve', 'lowerBodyPlaid', 'lowerBodyThinStripes',
                            'carryingLuggageCase', 'upperBodyNoSleeve', 'hairShort', 'footwearStocking', 
                            'upperBodySuit', 'carryingSuitcase', 'lowerBodySuits', 'upperBodySweater','upperBodyThickStripes']
    listAttributes = listAttributesPETAzs

    attr_complements = {
        attr: [other for other in group if other != attr]
        for group in listOfListUsedToPromptAtDS
        for attr in group
    }

    complementary_attributes = {
        "age less 30": ["age less 45", "age less 60", "age larger than 60"],
        "age less 45": ["age less 30", "age less 60", "age larger than 60"],
        "age less 60": ["age less 45", "age less 30", "age larger than 60"],
        "age larger than 60": ["age less 60", "age less 45"],

        #unified two type of others
        "other": ["backpack", "other", "messenger bag", "nothing", "plastic bags", 
        "baby buggy", "shopping tro", "umbrella", "folder", "luggage case", "suit case",
        "casual", "formal", "jacket", "logo", "plaid", "thin stripes", "t shirt", "vneck"],


        "backpack": ["messenger bag", "suit case", "luggage case"],
        "messenger bag": ["backpack", "handbag", "plastic bags"],
        "suit case": ["backpack", "luggage case", "shopping tro"],
        "luggage case": ["suit case", "backpack", "shopping tro"],

        "casual": ["formal", "tshirt", "jeans"],
        "formal": ["casual", "suit", "suits"],

        "hat": ["capri", "kerchief", "hairband"],
        "jacket": ["sweater", "coat", "long sleeve"],
        "jeans": ["tshirt", "sneakers", "casual"],
        "leather shoes": ["suit", "suit case", "formal"],
        "logo": ["tshirt", "vneck"],
        "long hair": ["short hair", "hairband"],

        "man": ["woman"],
        "woman": ["man"],

        "muffler": ["sweater", "coat", "kerchief"],
        "nothing": ["no sleeve", "short sleeve"],
        "plaid": ["thin stripes", "body thin stripes", "stripes thick"],
        "body thin stripes": ["thin stripes", "stripes thick"],
        "thin stripes": ["body thin stripes", "stripes thick"],
        "stripes thick": ["thin stripes", "body thin stripes"],

        "plastic bags": ["shopping tro", "backpack"],
        "sandals": ["shorts", "short skirt"],
        "shoes": ["sandals", "boots"],
        "shorts": ["tshirt", "sandals", "casual"],
        "short sleeve": ["tshirt", "no sleeve", "vneck"],
        "short skirt": ["long skirt", "stocking", "sandals"],
        "sneakers": ["jeans", "casual", "tshirt"],

        "sunglasses": ["hat", "capri", "muffler"],

        "trousers": ["jeans", "casual", "suit"],
        "tshirt": ["jeans", "shorts", "casual"],
        "vneck": ["tshirt", "short sleeve"],

        "black": ["white", "grey"],
        "blue": ["white", "grey"],
        "brown": ["black", "green"],
        "green": ["brown", "yellow"],
        "grey": ["black", "white"],
        "orange": ["red", "yellow"],
        "pink": ["purple", "white"],
        "purple": ["pink", "blue"],
        "red": ["black", "white"],
        "white": ["black", "blue"],
        "yellow": ["green", "orange"],

        "headphone": ["muffler", "hat"],
        "age less 15": ["baby buggy", "age less 30"],
        "baby buggy": ["age less 15", "woman"],
        "bald": ["man", "hat", "sunglasses"],
        "boots": ["trousers", "jacket"],
        "capri": ["hat", "shorts"],

        "shopping tro": ["plastic bags", "backpack"],
        "umbrella": ["raincoat", "jacket"],
        "folder": ["suit", "formal"],
        "hairband": ["long hair", "short hair"],
        "hotpants": ["shorts", "short skirt"],
        "kerchief": ["muffler", "hat"],

        "long skirt": ["short skirt", "stocking"],
        "long sleeve": ["sweater", "jacket", "coat"],
        "no sleeve": ["short sleeve", "vneck"],
        "short hair": ["long hair", "bald"],
        "stocking": ["long skirt", "short skirt"],
        "suit": ["formal", "leather shoes", "suit case"],
        "suits": ["formal", "suit"],
        "sweater": ["jacket", "long sleeve", "muffler"],
    }

    

    def __init__(self, split, path_dataset=None, path_gt=None, path_gt_img=None):

        #super(RAPzsDataset, self)
        super(PETADatasetAll, self)
        
        # Use provided paths or defaults
        if path_dataset is None:
            if split=="train":
                path_dataset = "/mnt/rhome/paa/pedestrian/datasetForFID/PETA/train/"
            elif split=="val":
                path_dataset = "/mnt/rhome/paa/pedestrian/datasetForFID/PETA/val/"
            elif split=="test":
                path_dataset = "/mnt/rhome/paa/pedestrian/datasetForFID/PETA/test/"
        
        self.pathDataset = path_dataset
        
        if path_gt is None:
            path_gt = "/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/PETA/dataset_all.pkl"
        
        if path_gt_img is None:
            path_gt_img = "/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/PETA/images/"

        self.path_gt = path_gt
        self.path_gt_img = path_gt_img
        self.all_images = []
        self.all_vector_label = []
        self.all_captions = []
        self.pathToImages = self.pathDataset
        # all filenames of the images
        listImages = os.listdir(self.pathToImages)
        listImages = [item for item in listImages if ".png" in item]
        #listImages = random.sample(listImages, numImages)

        self.all_images = listImages
        self.class_names=["person"]
        filePkl = open(self.path_gt, 'rb')
        self.dataPkl = pickle.load(filePkl)
        self.filenamesPkl = list(self.dataPkl['image_name'])
        labelsGT = list(self.dataPkl['label'])
        self.labelsGT = [array.tolist() for array in labelsGT]
        self.listAllAttrib = [str(attrib) for attrib in list(self.dataPkl['attr_name'])]

        self.genderIndex=[self.listAllAttrib.index(attrib) for attrib in self.listGenderAtDS]

        self.hsIndex=[self.listAllAttrib.index(attrib) for attrib in self.listHairAtDS]
        self.hsColorIndex=[self.listAllAttrib.index(attrib) for attrib in self.listHairColorAtDS]

        self.carryingIndex=[self.listAllAttrib.index(attrib) for attrib in self.listCarryingAtDS]
        self.accesoryIndex=[self.listAllAttrib.index(attrib) for attrib in self.listAccesoryAtDS]

        self.ubIndex=[self.listAllAttrib.index(attrib) for attrib in self.listUpperBodyAtDS]
        self.ubColorIndex=[self.listAllAttrib.index(attrib) for attrib in self.listUpperBodyColorAtDS]

        self.lbIndex=[self.listAllAttrib.index(attrib) for attrib in self.listLowerBodyAtDS]
        self.lbColorIndex=[self.listAllAttrib.index(attrib) for attrib in self.listLowerBodyColorAtDS]

        self.shoeIndex=[self.listAllAttrib.index(attrib) for attrib in self.listFootwearAtDS]
        self.shoeColorIndex=[self.listAllAttrib.index(attrib) for attrib in self.listFootwearColorAtDS]
        self.idxImgPath = 0
        self.idxImgRGB = 1
        self.idxVector = 2
        
    def __getitem__(self, idx: int) -> Tuple[Image.Image, Any]:
        img = Image.open(self.pathToImages+self.all_images[idx]).convert('RGB')

        indexToLabel = self.filenamesPkl.index(self.all_images[idx])
        labelGT = self.labelsGT[indexToLabel]
        imgPath = self.pathToImages+self.all_images[idx]

        tupla=[imgPath, img, labelGT]

        return tupla

    def generatePrompt(self, labelGT):
        prompt=""
        choice = random.randint(0, len(self.listStartingPrompt)-1)
        prompt+=self.listStartingPrompt[choice]



        #gender
        for idx in self.genderIndex:
            if labelGT[idx] == 1:
                choice = self.listGenderAtDS.index(self.listAllAttrib[idx])
                prompt+=" "+self.listGenderPrompt[choice]

        #color hair
        for idx in self.hsColorIndex:
            if labelGT[idx] == 1:
                #choice = random.randint(0, len(listHsPrompt)-1)
                choice = self.listHairColorAtDS.index(self.listAllAttrib[idx])
                prompt+=" with "+self.listHairColorAtPrompt[choice]
        #hair
        for idx in self.hsIndex:
            if labelGT[idx] == 1:
                if "with" not in prompt:
                    prompt+=" with "
                #choice = random.randint(0, len(listHsPrompt)-1)
                choice = self.listHairAtDS.index(self.listAllAttrib[idx])
                prompt+=" "+self.listHairAtPrompt[choice]

        #carrying
        for idx in self.carryingIndex:
            if labelGT[idx] == 1:
                #choice = random.randint(0, len(listActionPrompt)-1)
                choice = self.listCarryingAtDS.index(self.listAllAttrib[idx])
                prompt+=" carrying "+self.listCarryingAtPrompt[choice]

        #accesory
        for idx in self.accesoryIndex:
            if labelGT[idx] == 1:
                choice = self.listAccesoryAtDS.index(self.listAllAttrib[idx])
                #choice = random.randint(0, len(listAttPrompt)-1)
                prompt+=" with "+self.listAccesoryAtPrompt[choice]

        #color upper body
        for idx in self.ubColorIndex:
            if labelGT[idx] == 1:
                if "wearing" not in prompt:
                    prompt+=" wearing"

                choice = self.listUpperBodyColorAtDS.index(self.listAllAttrib[idx])
                #choice = random.randint(0, len(listUbPrompt)-1)
                prompt+=" "+self.listUpperBodyColorPrompt[choice]

        #upper body
        for idx in self.ubIndex:
            if labelGT[idx] == 1:
                if "wearing" not in prompt:
                    prompt+=" wearing"

                choice = self.listUpperBodyAtDS.index(self.listAllAttrib[idx])
                #choice = random.randint(0, len(listUbPrompt)-1)
                prompt+=" "+self.listUpperBodyPrompt[choice]


        #color lower body
        for idx in self.lbColorIndex:
            if labelGT[idx] == 1:
                if "wearing" not in prompt:
                    prompt+=" wearing"

                choice = self.listLowerBodyColorAtDS.index(self.listAllAttrib[idx])
                #choice = random.randint(0, len(listUbPrompt)-1)
                prompt+=" "+self.listLowerBodyColorAtPrompt[choice]

        #lower body
        for idx in self.lbIndex:
            if labelGT[idx] == 1:
                if "wearing" not in prompt:
                    prompt+=" wearing"

                choice = self.listLowerBodyAtDS.index(self.listAllAttrib[idx])
                #choice = random.randint(0, len(listUbPrompt)-1)
                prompt+=" "+self.listLowerBodyAtPrompt[choice]
        
        #color shoe
        for idx in self.shoeColorIndex:
            if labelGT[idx] == 1:
                if "wearing" not in prompt:
                    prompt+=" wearing"

                choice = self.listFootwearColorAtDS.index(self.listAllAttrib[idx])
                #choice = random.randint(0, len(listShoePrompt)-1)
                prompt+=" "+self.listFootwearColorAtPrompt[choice]
        
        #shoe
        for idx in self.shoeIndex:
            if labelGT[idx] == 1:
                if "wearing" not in prompt:
                    prompt+=" wearing"

                choice = self.listFootwearAtDS.index(self.listAllAttrib[idx])
                #choice = random.randint(0, len(listShoePrompt)-1)
                prompt+=" "+self.listFootwearAtPrompt[choice]

        return prompt, labelGT

    def getLabelByPrompt(self, prompt):
        labelOrig=self.dataPkl['attr_name']
        
        label=[0]*len(labelOrig)

        attributesInPrompt = self.getListAttributesFromPrompt(prompt)

        for attribute in attributesInPrompt:
            idx=labelOrig.index(attribute)
            label[idx]=1

        return label

    def getPrompt(self, idx):
        filenameImage = self.all_images[idx]
        indexToLabel = self.filenamesPkl.index(filenameImage)
        labelGT = self.labelsGT[indexToLabel]

        prompt, labelGT = self.generatePrompt(labelGT)

        return prompt, labelGT

    def getNumberOfAttributesFromVector(self, vector):
        """
        Count how many positions i (up to the min length) satisfy:
            - self.listAllAttrib[i] is in self.listUsedAllPromptAtDS
            - int(vector[i]) == 1
        Robust to NumPy arrays, floats, NaNs, and booleans.
        """
        # Build allowed set once per call (cheap and safe)
        allowed = set(self.listUsedAllPromptAtDS)

        # Determine range to check
        n = min(len(vector), len(self.listAllAttrib))
        if n <= 0:
            return 0

        # Vectorize safely: coerce to 1D float, then to int (Python's int() truncates, so we mimic that)
        try:
            arr = np.asarray(vector).ravel()[:n]               # 1D slice
            vals = arr.astype(float, copy=False)               # may raise if non-numeric objects
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            ints = vals.astype(np.int64, copy=False)           # matches int(1.9)->1, int(True)->1
        except Exception:
            # Fallback to the original per-element logic
            cnt = 0
            for i in range(n):
                if self.listAllAttrib[i] in allowed:
                    try:
                        if int(vector[i]) == 1:
                            cnt += 1
                    except Exception:
                        pass
            return cnt

        # Build a boolean mask of allowed attributes
        allowed_mask = np.fromiter(
            (name in allowed for name in self.listAllAttrib[:n]),
            dtype=bool,
            count=n
        )

        # Count positions where allowed AND value == 1
        return int(np.sum((ints == 1) & allowed_mask))

    def getNumberOfAttributesFromPrompt(self, prompt):
        return len(self.getListAttributesFromPrompt(prompt))

    def getListAttributesFromPrompt(self, prompt):
        listAttributesInPrompt = []

        femaleAtPrompt=False

        for listaAttAtPrompt, listaAttAtDS in zip(self.listOfListUsedToPromptAtPrompt, self.listOfListUsedToPromptAtDS):
            for attributeAtPrompt in listaAttAtPrompt:

                if attributeAtPrompt in prompt and attributeAtPrompt != "man":
                    
                    timesInPrompt = prompt.count(attributeAtPrompt)

                    for i in range(timesInPrompt):

                        idxAttributeAtList = listaAttAtPrompt.index(attributeAtPrompt)
                        
                        nameAtDS = listaAttAtDS[idxAttributeAtList]
                        listAttributesInPrompt.append(nameAtDS)
                    
                    if attributeAtPrompt =="woman":
                        femaleAtPrompt=True

        if femaleAtPrompt == False:
            idx = self.listAttributePrompt.index("man")
            listAttributesInPrompt.append(self.listAttributes[idx])

        return listAttributesInPrompt


    def getLabelVector(self, pathImg):
        filenameImage = pathImg
        indexToLabel = self.filenamesPkl.index(filenameImage)
        labelGT = self.labelsGT[indexToLabel]
        return labelGT

    def __len__(self):
        
        return len(self.all_images)


    def get_image_by_idx(self, idx: int) -> Image.Image:
        img = Image.open(self.pathToImages+self.all_images[idx]).convert('RGB')

        return img
    
    def getImagesWithAttribute(self, attribute, gtZero=False):
        idxAttrib=self.listAllAttrib.index(attribute)
        
        if gtZero==False:
            matching_indexes = [i for i, row in enumerate(self.labelsGT) if row[idxAttrib] == 1]
        else:
            matching_indexes = [i for i, row in enumerate(self.labelsGT) if row[idxAttrib] == 0]

        imgs = [str(self.filenamesPkl[idx]) for idx in matching_indexes]
        # to filter those that are not in the split
        imgs = [img for img in imgs if img in self.all_images ]
        return imgs
    
    def getPromptsWithAttributes(self, attribute, gtZero=False):
        idxAttrib=self.listAllAttrib.index(attribute)
        
        if gtZero==False:
            matching_indexes = [i for i, row in enumerate(self.labelsGT) if row[idxAttrib] == 1]
        else:
            matching_indexes = [i for i, row in enumerate(self.labelsGT) if row[idxAttrib] == 0]

        imgs = [str(self.filenamesPkl[idx]) for idx in matching_indexes]
        # to filter those that are not in the split
        imgsIdx = [self.all_images.index(img) for img in imgs if img in self.all_images ]

        prompts = [self.getPrompt(imgIdx) for imgIdx in imgsIdx]

        return prompts

    def getAllPrompts(self):
        
        # to filter those that are not in the split
        imgsIdx = [self.all_images.index(img) for img in self.all_images ]

        prompts = [self.getPrompt(imgIdx) for imgIdx in imgsIdx]
        vectors = [self.getPrompt(imgIdx)[1] for imgIdx in imgsIdx]

        return prompts, vectors

    def getAttributesAtDS(self, vector):
        # Flatten to 1-D (handles lists, tuples, numpy arrays, ragged inputs)
        arr = np.asarray(vector, dtype=object).ravel()
        n = min(len(arr), len(self.listAllAttrib))
        allowed = set(self.listUsedAllPromptAtDS)

        active_attrs: List[str] = []
        for i in range(n):
            name = self.listAllAttrib[i]
            if name not in allowed:
                continue
            # robust 0/1 read (handles bool/float/str/None); non-numeric -> 0
            try:
                v = int(float(arr[i]))
            except Exception:
                v = 0
            if v == 1:
                active_attrs.append(name)

        return active_attrs
    
    def getPromptsWith1AttributeRemoved(self, prompts, attribute, gtZero=False):

        attributeInPrompt = self.getTranslateAttrib(attribute)
        promptsNeg = [ prompt.replace(attributeInPrompt, "") for prompt in prompts]
        return promptsNeg

    def getPromptsWith1AttributeComplemented(self, prompts, attribute, gtZero=False):

        attributeInPrompt = self.getTranslateAttrib(attribute)

        promptsNeg = [ prompt.replace(attributeInPrompt, random.choice(self.complementary_attributes[attributeInPrompt])) for prompt in prompts]
        return promptsNeg


    def getPromptsComplementAndRemovePercentageAttribute(self, prompts, listImgs, listLabelsGt, percentage=1.0):
        promptsNeg=[]
        # this is to align because could be the option that the neg does not match
        promptsOrig=[]
        listImgsNew=[]
        
        medChange=0
        for prompt, vector, img in zip(prompts, listLabelsGt, listImgs):
            
            
            # Find indices of all 1s
            one_indices = [i for i, val in enumerate(vector) if val == 1]
            nAttributesToChange=int(percentage*len(one_indices))
            medChange+=nAttributesToChange
            # Ensure we don't try to flip more 1s than exist
            #n = min(nAttributesToChange, len(one_indices))

            # Randomly select n indices to flip
            indices_to_flip = random.sample(one_indices, nAttributesToChange)
            
            # Flip selected 1s to 0s
            for idx in indices_to_flip:
                
                vector[idx] = 0
            
            promptNew, labelGTNew =self.generatePrompt(vector)
            #complement everyprompt
            _, promptNeg, _, _ = self.getPromptsCompPercentageAttributes([promptNew], [img], percentageToChange=1.0)
            promptNeg = promptNeg[0]
            promptsNeg.append(promptNeg)
            promptsOrig.append(promptNew)
            listImgsNew.append(img)
        
        medChange=medChange/len(listImgs)

        return promptsOrig, promptsNeg, listImgsNew, medChange


    def getPromptsRemoveNAttribute(self, prompts, listImgs, listLabelsGt, nAttributesToChange=-1):
        promptsNeg=[]
        # this is to align because could be the option that the neg does not match
        promptsOrig=[]
        listImgsNew=[]
 
        for prompt, vector, img in zip(prompts, listLabelsGt, listImgs):
            
            
            # Find indices of all 1s
            one_indices = [i for i, val in enumerate(vector) if val == 1]

            # Ensure we don't try to flip more 1s than exist
            n = min(nAttributesToChange, len(one_indices))

            # Randomly select n indices to flip
            indices_to_flip = random.sample(one_indices, n)
            
            # Flip selected 1s to 0s
            for idx in indices_to_flip:

                vector[idx] = 0
            
            promptNew, labelGTNew=self.generatePrompt(vector)

            promptsNeg.append(promptNew)
            promptsOrig.append(prompt)
            listImgsNew.append(img)
            
        return promptsOrig, promptsNeg, listImgsNew

    import random

    def random_indices_unique_filtered(
        self,
        lst,   # binary vector (0/1 or bool), aligned with self.listAllAttrib
        x,
        seed=None,
    ):
        """
        Return up to x UNIQUE indices i such that:
        - self.listAllAttrib[i] is in self.listUsedAllPromptAtDS, and
        - lst[i] == 1  (truthy; robustly cast to int)

        If x > #eligible, returns all eligible indices in random order.
        """
        if seed is not None:
            random.seed(seed)

        n = min(len(lst), len(self.listAllAttrib))
        allowed = set(self.listUsedAllPromptAtDS)

        eligible = []
        for i in range(n):
            if self.listAllAttrib[i] not in allowed:
                continue
            # robustly interpret lst[i] as 0/1
            try:
                v = int(float(lst[i]))  # handles bools/floats/strings like "1"
            except Exception:
                v = 0
            if v == 1:
                eligible.append(i)

        if not eligible:
            return []

        k = min(max(int(x), 0), len(eligible))
        return random.sample(eligible, k=k)


    def getVectorCompPercentageAttributes(self, vectors, percentageToChange=1.0):
 
        vectorsNew=[]
        medChanged=0
        for vector in vectors:
           
            #listAttributesToChange=listAttributes
            

            #lenAttributes=len(listAttributesToChange)
            totalAttributes = self.getNumberOfAttributesFromVector(vector)
            nAttributesToChange=int(percentageToChange*totalAttributes)
            #print(totalAttributes)
            #print(percentageToChange)
            #print(nAttributesToChange)
            medChanged+=nAttributesToChange

            # get only idx that are into account
            idxs = self.random_indices_unique_filtered(vector, nAttributesToChange, seed=42)
            # print(vector)
            vector_np  = np.array(vector, dtype=int, copy=True)
            vectorOrig = vector_np.copy()     # snapshot to compare against
            newVector  = vector_np.copy()     # mutable working copy

            listAttributesComplemented = []

            if len(idxs)==0:
                print("Impossible to complement 0 labels")
                wellConstructedVector=False

            for i in idxs:
                wellConstructedVector=True
                # get attribute as is called in the dataset
                attributeCrudo = self.listAllAttrib[i]

                attributeCrudoComp=None

                complements=list(self.attr_complements[attributeCrudo])
                
                if len(complements) == 0:
                    print("Impossible to complement the prompt, pass")
                    attributeCrudoComp=None
                    compAvailable=True
                else:
                    compAvailable=False

                # get a complemented choice
                while compAvailable == False:
                    compAvailable=True

                    attributeCrudoComp = random.choice(complements)
                    
                    idxAttributeCrudoComp = self.listAllAttrib.index(attributeCrudoComp)

                    if attributeCrudoComp in listAttributesComplemented:
                        #print("Attribute already complemented, searching new")
                        #print(attributeCrudoComp)
                        compAvailable=False
                        complements.remove(attributeCrudoComp)
                    
                    elif vectorOrig[idxAttributeCrudoComp] == 1: # is at prompt and is not going to be changed
                        #print("Attribute already complemented, searching new")
                        #print(attributeCrudoComp)
                        compAvailable=False
                        complements.remove(attributeCrudoComp)
                    else:
                        for j in idxs:
                            # check if the attribute to comp is already in the list to be complemented or was complemented early
                            if self.listAllAttrib[j] == attributeCrudoComp:
                                compAvailable=False
                                #print("Attribute not compatible to be complemented, searching new")
                                #print(attributeCrudoComp)
                                complements.remove(attributeCrudoComp)

                                
                    if len(complements) == 0:
                        print("Impossible to complement the prompt, pass")
                        compAvailable=True
                        attributeCrudoComp=None
                        
                #print(attributeCrudoComp)

                if attributeCrudoComp != None:

                    # get the idx for the vector of that crudocomp
                    idxAttributeCrudoComp = self.listAllAttrib.index(attributeCrudoComp)
                    listAttributesComplemented.append(attributeCrudoComp)
                    # swap the vectors
                    newVector[idxAttributeCrudoComp]=1
                    newVector[i]=0    
                    attributeCrudoComp=None
                else:
                    # out the prompt because is impossible to make it
                    wellConstructedVector=False
                    break
            

            if wellConstructedVector:
                
                
                #assert labelGTNew == newVector, "caution vector neg was changed"
                assert not np.array_equal(newVector, vectorOrig), "caution: vector neg was not altered from the original"
                assert np.array_equal(vectorOrig, np.array(vector, dtype=int)), "caution: original vector was changed"
                assert self.getNumberOfAttributesFromVector(vector) == self.getNumberOfAttributesFromVector(newVector.tolist()), "prompts complemented not match number of attributes\n "
                #print(vector)

                
                vectorsNew.append(newVector.tolist())
        



        return vectorsNew

    def getPromptsCompPercentageAttributes(self, prompts, vectors, listImgs, percentageToChange=1.0):
        promptsNeg=[]
        # this is to align because could be the option that the neg does not match
        promptsOrig=[]
        vectorsOrig=[]
        vectorsNew=[]
        listImgsNew=[]
        #listAttributesFromPrompts=[self.getListAttributesFromPrompt(prompt) for prompt in prompts]
        medChanged=0
        for prompt, vector, img in zip(prompts, vectors, listImgs):
           
            #listAttributesToChange=listAttributes
            

            #lenAttributes=len(listAttributesToChange)
            totalAttributes = self.getNumberOfAttributesFromVector(vector)
            nAttributesToChange=int(percentageToChange*totalAttributes)
            #print(totalAttributes)
            #print(percentageToChange)
            #print(nAttributesToChange)
            medChanged+=nAttributesToChange

            # get only idx that are into account
            idxs = self.random_indices_unique_filtered(vector, nAttributesToChange, seed=42)
            # print(vector)
            vector_np  = np.array(vector, dtype=int, copy=True)
            vectorOrig = vector_np.copy()     # snapshot to compare against
            newVector  = vector_np.copy()     # mutable working copy

            listAttributesComplemented = []

            if len(idxs)==0:
                print("Impossible to complement 0 labels")
                wellConstructedVector=False

            for i in idxs:
                wellConstructedVector=True
                # get attribute as is called in the dataset
                attributeCrudo = self.listAllAttrib[i]

                attributeCrudoComp=None

                complements=list(self.attr_complements[attributeCrudo])
                
                if len(complements) == 0:
                    print("Impossible to complement the prompt, pass")
                    attributeCrudoComp=None
                    compAvailable=True
                else:
                    compAvailable=False

                # get a complemented choice
                while compAvailable == False:
                    compAvailable=True

                    attributeCrudoComp = random.choice(complements)
                    
                    idxAttributeCrudoComp = self.listAllAttrib.index(attributeCrudoComp)

                    if attributeCrudoComp in listAttributesComplemented:
                        #print("Attribute already complemented, searching new")
                        #print(attributeCrudoComp)
                        compAvailable=False
                        complements.remove(attributeCrudoComp)
                    
                    elif vectorOrig[idxAttributeCrudoComp] == 1: # is at prompt and is not going to be changed
                        #print("Attribute already complemented, searching new")
                        #print(attributeCrudoComp)
                        compAvailable=False
                        complements.remove(attributeCrudoComp)
                    else:
                        for j in idxs:
                            # check if the attribute to comp is already in the list to be complemented or was complemented early
                            if self.listAllAttrib[j] == attributeCrudoComp:
                                compAvailable=False
                                #print("Attribute not compatible to be complemented, searching new")
                                #print(attributeCrudoComp)
                                complements.remove(attributeCrudoComp)

                                
                    if len(complements) == 0:
                        print("Impossible to complement the prompt, pass")
                        compAvailable=True
                        attributeCrudoComp=None
                        
                #print(attributeCrudoComp)

                if attributeCrudoComp != None:

                    # get the idx for the vector of that crudocomp
                    idxAttributeCrudoComp = self.listAllAttrib.index(attributeCrudoComp)
                    listAttributesComplemented.append(attributeCrudoComp)
                    # swap the vectors
                    newVector[idxAttributeCrudoComp]=1
                    newVector[i]=0    
                    attributeCrudoComp=None
                else:
                    # out the prompt because is impossible to make it
                    wellConstructedVector=False
                    break
            

            if wellConstructedVector:
                
                promptNew, labelGTNew = self.generatePrompt(newVector.tolist())

                #assert labelGTNew == newVector, "caution vector neg was changed"
                assert not np.array_equal(newVector, vectorOrig), "caution: vector neg was not altered from the original"
                assert np.array_equal(vectorOrig, np.array(vector, dtype=int)), "caution: original vector was changed"
                assert self.getNumberOfAttributesFromVector(vector) == self.getNumberOfAttributesFromVector(labelGTNew), "prompts complemented not match number of attributes\n prompt pos: {} \n prompt neg: {}".format(prompt, promptNew)
                #print(vector)

                
                promptsNeg.append(promptNew)
                promptsOrig.append(prompt)
                listImgsNew.append(img)
                vectorsOrig.append(vector)
                vectorsNew.append(newVector.tolist())
        
        medChanged=medChanged/len(listImgs)


        return promptsOrig, promptsNeg, listImgsNew, medChanged, vectorsOrig, vectorsNew

    def getPromptCompSpecificAttributeAllPos(self, prompts, vector, listImgs, attribute):

        allNewVectors = []
  
        attributesInPromptPos = self.getAttributesAtDS(vector)
        
        attributeCrudo=attribute

        attributesInPromptPos.remove(attributeCrudo)

        i=self.dataPkl['attr_name'].index(attributeCrudo)

        vector_np  = np.array(vector, dtype=int, copy=True)
        vectorOrig = vector_np.copy()     # snapshot to compare against
        newVector  = vector_np.copy()     # mutable working copy

        attributeCrudoComp=None

        complements=list(self.attr_complements[attributeCrudo])
        
    
        if len(complements) == 0:
            print("Impossible to complement the prompt, pass")
            attributeCrudoComp=None
            compAvailable=True
        else:
            compAvailable=False


            
        # get a complemented choice
        while compAvailable == False and len(complements) > 0:
            compAvailable=True

            attributeCrudoComp = random.choice(complements)
            
            idxAttributeCrudoComp = self.listAllAttrib.index(attributeCrudoComp)
            if attributeCrudoComp in attributesInPromptPos:
                #print("Attribute already complemented, searching new")
                #print(attributeCrudoComp)
                compAvailable=False
                complements.remove(attributeCrudoComp)
            elif vectorOrig[idxAttributeCrudoComp] == 1: # is at prompt and is not going to be changed
                #print("Attribute already complemented, searching new")
                #print(attributeCrudoComp)
                compAvailable=False
                complements.remove(attributeCrudoComp)
       
            if len(complements) == 0:
                print("Impossible to complement the prompt, pass")
                compAvailable=True
                attributeCrudoComp=None
            
            if attributeCrudoComp != None:
                
                # get the idx for the vector of that crudocomp
                idxAttributeCrudoComp = self.listAllAttrib.index(attributeCrudoComp)
                
                # swap the vectors
                newVector[idxAttributeCrudoComp]=1
                newVector[i]=0  

                allNewVectors.append(newVector)

        if len(allNewVectors) > 0:
            return None, None, listImgs, 1, vectorOrig, allNewVectors

        else:
            return None, None, None, None, None, None


    def getPromptCompSpecificAttribute(self, prompts, vector, listImgs, attribute):

        attributesInPromptPos = self.getAttributesAtDS(vector)
        
        attributeCrudo=attribute

        attributesInPromptPos.remove(attributeCrudo)

        i=self.dataPkl['attr_name'].index(attributeCrudo)

        vector_np  = np.array(vector, dtype=int, copy=True)
        vectorOrig = vector_np.copy()     # snapshot to compare against
        newVector  = vector_np.copy()     # mutable working copy

        attributeCrudoComp=None

        complements=list(self.attr_complements[attributeCrudo])
        
    
        if len(complements) == 0:
            print("Impossible to complement the prompt, pass")
            attributeCrudoComp=None
            compAvailable=True
        else:
            compAvailable=False

 
        # get a complemented choice
        while compAvailable == False:
            compAvailable=True

            attributeCrudoComp = random.choice(complements)
            idxAttributeCrudoComp = self.listAllAttrib.index(attributeCrudoComp)
            if attributeCrudoComp in attributesInPromptPos:
                #print("Attribute already complemented, searching new")
                #print(attributeCrudoComp)
                compAvailable=False
                complements.remove(attributeCrudoComp)
            elif vectorOrig[idxAttributeCrudoComp] == 1: # is at prompt and is not going to be changed
                #print("Attribute already complemented, searching new")
                #print(attributeCrudoComp)
                compAvailable=False
                complements.remove(attributeCrudoComp)
       
            if len(complements) == 0:
                print("Impossible to complement the prompt, pass")
                compAvailable=True
                attributeCrudoComp=None
                

        if attributeCrudoComp != None:

            # get the idx for the vector of that crudocomp
            idxAttributeCrudoComp = self.listAllAttrib.index(attributeCrudoComp)
            
            # swap the vectors
            newVector[idxAttributeCrudoComp]=1
            newVector[i]=0    
            
            return None, None, listImgs, 1, vectorOrig.tolist(), newVector.tolist()

        else:
            return None, None, None, None, None, None

    def getPromptsRemovePercentageAttribute(self, prompts, listImgs, listLabelsGt, percentage=1.0):
        promptsNeg=[]
        # this is to align because could be the option that the neg does not match
        promptsOrig=[]
        listImgsNew=[]
        
        medChange=0
        for prompt, vector, img in zip(prompts, listLabelsGt, listImgs):
            
            # Find indices of all 1s
            one_indices = [i for i, val in enumerate(vector) if val == 1]
            nAttributesToChange=int(percentage*len(one_indices))
            medChange+=nAttributesToChange
            # Ensure we don't try to flip more 1s than exist
            #n = min(nAttributesToChange, len(one_indices))

            # Randomly select n indices to flip
            indices_to_flip = random.sample(one_indices, nAttributesToChange)
            
            # Flip selected 1s to 0s
            for idx in indices_to_flip:
                
                vector[idx] = 0
            
            promptNew, labelGTNew =self.generatePrompt(vector)
            #complement everyprompt
            #promptNeg=promptNew
            promptNeg = promptNew
            promptsNeg.append(promptNeg)
            promptsOrig.append(prompt)
            listImgsNew.append(img)
        
        medChange=medChange/len(listImgs)

        return promptsOrig, promptsNeg, listImgsNew, medChange

    def getPromptsCompNAttributes(self, prompts, listImgs, nAttributesToChange=-1):
        promptsNeg=[]
        # this is to align because could be the option that the neg does not match
        promptsOrig=[]
        listImgsNew=[]
        listAttributesFromPrompts=[self.getListAttributesFromPrompt(prompt) for prompt in prompts]
        medChanged=0
        for prompt, listAttributes, img in zip(prompts, listAttributesFromPrompts, listImgs):
            listAttributesToChange=listAttributes
            medChanged+=nAttributesToChange
            if nAttributesToChange != -1:

                if nAttributesToChange >= len(listAttributesToChange):
                    continue
                else:
                    
                    listAttributesToChange = random.sample(listAttributesToChange, nAttributesToChange)
                
            promptNew=prompt
            #print(promptNew)
            for attributeCrudo in listAttributesToChange:
                if attributeCrudo == 'man':
                    attribute='man'
                else:
                    attribute=self.listAttributePrompt[self.listAttributes.index(attributeCrudo)]

                if attribute in prompt:
                    if attribute == 'woman':
                        complementaryAttribute = 'man'
                    elif attribute == 'man':
                        complementaryAttribute = 'woman'
                    else:
                        complementaryAttribute = random.choice(self.complementary_attributes[attribute])

                    promptNew = promptNew.replace(attribute, complementaryAttribute)
                
            #print(promptNew)

            promptsNeg.append(promptNew)
            promptsOrig.append(prompt)
            listImgsNew.append(img)
        
        medChanged=int(medChanged/len(listImgs))

        return promptsOrig, promptsNeg, listImgsNew

    def getTranslateAttrib(self, attribute):
        idx=self.listAttributes.index(attribute)
        return self.listAttributePrompt[idx]
    
    def getPromptForOneAttribute(self, attribute):
        template="A photo of a "
        return template+attribute
    
    def getPromptForOneAttributeComp(self, attribute):
        template="A photo of a "
        
        return template+random.choice(self.complementary_attributes[attribute])

    def getPromptForOneAttributeNeg(self, attribute):
        template="A photo of a non "
        return template+attribute
     
    def getQuestionFromAttribute(self, attribute):
        template=f"Is there the attribute {attribute}?"
        return template
    
    def getQuestionFromAttributeNeg(self, attribute):
        template=f"Is not there the attribute {attribute}?"
        return template
    
