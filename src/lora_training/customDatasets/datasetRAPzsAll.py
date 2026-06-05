
from torch.utils.data import Dataset
import pickle
import random
from PIL import Image
import os
import numpy as np
from typing import Any, Tuple, Dict, List
TRAIN_FOLDER ="train/"

FEMALE_ATTR = "Femal"

class RAPzsDatasetAll(Dataset):

    listStartingPrompt=['a', 'there is a']

    listGenderAtDS=['Femal']
    listGenderPrompt=['woman', 'man']

    listHsAtDS=['hs-BaldHead', 'hs-LongHair', 'hs-BlackHair', 'hs-Hat', 'hs-Glasses']
    listHsPrompt=['bald head', 'long hair', 'black hair', 'hat', 'glasses']

    listActionsAtDS=['action-Calling','action-Talking','action-Gathering','action-Holding','action-Pushing','action-Pulling','action-CarryingByArm','action-CarryingByHand','action-Other']
    listActionPrompt=['calling','talking','gathering','holding','pushing','pulling','carrying by arm','carrying by hand','other']

    listAttAtDS=['attachment-Backpack','attachment-ShoulderBag','attachment-HandBag','attachment-Box','attachment-PlasticBag','attachment-PaperBag','attachment-HandTrunk','attachment-Other']
    listAttPrompt=['backpack','shoulder bag','hand bag','box','plastic bag','paper bag','hand trunk','other']

    listUbAtDS=['ub-Shirt','ub-Sweater','ub-Vest','ub-TShirt','ub-Cotton','ub-Jacket','ub-SuitUp','ub-Tight','ub-ShortSleeve','ub-Others']
    listUbPrompt = ['shirt','sweater','vest','t-shirt','cotton','jacket','suit','tight','short sleeve','others']

    listLbAtDS=['lb-LongTrousers','lb-Skirt','lb-ShortSkirt','lb-Dress','lb-Jeans','lb-TightTrousers']
    listLbPrompt=['long trousers','skirt','short skirt','dress','jeans','tight trousers']

    listShoeAtDS=['shoes-Leather', 'shoes-Sports', 'shoes-Boots', 'shoes-Cloth', 'shoes-Casual', 'shoes-Other']
    listShoePrompt=['leather shoes', 'sports shoes', 'boots', 'cloth shoes', 'casual shoes', 'other shoes']
    
    # listas para contar los attributos del prompt
    listOfListUsedToPromptAtDS=[listGenderAtDS, listHsAtDS, listActionsAtDS, listAttAtDS, listUbAtDS, \
                                listLbAtDS, listShoeAtDS]
    listOfListUsedToPromptAtPrompt=[listGenderPrompt, listHsPrompt, listActionPrompt, listAttPrompt, listUbPrompt, \
                                    listLbPrompt, listShoePrompt]
    listUsedAllPromptAtDS = [attribute for lista in listOfListUsedToPromptAtDS for attribute in lista ]

    attr_complements = {
        attr: [other for other in group if other != attr]
        for group in listOfListUsedToPromptAtDS
        for attr in group
    }

    listAttributePrompt=['bald head', 'long hair', 'black hair', 'hat', 'glasses', 'shirt','sweater','vest','t-shirt','cotton','jacket','suit','tight','short sleeve','others', 'long trousers','skirt','short skirt','dress','jeans','tight trousers', 'leather shoes', 'sports shoes', 'boots', 'cloth shoes', 'casual shoes', 'other shoes', 'backpack','shoulder bag','hand bag','box','plastic bag','paper bag','hand trunk','other', 'ages less 16', 'age between 17 and 30', 'age between 31 and 45', 'woman', 'fat body','normal body','thin body','customer','employee', 'calling','talking','gathering','holding','pushing','pulling','carrying by arm','carrying by hand','other']

    listAttributesRAPv2 = ['hs-BaldHead', 'hs-LongHair', 'hs-BlackHair', 'hs-Hat', 'hs-Glasses','ub-Shirt','ub-Sweater','ub-Vest','ub-TShirt','ub-Cotton','ub-Jacket','ub-SuitUp','ub-Tight','ub-ShortSleeve','ub-Others','lb-LongTrousers','lb-Skirt','lb-ShortSkirt','lb-Dress','lb-Jeans','lb-TightTrousers','shoes-Leather', 'shoes-Sports', 'shoes-Boots', 'shoes-Cloth', 'shoes-Casual', 'shoes-Other','attachment-Backpack','attachment-ShoulderBag','attachment-HandBag','attachment-Box','attachment-PlasticBag','attachment-PaperBag','attachment-HandTrunk','attachment-Other','AgeLess16', 'Age17-30', 'Age31-45', 'Age46-60','Femal','BodyFat','BodyNormal','BodyThin','Customer','Employee','action-Calling','action-Talking','action-Gathering','action-Holding','action-Pushing','action-Pulling','action-CarryingByArm','action-CarryingByHand','action-Other']
    # from preprocess rethinking
    listAttributesRAPv2.remove('Age46-60')
    listAttributes = listAttributesRAPv2

    complementary_attributes = {
            'bald head': ['long hair', 'black hair'],
            'long hair': ['bald head'],
            'black hair': ['bald head'],
            'hat': ['bald head', 'long hair', 'black hair'],
            'glasses': ['others'],

            'shirt': ['t-shirt', 'sweater', 'jacket'],
            'sweater': ['shirt', 't-shirt', 'vest'],
            'vest': ['shirt', 'jacket', 'sweater'],
            't-shirt': ['shirt', 'jacket', 'sweater'],
            'cotton': ['suit', 'jacket', 'others'],
            'jacket': ['shirt', 't-shirt', 'vest'],
            'suit': ['shirt', 'cotton', 'jeans'],
            'tight': ['long trousers', 'skirt', 'sweater'],
            'short sleeve': ['sweater', 'jacket', 'shirt'],
            'others': ['shirt', 't-shirt', 'jacket'],

            'long trousers': ['skirt', 'short skirt', 'dress'],
            'skirt': ['long trousers', 'jeans'],
            'short skirt': ['long trousers', 'skirt'],
            'dress': ['long trousers', 'shirt', 'jacket'],
            'jeans': ['skirt', 'suit', 'dress'],
            'tight trousers': ['jeans', 'long trousers', 'skirt'],

            'leather shoes': ['sports shoes', 'cloth shoes', 'casual shoes'],
            'sports shoes': ['leather shoes', 'boots', 'casual shoes'],
            'boots': ['casual shoes', 'cloth shoes', 'leather shoes'],
            'cloth shoes': ['leather shoes', 'boots', 'sports shoes'],
            'casual shoes': ['leather shoes', 'sports shoes'],
            'other shoes': ['leather shoes', 'sports shoes'],

            'backpack': ['shoulder bag', 'hand bag', 'box'],
            'shoulder bag': ['backpack', 'plastic bag', 'paper bag'],
            'hand bag': ['backpack', 'box', 'plastic bag'],
            'box': ['hand bag', 'plastic bag', 'paper bag'],
            'plastic bag': ['hand trunk', 'box', 'backpack'],
            'paper bag': ['box', 'plastic bag', 'backpack'],
            'hand trunk': ['plastic bag', 'box', 'shoulder bag'],
            'other': ['backpack', 'hand bag', 'box'],

            'ages less 16': ['age between 17 and 30', 'age between 31 and 45'],
            'age between 17 and 30': ['ages less 16', 'age between 31 and 45'],
            'age between 31 and 45': ['ages less 16', 'age between 17 and 30'],
            'woman': ['man'],
            'man': ['woman'],

            'fat body': ['thin body', 'normal body'],
            'normal body': ['fat body', 'thin body'],
            'thin body': ['fat body', 'normal body'],

            'customer': ['employee'],
            'employee': ['customer'],

            'calling': ['talking', 'gathering', 'other'],
            'talking': ['calling', 'gathering', 'pushing'],
            'gathering': ['talking', 'pushing', 'pulling'],
            'holding': ['pushing', 'pulling', 'carrying by hand'],
            'pushing': ['pulling', 'carrying by hand'],
            'pulling': ['pushing', 'holding'],
            'carrying by arm': ['carrying by hand', 'holding'],
            'carrying by hand': ['carrying by arm', 'holding'],
            'other': ['calling', 'talking', 'gathering']
        }


    def __init__(self, split, path_dataset=None, path_gt=None, path_gt_img=None):

        #super(RAPzsDataset, self)
        super(RAPzsDatasetAll, self)
        
        # Use provided paths or defaults
        if path_dataset is None:
            if split=="train":
                path_dataset = "/mnt/rhome/paa/pedestrian/datasetForFID/RAPzs/train/"
            elif split=="val":
                path_dataset = "/mnt/rhome/paa/pedestrian/datasetForFID/RAPzs/val/"
            elif split=="test":
                path_dataset = "/mnt/rhome/paa/pedestrian/datasetForFID/RAPzs/test/"
        
        self.pathDataset = path_dataset
        
        if path_gt is None:
            path_gt = "/mnt/rhome/paa/pedestrian/dataAugmentationMethods/datasets/realOnes/RAPzs_100/dataset_zs_run0.pkl"
        
        if path_gt_img is None:
            path_gt_img = "/mnt/rhome/paa/pedestrian/Rethinking_of_PAR/data/RAP2/RAP_dataset/"
        
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
        self.hsIndex=[self.listAllAttrib.index(attrib) for attrib in self.listHsAtDS]
        self.actionIndex=[self.listAllAttrib.index(attrib) for attrib in self.listActionsAtDS]
        self.attIndex=[self.listAllAttrib.index(attrib) for attrib in self.listAttAtDS]
        self.ubIndex=[self.listAllAttrib.index(attrib) for attrib in self.listUbAtDS]
        self.lbIndex=[self.listAllAttrib.index(attrib) for attrib in self.listLbAtDS]
        self.shoeIndex=[self.listAllAttrib.index(attrib) for attrib in self.listShoeAtDS]
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

        choiceMan = 1
        choiceWoman = 0
        for idx in self.genderIndex:
            if labelGT[idx] == 1:
                #choice = random.randint(0, len(listGenderPrompt)-1)
                prompt+=" "+self.listGenderPrompt[choiceWoman]
            elif labelGT[idx] == 0:
                prompt+=" "+self.listGenderPrompt[choiceMan]


        for idx in self.hsIndex:
            if labelGT[idx] == 1:
                #choice = random.randint(0, len(listHsPrompt)-1)
                choice = self.listHsAtDS.index(self.listAllAttrib[idx])
                prompt+=" with "+self.listHsPrompt[choice]

        for idx in self.actionIndex:
            if labelGT[idx] == 1:
                #choice = random.randint(0, len(listActionPrompt)-1)
                choice = self.listActionsAtDS.index(self.listAllAttrib[idx])
                prompt+=" is "+self.listActionPrompt[choice]

        for idx in self.attIndex:
            if labelGT[idx] == 1:
                choice = self.listAttAtDS.index(self.listAllAttrib[idx])
                #choice = random.randint(0, len(listAttPrompt)-1)
                prompt+=" "+self.listAttPrompt[choice]


        for idx in self.ubIndex:
            if labelGT[idx] == 1:
                if "wearing" not in prompt:
                    prompt+=" wearing"

                choice = self.listUbAtDS.index(self.listAllAttrib[idx])
                #choice = random.randint(0, len(listUbPrompt)-1)
                prompt+=" "+self.listUbPrompt[choice]


        for idx in self.lbIndex:
            if labelGT[idx] == 1:
                if "wearing" not in prompt:
                    prompt+=" wearing"

                choice = self.listLbAtDS.index(self.listAllAttrib[idx])
                #choice = random.randint(0, len(listLbPrompt)-1)
                prompt+=" "+self.listLbPrompt[choice]

        for idx in self.shoeIndex:
            if labelGT[idx] == 1:
                if "wearing" not in prompt:
                    prompt+=" wearing"

                choice = self.listShoeAtDS.index(self.listAllAttrib[idx])
                #choice = random.randint(0, len(listShoePrompt)-1)
                prompt+=" "+self.listShoePrompt[choice]
        return prompt, labelGT

    def getLabelByPrompt(self, prompt):
        labelOrig=self.dataPkl['attr_name']
        
        label=[0]*len(labelOrig)

        attributesInPrompt = self.getListAttributesFromPrompt(prompt)

        for attribute in attributesInPrompt:
            idx=labelOrig.index(attribute)
            label[idx]=1

        return label

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

    def getPrompt(self, idx):
        filenameImage = self.all_images[idx]
        indexToLabel = self.filenamesPkl.index(filenameImage)
        labelGT = self.labelsGT[indexToLabel]

        prompt, labelGT = self.generatePrompt(labelGT)

        return prompt, labelGT

    def getLabelByPrompt(self, prompt):
        labelOrig=self.dataPkl['attr_name']
        
        label=[0]*len(labelOrig)

        attributesInPrompt = self.getListAttributesFromPrompt(prompt)

        for attribute in attributesInPrompt:
            if attribute != 'man':
                idx=labelOrig.index(attribute)
                label[idx]=1

        return label
    
    import numpy as np
    
    def getNumberOfAttributesFromVector(self, vector) -> int:
        """
        Count how many positions i (up to the min length) satisfy:
        - self.listAllAttrib[i] is in self.listUsedAllPromptAtDS
        - vector[i] == 1
        Plus: ensure the 'personalFemale' attribute contributes +1 if its value is 0 or 1
            (so gender counts whether it's male(0) or female(1)).

        Always returns a Python int.
        """
        allowed = set(self.listUsedAllPromptAtDS)
        

        n = min(len(vector), len(self.listAllAttrib))
        if n <= 0:
            return 0

        try:
            arr  = np.asarray(vector).ravel()[:n]          # original values (any dtype)
            vals = arr.astype(float, copy=False)           # may contain NaN/inf
            ints = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
        except Exception:
            # Safe fallback (element-wise)
            cnt = 0
            female_extra = 0
            fidx = None
            try:
                fidx = self.listAllAttrib.index(FEMALE_ATTR)
            except ValueError:
                fidx = None

            for i in range(n):
                name = self.listAllAttrib[i]
                if name not in allowed:
                    continue
                # count 1's
                try:
                    if int(vector[i]) == 1:
                        cnt += 1
                except Exception:
                    pass

            # female bonus (+1 if value is 0 and attribute allowed)
            if fidx is not None and fidx < n and self.listAllAttrib[fidx] in allowed:
                try:
                    v = int(float(vector[fidx]))
                    if v in (0, 1) and v == 0:
                        female_extra = 1
                except Exception:
                    pass

            return int(cnt + female_extra)

        # vectorized path
        allowed_mask = np.fromiter(
            (name in allowed for name in self.listAllAttrib[:n]),
            dtype=bool,
            count=n
        )
        base_count = int(np.sum((ints == 1) & allowed_mask))

        # female +1 if value is 0 (so gender contributes either way)
        female_extra = 0
        if FEMALE_ATTR in self.listAllAttrib:
            fidx = self.listAllAttrib.index(FEMALE_ATTR)
            if fidx < n and allowed_mask[fidx]:
                # check it's a proper binary 0/1 (avoid counting weird values)
                is_binary = np.isfinite(vals[fidx]) and (vals[fidx] == 0.0 or vals[fidx] == 1.0)
                if is_binary and ints[fidx] == 0:
                    female_extra = 1

        return int(base_count + female_extra)

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
            listAttributesInPrompt.append("man")

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
        
        special_index=self.listAllAttrib.index('Femal')  # This is the index we must check
        medChange=0
        for prompt, vector, img in zip(prompts, listLabelsGt, listImgs):
            
            special_PreviousValue=vector[special_index]
            # Step 1: Force special index to 1 if it's 0
            if vector[special_index] == 0:
                vector[special_index] = 1

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
                if special_index==idx:
                    vector[idx] = -1
                else:
                    vector[idx] = 0
            
            # if woman/man was not selected to change, mantain it as previous one
            # if has to be changed, will be a -1 just to not take into account like
            # woman or man
            if special_index not in indices_to_flip:
                vector[special_index]=special_PreviousValue

            promptNew=self.generatePrompt(vector)
            #complement everyprompt
            _, promptNeg, _, _ = self.getPromptsCompPercentageAttributes([promptNew], [img], percentageToChange=1.0)
            promptNeg = promptNeg[0]
            promptsNeg.append(promptNeg)
            promptsOrig.append(promptNew)
            listImgsNew.append(img)
        
        medChange=medChange/len(listImgs)

        return promptsOrig, promptsNeg, listImgsNew, medChange

    def getPromptsRemovePercentageAttribute(self, prompts, listImgs, listLabelsGt, percentage=1.0):
        promptsNeg=[]
        # this is to align because could be the option that the neg does not match
        promptsOrig=[]
        listImgsNew=[]
        
        special_index=self.listAllAttrib.index('Femal')  # This is the index we must check
        medChange=0
        for prompt, vector, img in zip(prompts, listLabelsGt, listImgs):
            
            special_PreviousValue=vector[special_index]
            # Step 1: Force special index to 1 if it's 0
            if vector[special_index] == 0:
                vector[special_index] = 1

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
                if special_index==idx:
                    vector[idx] = -1
                else:
                    vector[idx] = 0
            
            # if woman/man was not selected to change, mantain it as previous one
            # if has to be changed, will be a -1 just to not take into account like
            # woman or man
            if special_index not in indices_to_flip:
                vector[special_index]=special_PreviousValue

            promptNew=self.generatePrompt(vector)
            #complement everyprompt
            #promptNeg=promptNew
            promptNeg = promptNew
            promptsNeg.append(promptNeg)
            promptsOrig.append(prompt)
            listImgsNew.append(img)
        
        medChange=medChange/len(listImgs)

        return promptsOrig, promptsNeg, listImgsNew, medChange


    def getPromptsRemoveNAttribute(self, prompts, listImgs, listLabelsGt, nAttributesToChange=-1):
        promptsNeg=[]
        # this is to align because could be the option that the neg does not match
        promptsOrig=[]
        listImgsNew=[]
        
        special_index=self.listAllAttrib.index('Femal')  # This is the index we must check

        for prompt, vector, img in zip(prompts, listLabelsGt, listImgs):
            
            
            previousSpecialIndex=vector[special_index]
           
            # Step 1: Force special index to 1 if it's 0
            if vector[special_index] == 0:
                vector[special_index] = 1

            # Find indices of all 1s
            one_indices = [i for i, val in enumerate(vector) if val == 1]

            # Ensure we don't try to flip more 1s than exist
            n = min(nAttributesToChange, len(one_indices))

            # Randomly select n indices to flip
            indices_to_flip = random.sample(one_indices, n)
            
            # Flip selected 1s to 0s
            for idx in indices_to_flip:
                if special_index==idx:
                    vector[idx] = -1
                else:
                    vector[idx] = 0
            
            promptNew=self.generatePrompt(vector)

            promptsNeg.append(promptNew)
            promptsOrig.append(prompt)
            listImgsNew.append(img)
            
        return promptsOrig, promptsNeg, listImgsNew

    def random_indices_unique_filtered(
        self,
        lst,   # vector aligned with self.listAllAttrib
        x,
        seed=None,
    ):
        """
        Return up to x UNIQUE indices i such that:
        - self.listAllAttrib[i] is in self.listUsedAllPromptAtDS, and
        - if attribute == 'personalFemale': value must be exactly 0 or 1 (both eligible)
        - else: value must be 1

        If x > #eligible, returns all eligible indices in random order.
        """
        if seed is not None:
            random.seed(seed)

        #FEMALE_ATTR = "personalFemale"
        n = min(len(lst), len(self.listAllAttrib))
        allowed = set(self.listUsedAllPromptAtDS)

        def to_float(v):
            try:
                f = float(v)
                # NaN check without numpy
                return f if f == f else None
            except Exception:
                return None

        eligible = []
        for i in range(n):
            attr = self.listAllAttrib[i]
            if attr not in allowed:
                continue

            v = to_float(lst[i])

            if attr == FEMALE_ATTR:
                # eligible only if value is exactly 0 or 1
                if v in (0.0, 1.0):
                    eligible.append(i)
            else:
                # other attributes: require value == 1
                if v is not None and int(v) == 1:
                    eligible.append(i)

        if not eligible:
            return []

        k = min(max(int(x), 0), len(eligible))
        return random.sample(eligible, k=k)

    def getVectorCompPercentageAttributes(self, vectors, percentageToChange=1.0):

        vectorsNew=[]

        #listAttributesFromPrompts=[self.getListAttributesFromPrompt(prompt) for prompt in prompts]
        medChanged=0
        for vector in vectors:

            #listAttributesToChange=listAttributes
            

            #lenAttributes=len(listAttributesToChange)
            totalAttributes = self.getNumberOfAttributesFromVector(vector)
            nAttributesToChange=int(percentageToChange*totalAttributes)
            medChanged+=nAttributesToChange

            # get only idx that are into account
            idxs = self.random_indices_unique_filtered(vector, nAttributesToChange, seed=42)
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

                if attributeCrudo == FEMALE_ATTR:
                    
                    if vectorOrig[i] == 0: # means that is a man
                        attributeCrudoComp='woman'
                        complements=['woman']
                    else:
                        attributeCrudoComp='man'
                        complements=['man']
                        
                else:
                    complements=list(self.attr_complements[attributeCrudo])
                
         
                if len(complements) == 0:
                    print("Impossible to complement the prompt, pass")
                    attributeCrudoComp=None
                    compAvailable=True
                else:
                    compAvailable=False

                
                
                if attributeCrudoComp == 'man' or attributeCrudoComp == 'woman':
                    compAvailable=True

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
                                break
                                
                    if len(complements) == 0:
                        print("Impossible to complement the prompt, pass")
                        compAvailable=True
                        attributeCrudoComp=None
                        

                if attributeCrudoComp != None:

                    if attributeCrudoComp == 'man':  # should be 0

                        newVector[i] = 0
                        
                    elif attributeCrudoComp == 'woman': # should be 1 

                        newVector[i] = 1
                        
                    else:
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
                assert self.getNumberOfAttributesFromVector(vector) == self.getNumberOfAttributesFromVector(newVector), "prompts complemented not match number of attributes\n "
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
            medChanged+=nAttributesToChange

            # get only idx that are into account
            idxs = self.random_indices_unique_filtered(vector, nAttributesToChange, seed=42)
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

                if attributeCrudo == FEMALE_ATTR:
                    
                    if vectorOrig[i] == 0: # means that is a man
                        attributeCrudoComp='woman'
                        complements=['woman']
                    else:
                        attributeCrudoComp='man'
                        complements=['man']
                        
                else:
                    complements=list(self.attr_complements[attributeCrudo])
                
         
                if len(complements) == 0:
                    print("Impossible to complement the prompt, pass")
                    attributeCrudoComp=None
                    compAvailable=True
                else:
                    compAvailable=False

                
                
                if attributeCrudoComp == 'man' or attributeCrudoComp == 'woman':
                    compAvailable=True
                    
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
                        

                if attributeCrudoComp != None:

                    if attributeCrudoComp == 'man':  # should be 0

                        newVector[i] = 0
                        
                    elif attributeCrudoComp == 'woman': # should be 1 

                        newVector[i] = 1
                        
                    else:
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

        attributesInPromptPos = self.getAttributesAtDS(vector)
        
        attributeCrudo=attribute

        attributesInPromptPos.remove(attributeCrudo)

        i=self.dataPkl['attr_name'].index(attributeCrudo)

        vector_np  = np.array(vector, dtype=int, copy=True)
        vectorOrig = vector_np.copy()     # snapshot to compare against
        newVector  = vector_np.copy()     # mutable working copy
        allNewVectors = []
        attributeCrudoComp=None

        if attributeCrudo == FEMALE_ATTR:
            
            if vectorOrig[i] == 0: # means that is a man
                attributeCrudoComp='woman'
                complements=['woman']
            else:
                attributeCrudoComp='man'
                complements=['man']
                
        else:
            complements=list(self.attr_complements[attributeCrudo])
        
    
        if len(complements) == 0:
            print("Impossible to complement the prompt, pass")
            attributeCrudoComp=None
            compAvailable=True
        else:
            compAvailable=False


        
        if attributeCrudoComp == 'man' or attributeCrudoComp == 'woman':
            compAvailable=True
            
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


        if attributeCrudoComp != None:

            if attributeCrudoComp == 'man':  # should be 0

                newVector[i] = 0
                
            elif attributeCrudoComp == 'woman': # should be 1 

                newVector[i] = 1

            return None, None, listImgs, 1, vectorOrig, [newVector.tolist()]

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

        if attributeCrudo == FEMALE_ATTR:
            
            if vectorOrig[i] == 0: # means that is a man
                attributeCrudoComp='woman'
                complements=['woman']
            else:
                attributeCrudoComp='man'
                complements=['man']
                
        else:
            complements=list(self.attr_complements[attributeCrudo])
        
    
        if len(complements) == 0:
            print("Impossible to complement the prompt, pass")
            attributeCrudoComp=None
            compAvailable=True
        else:
            compAvailable=False

        
        
        if attributeCrudoComp == 'man' or attributeCrudoComp == 'woman':
            compAvailable=True
            
        # get a complemented choice
        while compAvailable == False:
            compAvailable=True

            attributeCrudoComp = random.choice(complements)
            
            if attributeCrudoComp in attributesInPromptPos:
                #print("Attribute already complemented, searching new")
                #print(attributeCrudoComp)
                compAvailable=False
                complements.remove(attributeCrudoComp)
       
            if len(complements) == 0:
                print("Impossible to complement the prompt, pass")
                compAvailable=True
                attributeCrudoComp=None
                

        if attributeCrudoComp != None:

            if attributeCrudoComp == 'man':  # should be 0

                newVector[i] = 0
                
            elif attributeCrudoComp == 'woman': # should be 1 

                newVector[i] = 1
                
            else:
                # get the idx for the vector of that crudocomp
                idxAttributeCrudoComp = self.listAllAttrib.index(attributeCrudoComp)
                
                # swap the vectors
                newVector[idxAttributeCrudoComp]=1
                newVector[i]=0    
            
            return None, None, listImgs, 1, vectorOrig.tolist(), newVector.tolist()

        else:
            return None, None, None, None, None, None
        



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
                if not 'Femal' in listAttributesToChange:
                    listAttributesToChange.append('man')

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
    
