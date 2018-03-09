import glob
import json
import torch
import numpy as np
import datetime, time

def Main():
    top_path = "top_B.parse"
    u20_path = "u20_B.parse"

    list_document, top_document_obj = GetDocumentObj(top_path)
    time_tensor, loc_tensor, person_organ_tensor, morph_tensor = GetTensor(list_document, top_document_obj)
    print(time_tensor.shape, loc_tensor.shape, person_organ_tensor.shape, morph_tensor.shape)
    # list_document, u20_document_obj = GetDocumentObj(u20_path)
    # time_tensor, loc_tensor, person_organ_tensor, morph_tensor = GetTensor(list_document, u20_document_obj)

class DocumentObj(object):
    def __init__(self, time, location, person_organ, morph):        
        self.time = time
        self.location = location
        self.person_organ = person_organ
        self.morph = morph

class Dictionary(object):
    def __init__(self):
        self.element2idx = {}
        self.idx2element = {}
        self.idx_in_element = 0
    def add_element(self, element):
        if not element in self.element2idx:
            self.element2idx[element] = self.idx_in_element
            self.idx2element[self.idx_in_element] = element
            self.idx_in_element += 1

class DictObj(object):
    def __init__(self):
        self.time = Dictionary()
        self.location = Dictionary()
        self.person_organ = Dictionary()        
        self.morph = Dictionary()

    def SetDict(self, list_document):
        for doc_element in list_document:
            for loc_element in doc_element.location:
                self.location.add_element(loc_element)
            for person_organ_element in doc_element.person_organ:
                self.person_organ.add_element(person_organ_element)
            for morph_element in doc_element.morph:
                self.morph.add_element(morph_element)

# x : interation 돌리는 document 시간, mu : 기준 document 시간, sig : 5는 5일 차이까지 0.5이상
def gaussian(time_delta, sig=10):
    # x_mu = x - mu
    x_mu = time_delta
    return np.exp(-np.power(x_mu, 2.) / (2 * np.power(sig, 2.)))

def GetDocumentObj(file_path):
    list_document = []
    
    for e in sorted(glob.glob(file_path + "/*")):
        start_idx = e.find("/")
        end_idx = e[start_idx:].find("_") + start_idx
        time_element = e[start_idx+1:end_idx]
        list_person_organization = []    
        list_location = []
        list_morph = []

        read_obj = open(e,"r",encoding="utf-8").read()
        for e_sub in read_obj.split("\n\n"):
            if len(e_sub) == 0:
                continue
            json_e_sub = json.loads(e_sub)
            if len(json_e_sub["sentence"]) == 0:
                continue

            for sentence_element in json_e_sub["sentence"]:
                for morph_element in sentence_element["morp"]:
                    if morph_element["type"].startswith("NNG") == True or morph_element["type"].startswith("NNP") == True:
                        list_morph.append(morph_element["lemma"])
                for ne_element in sentence_element["NE"]:
                    if ne_element["type"].startswith("PS") == True or ne_element["type"].startswith("OG") == True:
                        list_person_organization.append(ne_element["text"])
                    if ne_element["type"].startswith("LC") == True:
                        list_location.append(ne_element["text"])
    #     print("[**] time ",time_element)
    #     print("[**] location ",list_location)
    #     print("[**] person_organization ",list_person_organization)
    #     print("[**] morphology ",list_morph)

        set_person_organization = set(list_person_organization)
        for person_organization_e in set_person_organization:
            check_value = person_organization_e in list_morph
            while check_value == True:
                list_morph.remove(person_organization_e)
                check_value = person_organization_e in list_morph
    #     print("[**] morphology after preprocess ",list_morph)

        doc_obj_element = DocumentObj(time_element, list_location, list_person_organization, list_morph)
        list_document.append(doc_obj_element)

    documents_obj = DictObj()
    documents_obj.SetDict(list_document)
    
    return list_document, documents_obj

def GetTensor(list_document, documents_obj):
    splited_time_0 = list_document[0].time.split("-")
    datetime_0 = datetime.date(int(splited_time_0[0]), int(splited_time_0[1]), int(splited_time_0[2]))
    splited_time_1 = list_document[-1].time.split("-")
    datetime_1 = datetime.date(int(splited_time_1[0]), int(splited_time_1[1]), int(splited_time_1[2]))

    start_value = datetime_0 - datetime_1
    end_value = datetime_1 - datetime_0
    gauss_dict = {i:gaussian(i) for i in range(start_value.days, end_value.days+1)}
        
    # shape : documnet 개수 x 1 [TEMP]
    time_tensor = torch.FloatTensor(len(list_document),end_value.days+1).zero_()
    # shape : documnet 개수 x location 사전 개수
    loc_tensor = torch.IntTensor(len(list_document),len(documents_obj.location.element2idx)).zero_()
    # shape : documnet 개수 x person_organ 사전 개수
    person_organ_tensor = torch.IntTensor(len(list_document),len(documents_obj.person_organ.element2idx)).zero_()
    # shape : documnet 개수 x morph 사전 개수
    morph_tensor = torch.IntTensor(len(list_document),len(documents_obj.morph.element2idx)).zero_()
    
    for i, doc_element in enumerate(list_document):
        temp_splited_time = doc_element.time.split("-")
        temp_datetime = datetime.date(int(temp_splited_time[0]), int(temp_splited_time[1]), int(temp_splited_time[2]))
        for j in range(end_value.days+1):
            time_tensor[i][j] = gauss_dict[(datetime_0 - temp_datetime).days + j]
        
        set_location = set(doc_element.location)
        for loc_element in set_location:
            j = documents_obj.location.element2idx[loc_element]
            loc_tensor[i][j] = doc_element.location.count(loc_element)

        set_person_organ = set(doc_element.person_organ)
        for person_organ_element in set_person_organ:
            j = documents_obj.person_organ.element2idx[person_organ_element]
            person_organ_tensor[i][j] = doc_element.person_organ.count(person_organ_element)

        set_morph = set(doc_element.morph)
        for morph_element in set_morph:
            j = documents_obj.morph.element2idx[morph_element]
            morph_tensor[i][j] = doc_element.morph.count(morph_element)
    return time_tensor, loc_tensor, person_organ_tensor, morph_tensor

if __name__ == "__main__":
    Main()
