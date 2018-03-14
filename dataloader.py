import glob
import json
import torch
import numpy as np
import datetime, time

if __name__ == '__main__':
    top_path = "top_B.parse"
    u20_path = "u20_B.parse"

    label_dict = {top_path:0, u20_path:1}
    list_document = []; start_date = None ; end_date = None
    list_document, start_date, end_date = GetDocumentObj(list_document, top_path, label_dict[top_path], start_date, end_date)
    list_document, start_date, end_date = GetDocumentObj(list_document, u20_path, label_dict[u20_path], start_date, end_date)

    documents_obj = DictObj()
    documents_obj.SetDict(list_document)    
    time_tensor, loc_tensor, person_organ_tensor, morph_tensor, list_label = GetTensor(list_document, documents_obj, start_date, end_date)	
    print(time_tensor, loc_tensor, person_organ_tensor, morph_tensor, len(list_label))
    

class DocumentObj(object):
    def __init__(self, time, location, person_organ, morph, label):        
        self.time = time
        self.location = location
        self.person_organ = person_organ
        self.morph = morph
        self.label = label

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

def GetDocumentObj(list_document, file_path, label, start_date, end_date):
    list_temp_document = []
        
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
        
        # Morph안에서 Location의 중복 제거
        set_location = set(list_location)
        for location_e in set_location:
            check_value = location_e in list_morph
            while check_value == True:
                list_morph.remove(location_e)
                check_value = location_e in list_morph

        # Morph안에서 Person Organization의 중복 제거
        set_person_organization = set(list_person_organization)
        for person_organization_e in set_person_organization:
            check_value = person_organization_e in list_morph
            while check_value == True:
                list_morph.remove(person_organization_e)
                check_value = person_organization_e in list_morph
    #     print("[**] morphology after preprocess ",list_morph)

        doc_obj_element = DocumentObj(time_element, list_location, list_person_organization, list_morph, label)
        list_temp_document.append(doc_obj_element)
        
    splited_time_0 = list_temp_document[0].time.split("-")
    temp_start_date = datetime.date(int(splited_time_0[0]), int(splited_time_0[1]), int(splited_time_0[2]))
    if start_date == None:
        start_date = temp_start_date
    elif temp_start_date < start_date:
        start_date = temp_start_date
        
    splited_time_1 = list_temp_document[-1].time.split("-")
    temp_end_date = datetime.date(int(splited_time_1[0]), int(splited_time_1[1]), int(splited_time_1[2]))
    if end_date == None:
        end_date = temp_end_date
    elif end_date < temp_end_date:
        end_date = temp_end_date
        
    list_document.extend(list_temp_document)

    return list_document, start_date, end_date

def GetTensor(list_document, documents_obj, start_date, end_date):
    start_value = start_date - end_date
    end_value = end_date - start_date
    gauss_dict = {i:gaussian(i) for i in range(start_value.days, end_value.days+1)}
        
    # shape : documnet 개수 x 1 [TEMP]
    time_tensor = torch.FloatTensor(len(list_document),end_value.days+1).zero_()
    # shape : documnet 개수 x location 사전 개수
    loc_tensor = torch.FloatTensor(len(list_document),len(documents_obj.location.element2idx)).zero_()
    # shape : documnet 개수 x person_organ 사전 개수
    person_organ_tensor = torch.FloatTensor(len(list_document),len(documents_obj.person_organ.element2idx)).zero_()
    # shape : documnet 개수 x morph 사전 개수
    morph_tensor = torch.FloatTensor(len(list_document),len(documents_obj.morph.element2idx)).zero_()
    
    list_label = []
    for i, doc_element in enumerate(list_document):
        temp_splited_time = doc_element.time.split("-")
        temp_datetime = datetime.date(int(temp_splited_time[0]), int(temp_splited_time[1]), int(temp_splited_time[2]))
        for j in range(end_value.days+1):
            time_tensor[i][j] = gauss_dict[(start_date - temp_datetime).days + j]
        
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
        list_label.append(doc_element.label)
        
    return time_tensor, loc_tensor, person_organ_tensor, morph_tensor, list_label