import json
from os import path
import os
from urllib.parse import unquote

import chardet


doc_toc_list :list[tuple] = [
    (r"C:\Users\enyao\SouceCode\WindowsCloud.wiki\Components\Hybrid-Services\AKS-Hybrid\\",
     ".order",
     "https://supportability.visualstudio.com/WindowsCloud/_wiki/wikis/WindowsCloud.wiki?wikiVersion=GBwikiMaster&pagePath=/Components/Hybrid-Services/AKS-Hybrid/",
     "csswiki",
     [],
     [],
     ),
    (r"C:\Users\enyao\SouceCode\msk8s.wiki\ICM-%2D-TSG\AKS-Arc-%2D-ASZ\\",
     ".order",
     "https://msazure.visualstudio.com/msk8s/_wiki/wikis/msk8s.wiki?wikiVersion=GBwikiMaster&pagePath=/ICM%20%252D%20TSG/AKS%20Arc%20%252D%20ASZ/",
     "devwiki",
     [],
     ["39c09a25-635f-4d56-a4b0-5ce296ee3fd7","db13c5be-ea54-4018-9a9e-c1c7077afb2f"]),
]

def copy_file(file_path: str, target_folder:str, title: str, link: str, acl_oid:list, acl_group:list):

    rawdata = open(file_path, 'rb').read()
    encoding = chardet.detect(rawdata)['encoding']
    print(encoding)

    file = open(file_path, "r",encoding=encoding)
    decode_title = unquote(title).replace("-"," ")
    lines = [line for line in file]

    directory = path.join(path.dirname(path.realpath(__file__)),"..\\..\\data\\msk8s", target_folder)
    if not path.exists(directory):
        os.makedirs(directory)

    filename = title.replace("[","I").replace("]","I")
    new_file_path = path.join(path.dirname(path.realpath(__file__)),"..\\..\\data\\msk8s", target_folder, filename + ".md")
    if path.isfile(new_file_path):
        print(f"Warning: {new_file_path} already exist, will be overwrite.")

    new_file = open(new_file_path, "w",encoding="utf-8")
    new_file.write(f"# {decode_title}\n\n")
    new_file.write(f"Link: [{link}]({link})\n\n")
    for line in lines:
        new_file.write(line + "\n")
    new_file.close()

    acl = { 
        "oids": acl_oid,
        "groups": acl_group,
    }
    new_file_acl_path = new_file_path+".acl"
    new_file = open(new_file_acl_path, "w",encoding="utf-8")
    new_file.write(json.dumps(acl))
    new_file.close()

if __name__ == "__main__":
    for toc_path in doc_toc_list:
        root_path = toc_path[0]
        file = open(path.join(root_path, toc_path[1]), "r")
        lines = [line.rstrip() for line in file]
        for line in lines:
            if line != '':
                doc_file_path = path.join(root_path, line + ".md")
                print(f"Begin to copy {doc_file_path}")
                if path.isfile(doc_file_path):
                    print(f"File {doc_file_path} exists")
                    copy_file(doc_file_path, toc_path[3], line, toc_path[2] + line, toc_path[4], toc_path[5])
