from os import path
from urllib.parse import unquote

import chardet


doc_toc_list :list[tuple] = [
    (r"S:\SourceCode\WindowsCloud.wiki\Components\Hybrid-Services\AKS-Hybrid\\",".order","https://supportability.visualstudio.com/WindowsCloud/_wiki/wikis/WindowsCloud.wiki?wikiVersion=GBwikiMaster&pagePath=/Components/Hybrid-Services/AKS-Hybrid/"),
]

def copy_file(file_path: str, title: str, link: str):

    rawdata = open(file_path, 'rb').read()
    encoding = chardet.detect(rawdata)['encoding']
    print(encoding)

    file = open(file_path, "r",encoding=encoding)
    decode_title = unquote(title).replace("-"," ")
    lines = [line for line in file]
    new_file_path = path.join(path.dirname(path.realpath(__file__)),"..\\..\\data\\msk8s", title + ".md")
    new_file = open(new_file_path, "w",encoding="utf-8")
    new_file.write(f"# {decode_title}\n\n")
    new_file.write(f"Link: [{link}]({link})\n\n")
    for line in lines:
        new_file.write(line + "\n")
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
                    copy_file(doc_file_path, line, toc_path[2] + line)
