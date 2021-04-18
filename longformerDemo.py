from transformers import AutoModel, AutoTokenizer, pipeline
import torch

model_name = 'pre-model/' + 'longformer-encdec-base-16384'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
classifier = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

# encoded_inputs = tokenizer(["ATGCATGCNACT"], ["ATGCATGCNACT"], return_token_type_ids=True, return_tensors='pt')
encoded_inputs = tokenizer(["ATGCATGCNACT" * 2000, "ATGCATG" * 2000, "ACTGGTCATGCAC" * 500], return_tensors='pt',
                           padding=True)
print(encoded_inputs)
# feature = model(input_ids=encoded_inputs['input_ids'], attention_mask=encoded_inputs['attention_mask'],
#                 return_netsors='pt')
feature = model(**encoded_inputs,
                return_netsors='pt')
print(feature[0])
print(type(feature[0]))
# feature = torch.as_tensor(feature)
# print(feature.shape)
print("***" * 48)

feature = classifier(["ATG" * 1000, "ATGCATG" * 1000, "ACTGGTCATGCAC" * 300])
print(type(feature))
feature = torch.as_tensor(feature)
print(feature)
print(feature.shape)
print("***" * 48)

'''
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
absl-py                   0.12.0                   pypi_0    pypi
astunparse                1.6.3                      py_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
blas                      1.0                         mkl    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
boto3                     1.17.53                  pypi_0    pypi
botocore                  1.20.53                  pypi_0    pypi
brotlipy                  0.7.0           py36h27cfd23_1003    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ca-certificates           2021.4.13            h06a4308_1  
cachetools                4.2.1                    pypi_0    pypi
certifi                   2020.12.5        py36h06a4308_0  
cffi                      1.14.5           py36h261ae71_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
chardet                   4.0.0           py36h06a4308_1003    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
click                     7.1.2              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
cryptography              3.4.7            py36hd23ed53_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
cudatoolkit               10.0.130                      0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
dataclasses               0.8                pyh4f3eec9_6    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
dill                      0.3.3                    pypi_0    pypi
filelock                  3.0.12             pyhd3eb1b0_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
freetype                  2.10.4               h5ab3b9f_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
future                    0.18.2                   pypi_0    pypi
google-auth               1.29.0                   pypi_0    pypi
google-auth-oauthlib      0.4.4                    pypi_0    pypi
grpcio                    1.37.0                   pypi_0    pypi
idna                      2.10               pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
imageio                   2.9.0                    pypi_0    pypi
importlib-metadata        2.0.0                      py_1    anaconda
intel-openmp              2020.2                      254    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
jmespath                  0.10.0                   pypi_0    pypi
joblib                    1.0.1              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
jpeg                      9b                   h024ee3a_2    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
lcms2                     2.12                 h3be6417_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ld_impl_linux-64          2.33.1               h53a641e_7    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libffi                    3.3                  he6710b0_2    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libgcc-ng                 9.1.0                hdf63c60_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libpng                    1.6.37               hbc83047_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libprotobuf               3.14.0               h8c45485_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libstdcxx-ng              9.1.0                hdf63c60_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libtiff                   4.1.0                h2733197_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
longformer                0.1                      pypi_0    pypi
lz4-c                     1.9.3                h2531618_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
markdown                  3.3.4                    pypi_0    pypi
mkl                       2020.2                      256    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
mkl-service               2.3.0            py36he8ac12f_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
mkl_fft                   1.3.0            py36h54f3939_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
mkl_random                1.1.1            py36h0573a6f_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ncurses                   6.2                  he6710b0_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ninja                     1.10.2           py36hff7bd54_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
nlp                       0.4.0                    pypi_0    pypi
nltk                      3.6.1                    pypi_0    pypi
numpy                     1.19.2           py36h54aff64_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
numpy-base                1.19.2           py36hfa32c7d_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
oauthlib                  3.1.0                    pypi_0    pypi
olefile                   0.46                     py36_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
openssl                   1.1.1k               h27cfd23_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
packaging                 20.9               pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pandas                    1.1.5                    pypi_0    pypi
pillow                    8.2.0            py36he98fc37_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pip                       21.0.1           py36h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
protobuf                  3.15.8                   pypi_0    pypi
pyarrow                   3.0.0                    pypi_0    pypi
pyasn1                    0.4.8                    pypi_0    pypi
pyasn1-modules            0.2.8                    pypi_0    pypi
pycparser                 2.20                       py_2    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pyopenssl                 20.0.1             pyhd3eb1b0_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pyparsing                 2.4.7              pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pysocks                   1.7.1            py36h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
python                    3.6.13               hdb3f193_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
python-dateutil           2.8.1                    pypi_0    pypi
python_abi                3.6                     1_cp36m    huggingface
pytorch-lightning         0.8.5                    pypi_0    pypi
pytorch-transformers      1.2.0                    pypi_0    pypi
pytz                      2021.1                   pypi_0    pypi
pyyaml                    5.4.1                    pypi_0    pypi
readline                  8.1                  h27cfd23_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
regex                     2021.4.4         py36h27cfd23_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
requests                  2.25.1             pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
requests-oauthlib         1.3.0                    pypi_0    pypi
rouge-score               0.0.4                    pypi_0    pypi
rsa                       4.7.2                    pypi_0    pypi
s3transfer                0.3.7                    pypi_0    pypi
sacremoses                0.0.44                   pypi_0    pypi
sentencepiece             0.1.95                   pypi_0    pypi
setuptools                52.0.0           py36h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
six                       1.15.0           py36h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
sqlite                    3.35.4               hdfb4753_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
tensorboard               2.4.1                    pypi_0    pypi
tensorboard-plugin-wit    1.8.0                    pypi_0    pypi
tensorboardx              2.2                      pypi_0    pypi
test-tube                 0.7.5                    pypi_0    pypi
tk                        8.6.10               hbc83047_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
tokenizers                0.8.1rc2                 pypi_0    pypi
torch                     1.6.0                    pypi_0    pypi
torchvision               0.5.0                py36_cu100    pytorch
tqdm                      4.60.0                   pypi_0    pypi
transformers              3.1.0                    pypi_0    pypi
urllib3                   1.26.4             pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
werkzeug                  1.0.1                    pypi_0    pypi
wheel                     0.36.2             pyhd3eb1b0_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
xxhash                    2.0.2                    pypi_0    pypi
xz                        5.2.5                h7b6447c_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
zipp                      3.4.1              pyhd3eb1b0_0  
zlib                      1.2.11               h7b6447c_3    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
zstd                      1.4.9                haebb681_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main

'''