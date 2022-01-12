## Este é o trabalho de cognição visual de Marcelo Pedro e Lucas Luppi.

## Instalar:
Para baixar o código e o relatório desse trabalho basta dá o comando num terminal linux:

```bash
git pull https://github.com/marcelo-ped/trabalho_cognicao_viusal_2021-2.git
```

Caso não tenha o git instalado na sua máquina dê o comando abaixo:

```bash
sudo apt-get install git
```
Para instalar os pacotes necessários deste trabalho basta dá o comando:

```bash
pip install -r requirements.txt
```

Esse trabalho foi executado em python na versão 3.6, caso você tenha algum problema com a instalação, instalar a versão 3.6 pode resolver o problema. Para isso basta dar os seguintes comando em um terminal:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.6
```

## Resultados deste trabalho:

Em cada pasta de nome Resultado_experimento_1,2,3 ou 4 contém os arquivos confusion_mat.npy,sendo a matriz e confusão do experimento correspondente, try_hist.csv, sendo o arquivo que descreve o histórico da acurácia e loss da rede durante o experimento executado e o arquivo rede_MobileVGG.h5, sendo a rede final do experimento.

## Aviso de como deve ser o conteúdo dos arquivos .csv para treino ou teste da rede deste trabalho:

Em cada linha dos arquivos csv deve conter o caminho completo para uma imagem do dataset, seguida de um número que corresponde a classe dessa imagem. Um exemplo dos arquivos train.csv, valid.csv e test.csv pode ser encontrado na pasta dataset, junto com as imagens do dataset do experimento 1, que pode ser usado também para treino ou teste da rede deste trabalho. A rede já treinada para esse exemplo encontra-se dentro da pasta Resultado_experimento_1.

## Como treinar:

Para treinar a rede deste trabalho do zero basta dá o comando em um terminal:

```bash
python mobileVGG.py --dataset PATH_DATASET
```

Sendo PATH_DATASET o caminho da pasta que contém os arquivos train.csv e valid.csv. 

## Como testar:

Para testar a rede desse trabalho basta dá o comando em um terminal:

```bash
python mobileVGG.py --dataset PATH_DATASET --test PATH_WEIGHT
```

Sendo PATH_DATASET o caminho da pasta que contém o arquivo test.csv e PATH_WEIGHT o caminho completo para o arquivo .h5 da rede que você quer testá-la.


## Como baixar os datasets utilizados neste trabalho:
Para baixar o dataset utilizado no experimento 1, basta seguir as instruções deste site:
[dataset experimento 1](https://abouelnaga.io/distracted-driver/)
Para baixar o dataset utilizado no experimento 2 entre no link abaixo:
[dataset experimento 2](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)
O dataset utlizado para o experimento 3 está no link abaixo:
[dataset experimento 3](https://drive.google.com/file/d/1V5aIZhQJr7C_-eB94xpxMoT0MyrxlB-D/view?usp=sharing)
O dataset utlizado para o experimento 4 está no link abaixo:
[dataset experimento 4](https://drive.google.com/file/d/1zt73vtXHK4P2xMO7jvkbIibyFYR09jJB/view?usp=sharing)
