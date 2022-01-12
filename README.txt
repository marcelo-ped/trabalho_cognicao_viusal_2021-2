Este é o trabalho de cognição visual de Marcelo Pedro e Lucas Luppi.
Para baixar o código e o relatório desse trabalho basta dá o comando num terminal linux:
git pull https://github.com/marcelo-ped/trabalho_cognicao_viusal_2021-2.git
Caso não tenha o git instalado na sua máquina dê o comando abaixo:
sudo apt-get install git
Para instalar e compilar o código deste trabalho basta dá o comando:
pip install -r requirements.txt
Em cada pasta de nome Resultado_experimento_1,2,3 ou 4 contém os arquivos confusion_mat.npy,sendo a matriz e confusão do experimento correspondente, try_hist.csv, sendo o arquivo que descreve o histórico da acurácia e loss da rede durante o experimento executado e o arquivo rede_MobileVGG.h5, sendo a rede final do experimento.
Para treinar a rede deste trabalho do zero basta dá o comando:
python mobileVGG.py --dataset PATH_DATASET
Sendo PATH_DATASET o caminho da pasta que contém os arquivos train.csv e valid.csv. 
Para testar a rede desse trabalho basta dá o comando:
python mobileVGG.py --dataset PATH_DATASET --test PATH_WEIGHT
Sendo PATH_DATASET o caminho da pasta que contém o arquivo test.csv e PATH_WEIGHT o caminho completo para o arquivo .h5 da rede que você quer testá-la.
Em cada linha dos arquivos csv deve conter o caminho completo para uma imagem do dataset, seguida pôr um número que coresponde a classe dessa imagem
Para baixar o dataset utilizado basta seguir as instruções deste site:
https://abouelnaga.io/distracted-driver/
Para baixar o dataset do experimento entre no link abaixo:
https://www.kaggle.com/c/state-farm-distracted-driver-detection/data
O dataset utlizado para o experimento 3 está no link abaixo:
https://drive.google.com/file/d/1V5aIZhQJr7C_-eB94xpxMoT0MyrxlB-D/view?usp=sharing
O dataset utlizado para o experimento 4 está no link abaixo:
https://drive.google.com/file/d/1zt73vtXHK4P2xMO7jvkbIibyFYR09jJB/view?usp=sharing
