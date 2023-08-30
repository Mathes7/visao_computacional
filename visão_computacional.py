#detecção de faces.

import cv2 # OpenCV

imagem = cv2.imread('D:/Estudos Python/bancos de dados/workplace-1245776_1920.jpg')

cv2.imshow(imagem) #para visualizar a imagem.

detector_face = cv2.CascadeClassifier('D:/Estudos Python/bancos de dados/haarcascade_frontalface_default.xml') #banco com rostos já treinados.

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) #para mudar a imagem para cinza.
cv2.imshow(imagem_cinza)

deteccoes = detector_face.detecMultiScale(imagem_cinza, scaleFactor=1.3, minSize=(30,30)) #para gerar a detecção.
len(deteccoes)

for(x, y, l, a) in deteccoes:    #para visualiz\ar as detecções.
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,255,0), 2)
cv2.imshow(imagem)

#%%execício para detecção de corpo inteiro.

imagem = cv2.imread('D:/Estudos Python/bancos de dados/pessoas.jpg')

detector_pessoas = cv2.CascadeClassifier('D:/Estudos Python/bancos de dados/fullbody.xml') #banco com corpos já treinados.

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) #para mudar a imagem para cinza.

deteccoes = detector_face.detecMultiScale(imagem_cinza, scaleFactor=1.1, minSize=(50,50))#para gerar a detecção.
len(deteccoes)

for(x, y, l, a) in deteccoes:    #para visualiz\ar as detecções.
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,255,0), 2)
cv2.imshow(imagem)

#%%reconhecimento facil.
from PIL import Image
import numpy as np

# obs:
# para puxar arquivos do drive, usando o google colab.
# from google.colab import drive
# drive.mount('/content/drive)

# para extrair um arquivo zip.
# import zipfile 
# path = 'D:/Estudos Python/bancos de dados/yalefaces.zip'
# zip_object = zipfile.ZipFile(file=path, mode='r')
# zip_object.extractall('./')
# zip_object.close()

import os
os.listdir('D:/Estudos Python/bancos de dados/yalefaces/train')

#
def dados_imagem():
  caminhos = [os.path.join('D:/Estudos Python/bancos de dados/yalefaces/train', f) for f in os.listdir('D:/Estudos Python/bancos de dados/yalefaces/train')]
  faces = []  
  ids = []
  for caminho in caminhos:
    imagem = Imagem.open(caminho).convert('L')
    imagem_np = np.array(imagem, 'uint8')
    id = int(os.path,split(caminho)[1].split('.')[0].replace('subject',''))
    ids.append(id)
    faces.append(imagem_np)
  return np.array(ids), faces 
    
ids, faces = dados_imagem()

lbph = cv2.face.LBPHFaceRecognize_create()
lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

#classificação.
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read('/content/classificadorLBPH.yml')

imagem_teste = 'D:/Estudos Python/bancos de dados/yalefaces/test/subject10.sad.gif'
imagem = Imagem.open(imagem_teste).convert('L')
imagem_np = np.array(imagem, 'uint8')
print(imagem_np)

idprevisto, _ = reconhecedor.predict(imagem_np)
idprevisto

idcorreto = int(os.path,split(imagem_teste)[1].split('.')[0].replace('subject',''))
idcorreto

cv2.putText(imagem_np, 'P: ' + str(idprevisto), (x,y + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
cv2.putText(imagem_np, 'c: ' + str(idcorreto), (x,y + 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
cv2.imshow(imagem_np)

#%% rastreamento deobjetos.
import cv2

rastreador = cv2.TrackerCSRT_create()

video = cv2.VideoCapture('rua.mp4')
ok, frame = video.read()

bbox = cv2.selectROI(frame)
print(bbox)

ok = rastreador.init(frame, bbox)
print(ok)

while True:
    ok, frame = video.read()
    if not ok:
        break
    
    ok, bbox = rastreador.update(frame)
    print(bbox)
    
    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2, 1)
    else:
        cv2.putText(frame, 'Falha no rastreamento', (100,80),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255), 2)
        
    cv2.imshow('Rastreando', frame)
    if cv2.waitKey(1) & OXFF ==27:
        break
    
