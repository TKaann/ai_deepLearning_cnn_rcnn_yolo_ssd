#yoloya bir nesne dedktoru olarak bakabiliriz. nesne dedektoru dedigimiz sey nesneyi bulur ve nesneyi tanimlar yani nesneyi
#siniflandirir. nesne algilama icin evrisimli sinir aglarini kullanir ayni rcnn de oldugu gibi.
#yoloV3 en yeni yolo cesididir.
#gonruntu veya videolarda 80 farkli nesneyi tanir ve super hizlidir.
#yoloyu bize verilen kutuphanesi ile kullanicaz.



import cv2
import numpy as np

#yolo modelinden yoloyu cekiyoruz.
from yolo_model import YOLO


#literaturde kullanilan 2 tane parametre var 0.6 ve 0.5 bunlari kullaniyortuzx.
yolo = YOLO(0.6, 0.5)
#burada da yolo ile siniflandirilan etiketler var onu cekiyoruz.
file = "data/coco_classes.txt"

with open(file) as f:
    class_name = f.readlines()

#aralarindaki bosluklari siliyoruz \n leri.
all_classes = [c.strip() for c in class_name]

#simdi siniflandiracagimiz ve nesne tespiti yapacagimiz yere gecelim.
#oncelikle reimlerimizi import ediyoruz.
f = "dog_cat.jpg"
path = "images/"+f
image = cv2.imread(path)
cv2.imshow("image",image)

#simdi preprocces yapicaz, bu resim buyuk ve bu resmi yolo icin uygun hale getiricez. 416 ya 416 yolonun istedigi size
pimage = cv2.resize(image, (416,416))
#bu bir resim oldguu icin resmi array e ceviriyoruz.
pimage = np.array(pimage, dtype = "float32")
#resmi normalize ediyoruz.
pimage /= 255.0
#burda da yolo icin gerekli olan satir icine bir [] liste daha ekliyoruz.
pimage = np.expand_dims(pimage, axis = 0)


#yolo ile predict islemini yapalim.
#bu bize 3 tane sey veriyor 1. olarak nesneleri cevreleyen kutuck 2. olarak tespit edilen nesnelerin siniflari
#3. olarak tahmin yuzdeleri yani yuzde kac olasilikla bu nesne oldugu.
boxes, classes, scores = yolo.predict(pimage, image.shape)
#cikti olarak 2 tane kutucuk ve 16 ve 15 classlari aldik.


#simdi burda ise yaptigimiz isi gorsellestiriyoruz. burda box score ve cl donduruyoruz.
for box, score, cl in zip(boxes, scores, classes):
    #kutucuk bize x y kordinatlari w ve h donduruyor.
    x,y,w,h = box
    
    #burda hafif bir pay birakiyoruz bunu yapmasak da olur ama yaptigimiz icin biraz daha guzel durucak.
    #burdaki floor ise sayilari asagiya yuvarliyor. yani 1.5 ise 1 yapiyor gibi.
    top = max(0, np.floor(x + 0.5).astype(int))
    left = max(0, np.floor(y + 0.5).astype(int))
    right = max(0, np.floor(x + w + 0.5).astype(int))
    bottom = max(0, np.floor(y + h + 0.5).astype(int))

    
    #rect i img nin ustune cizdiriyoruz.
    cv2.rectangle(image, (top,left), (right, bottom),(255,0,0),2)
    #burda da cektigi klaslari yazdiriyoruz resmin ustune. ve score u yazdiriyoruz yani yuzde kac o cikmis.
    cv2.putText(image, "{} {}".format(all_classes[cl],score),(top,left-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1,cv2.LINE_AA)
    
#gorselimizi cizdiriyoruz.
cv2.imshow("yolo",image)    



#bunu yolo kutuphanesinin icinde olan nesneler icin de kameramizi acip yapabiliriz. while dongusu icine acip gercek zamanli.













































































