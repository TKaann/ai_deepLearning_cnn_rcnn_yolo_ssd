from tensorflow.keras.applications.resnet50 import preprocess_input 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2

#burdaki 3 fronksiyon metod bizim onceki sayfalarda yazdigimiz fonksiyonlar & metodlar. onlari import ediyoruz.
from sliding_window import sliding_window
from image_pyramid import image_pyramid
from non_max_supression import non_max_suppression



#ilklendirme parametrelerini belirliyoruz.
WIDTH = 600
HEIGHT = 600
#bu da pyramid metodumuzun scale factor u yani olcutu.
PYR_SCALE = 1.5
#bu da sliding windowdaki kac tane piksel atlayip gezinecegini belirleyen paremetremiz.
WIN_STEP = 16
ROI_SIZE = (200,150)
#resnet neural network unu 244 e 244 girdi alacak sekilde egitmisler o yuzden girdi size imizin 224 e 224 olmasi gerekli.
INPUT_SIZE = (224, 224)

print("Resnet yukleniyor")
#weightleri resnetin egitildigi veri setine gore seciyoruz.
model = ResNet50(weights = "imagenet", include_top = True)

#sudan resmimizi 600 e 600 yapiyoruz cunku ilk once remimizi pyramid ve sliding window a sokucaz, o yuzden direk 224 e 224
#yapmiyoruz o zaman bir mantigi kalmiyor, ilk once ilk asamalarimizi yapip ondan sonra modele sokarken 224 e 224 yapicaz.
#direk kucultursek bilgi kaybi yasariz.
orig = cv2.imread("husky.jpg")
orig = cv2.resize(orig, dsize = (WIDTH, HEIGHT))
cv2.imshow("Husky",orig)

#orig shapenin ilk ikisi bize 600 e 600 u veriyor onlari da h ve w icine atiyoruz.
(H,W) = orig.shape[:2]




#image pyramid ile basliyoruz. image pyramid ile resimlerin skalasini olcegini degistiriyoruz.
#simdi burada bir generator olusturup resimlerin skalasinin degismis halini bir resim olarak degil de generatore atiyoruz.
#cunku generatore atmazsak bu memory de kalacagi icin bu da hafiza icin dezavantaj oluyor.
pyramid = image_pyramid(orig, PYR_SCALE, ROI_SIZE)


#burada her bir pyramid iterasyonunda bir de sliding window kosturucaz cunku pyramid bize yeni bir resim veriyor ve bu resmin
#uzerinde nesne tespit ederken her bir resmin uzerinde window yonetemi kullanarak her bir pencerenin icerisndeki resmi 
#siniflandiriyor. bunun sonucunda da ortaya bizim belli basli degerlerimiz cikiyor(region of interest ve lokasyonu gibi),  
#degerler cikicak ve bu degerlerle roi ve locs diye listeler olusturuyorum ve cikan degerlerle dolduruyorum.
rois = []
locs = []

for image in pyramid:
    
    #bizim pyramid scale de uyguladigimiz skalayi windowda da uygulamamiz gerekli cunku ikisi arasinda uyumsuzluk olmasin.
    scale = W/float(image.shape[1])
    
    #burda bize x y ve resmimizin kendini return ediyor.
    for (x,y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
        
        x = int(x*scale)
        y = int(y*scale)
        w = int(ROI_SIZE[0]*scale)
        h = int(ROI_SIZE[1]*scale)
        
        #simdi burada elde ettigimiz resimleri preprocces etmemiz gerekli. resnet icinde kendi procces komutu var.
        roi = cv2.resize(roiOrig, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
    
        rois.append(roi)
        locs.append((x,y,x+w,y+h))
        
#roileri islemden gecirebilmek icin numpy arraylere ceviriyoruz.
rois = np.array(rois, dtype = "float32")



#simdi siniflandirma islemine geciyoruz.
#siniflandirma isleminden sonra ise burdaki predictlere gore hangisinin duzgun predict yaptigini hangisinin yapamadigina gore
#olasilik degeri, thresholdu koyucaz. o ortaya cikan degere gore de (ornk 0,85) bunun ustundeyse istedigimizdir diyip 
#kutulama boxing islemine gecis yapicaz.
print("sınıflandırma işlemi")
preds = model.predict(rois)

#predickler kendi basina pek bir sey ifade etmiyor, bu yuzden bunlari decode etmemiz gerekli. 
#kerasin kendi icinde olan decode islemini yapiyoruz burada da. bunlar resnet ile ilgili seyler.
preds = imagenet_utils.decode_predictions(preds, top = 1)


#simdi burada siniflandirdik ama 0.10 olasilikla kopektir diyor bazilari bazilari 0.5 ihtimalle diyor, bunlari siniflandirmak
#icin min conf degerimizi olusturuyoruz.
#labels ile tahmin olarak ne tahmin etmis onlari aliaz o yuzden olusturduk.
labels = {}
min_conf = 0.9

#burada predictleri ayirma islemi yapiyoruz.
for (i,p) in enumerate(preds):
    
    #burada ilk olarak resmin id sini donduruyor,o bizim icin onemli degil o yuzden bos donduruyoruz
    #2. olarak da tahmin ettigi seyi donduruyor onu aliyoruz label olarak, 3. olarakda tahmin yuzdesi onu da aliyoruz.
    (_, label, prob) = p[0]
    
    #ihtimali bizim belirledigimiz degerden buyuk olanlari aliyoruz sadece.
    if prob >= min_conf:
        
        #kutu icin daha once belirledigimiz locs in i. degerini aliyoruz.
        box = locs[i]
        
        #burda da 0.9 dan buyuk olanlari labels in icin aliyoruz.
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L


#burada labellari aliyoruz, 0.9 yaptigimiz icin de sadece eskimo dog geldi buraya.
for label in labels.keys():
    
    #orginal resmi bozmamak icin clone olusturuyoruz.
    clone = orig.copy()
    
    #kutucuk cizdirme islemi yapiyoruz.
    for (box, prob) in labels[label]:
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY),(endX, endY), (0,255,0),2)
    
    cv2.imshow("ilk",clone)
    
    #sudanda yanlis buluyor, zaten ayalarini aliyor kafasini almasi gerekirken hem de 2 tane kutucuk olusturdu 
    #2 tane kopek var diyo. suanda onu duzelticez supression uygulicaz.
    clone = orig.copy()
    
    #non-maximaç burada labelları kutular ve olasiliklar olmak uzere 2 ye boluyoruz. cunku girdimiz kutular ve olasiliklar.
    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    
    boxes = non_max_suppression(boxes, proba)
    
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(clone, (startX, startY),(endX, endY), (0,255,0),2)
        #buraya da label yazisini koyuyoruz.
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX , y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,255,0),2)
        
    cv2.imshow("Maxima", clone)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break
        
#uzun vardede sliding vindow ve pyramid yontemleri yetersiz kaliyor, nesne cogaldikca resimler cogaldikca bunun yerine ilerki
#derslerde daha iyi yontemlere bakicaz.


































































