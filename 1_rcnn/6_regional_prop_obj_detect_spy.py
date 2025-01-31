#bu algoritmamiz 2 asamadan olusmaktadir. birincisi secmeli arama yapiyoruz. onceki derste gordugumuz secmeli arama ile
#resimler uzerinde segmentasyon islemi yapiyoruz.

#ikincisi ise imageNet veri seti uzerinden egitilmis olan resnet evrisimsel sinir agini kullanarak siniflandirma yapiyoruz.
#bu daha onceden kullandigimi cnn di.



from tensorflow.keras.applications.resnet50 import preprocess_input 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2

#bu non max ayni objeyi birden fazla kez tespit ettiysek onu azaltmaya yariyordu o yuzden bunu ellemiyoruz.
from non_max_supression import non_max_suppression




#onceki derste yazdigimiz selective_search algoritmasini yaziyoruz.
def selective_search(image):
    print("ss")
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)

    ss.switchToSelectiveSearchQuality()
    
    rects = ss.process()
    #hepsini degil ilk 1000 tanesini return ediyoruz hepsini return edince siniflandirma islemi uzun suruyor.
    return rects[:1000]


#model olarka resneti kullandigmiz icin import ediyoruz resneti.
model = ResNet50(weights="imagenet")    
image = cv2.imread("animals.jpg")
image = cv2.resize(image, dsize = (400,400))
(H, W) = image.shape[:2]


#ss islemini kosturuyoruz.
rects = selective_search(image)

#yukardaki islem sonuucnda region of interest buluyoruz ve bunlara ait koordinatlari burdaki iki liste icine aticaz.
proposals = []
boxes = []
for (x, y, w, h) in rects:

    #oncekinden farkli olarak bu komut satirini yaziyoruz burda ise
    #bizim genisligimiz ve yuksekligimiz regionumuzun %10 undan daha azsa buraya girmeden devam et diyoruz.
    if w / float(W) < 0.1 or h / float(H) < 0.1: continue
    
    #roileri resim uzeinden buluyoruz sonrasinda rengini degistiriyoruz ve resize islemi yapiyoruz
    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224, 224))
    
    #sonra bunlai arraye ceviriyoruz ve processec i kullanarak resnet icin uygun hale getiriyoruz.
    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    #sonrasinda elde ettiklerimizi yukardaki yazdigimiz 2 tane liste icine atiyoruz.
    proposals.append(roi)
    boxes.append((x, y, w, h))

#arraye ceviriyoruz.
proposals = np.array(proposals)

#modelimizin predict islemini gerceklestiriytoruz.
print("predict")
preds = model.predict(proposals)
preds = imagenet_utils.decode_predictions(preds, top=1)

#simdi ise belli bir predict oraninin uzerindekileri aliyoruz. onceki derslerde bu islemi yapmistik.
labels = {}
min_conf = 0.8
for (i, p) in enumerate(preds):
    
    #imagenetId onemdiz oldugu icin onu almiyoruz label ve prob u aliyoruz.
    (_, label, prob) = p[0]
    if prob >= min_conf:
        (x, y, w, h) = boxes[i]
        box = (x, y, x + w, y + h)
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

clone = image.copy()

for label in labels.keys():
    for (box, prob) in labels[label]:
        #her bit kutu ve olasilik degerini ayikliyoruz ve bunlari kullanarak non max yontemimizi cagiriyoruz.
        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])
        boxes = non_max_suppression(boxes, proba)
        
    
        #burda ise elde ettigmiz nihai kutucuklari gorsellestiriyoruz.
        for (startX, startY, endX, endY) in boxes:
            cv2.rectangle(clone, (startX, startY), (endX, endY),(0, 0, 255), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(clone, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        cv2.imshow("After", clone)
        if cv2.waitKey(1) & 0xFF == ord('q'):break
        
        
        
        




























































