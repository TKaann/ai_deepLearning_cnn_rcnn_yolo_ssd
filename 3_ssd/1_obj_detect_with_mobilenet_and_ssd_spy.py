import numpy as np
import os
import cv2

#ssd denilen kavram kayan pencere kullanmak yerine goruntuyu bir izgara kullanarak boluyor ve her bir izgara hucresinin
#goruntunun o bolgedeki nesneleri tespit etmekten sorumlu olmasini sagliyor.

#nesneleri algilama, o bolgedeki bir nesnenin sinifini ve tahmine etmek anlamina geliyor.

#mesela normal bir resmi 8 e 8 parcalara boldugunu ve ordaki her bir izgaranin icerisindeki resmi tespit etmeye calistigini
#dusunebiliriz. hem siniflandiriyoruz hem de siniflandirma sonucunda bir sey bulabilirse resim icinde tespit etmeye calisiyoruz.
#sonrasinda bunu birlstirdigmiz zaman ortaya bizim tespit etmek istedigimiz obje cikiyor.
#objeyi tespit ettigmizde ise otomatik olarak x y kordinatlari ve genislik yukselik degerleri de cikiyor.
#sonrasinda bir de confidencesi (dogru sekilde nesne tespiti, yani prob olarak dusunebiliriz. tahmin.) ortaya cikiyor 



#burada siniflandirma yapabildigmiz classlar var.
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

#her bir sinif icin renk blirliyoruz.
COLORS = np.random.uniform(0,255,size=(len(CLASSES),3))

#simdi ise cv2 kutuphanesini kullanarak deeopneeuralneetwork kismindan gerekli olan mobileneetssd yi iceriye aktariyoruz.
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")


#resimleri ice aktariyoruz.
files = os.listdir()
img_path_list = []
for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(f)
        
for i in img_path_list:
    #resimleri okuyoruz.
    image = cv2.imread(i)
    #shape deegerlerindene yuksekelik gnislik aliyoruz.,
    (h,w) = image.shape[:2]
    #burda cv2 nin neural nworkunu kullanarak, bunu preprocces olarak dusunebiliriz. meesela rsize yapiyoruz, 300 e 300
    #300 e 300 olmasini sebebi ssd boylee kabul ediyor.geri kalanlar ise modlle ilgili parametreler. onlari degistirmiyoruz.
    blob = cv2.dnn.blobFromImage(cv2.resize(image,(300, 300)), 0.007843,(300, 300), 127.5)
    
    net.setInput(blob)
    #detectionlarimizi elde ettik.
    detections = net.forward()
    
    #simdi bu detectionlari gorsellestiricez.
    for j in np.arange(0, detections.shape[2]):
        
        confidence = detections[0,0,j,2]
        
        #burdaki confidence degerini degistirerek tespit uzerinde oynamalara yapabiliriz.
        if confidence > 0.10:
            
            idx = int(detections[0,0,j,1])
            box = detections[0,0,j,3:7]*np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            
            #burada labeli ve % kac tahmin ettigniz yaziyoruz.
            label = "{}: {}".format(CLASSES[idx], confidence)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx],2)
            y = startY - 16 if startY -16 >15 else startY + 16
            cv2.putText(image, label, (startX,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5,COLORS[idx],2)
            
    cv2.imshow("ssd",image)
    if cv2.waitKey(0) & 0xFF == ord("q"): continue

































































