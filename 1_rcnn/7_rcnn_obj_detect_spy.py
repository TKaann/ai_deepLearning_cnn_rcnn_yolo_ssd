import cv2
#pickle i modelimizi iceriye aktarmak icin kullaniyoruz.
import pickle
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array



#secmeli arama ile resimlerimizi elde edicez ve kendimiz egittimiz evrisimsel sinir agini kullanarak nesne tespitini 
#gerceklestiricez.


#daha onceden egittigimiz rakamlari taniyan modelimiz ile tespit ve siniflandirma yaptirticaz.




image = cv2.imread("mnist.png")
cv2.imshow("Image",image)

#selective search ilklendirme islemini yapiyoruz.
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality()

print("SS")
rects = ss.process()

proposals = []
boxes = []
output = image.copy()

for (x,y,w,h) in rects[:100]:
    
    color = [random.randint(0,255) for j in range(0,3)]
    cv2.rectangle(output, (x,y), (x+w,y+h),color, 2)
    
    roi = image[y:y+h,x:x+w]
    #algirotmamizda 32 ye 32 olarak egittigimiz icin boyutunu 32 ye 32 yapiyoruz.
    #INTER_LANCZOS4 ise resmi kuculturken yeni bir resim elde ettigimiz icin aradaki seyleri nasil dolduracagimizi ifade eden
    #bir parametre. yani bunlari interpole eden parametremiz. 
    roi = cv2.resize(roi, dsize=(32,32), interpolation = cv2.INTER_LANCZOS4)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    roi = img_to_array(roi)
    
    #propos region of interestlerimiz box ise koordinatlarimiz.
    proposals.append(roi)
    boxes.append((x,y,w+x,h+y))
    
#biri int biri float olacak.
proposals = np.array(proposals, dtype = "float64")    
boxes = np.array(boxes, dtype = "int32")    

print("sınıflandırma")
#modelimizi okutuyoruz.
pickle_in = open("model_trained_v4.p", "rb")   
model = pickle.load(pickle_in)
#olasilik degerlerimize tahminlerimize bakiyoruz.
proba = model.predict(proposals)

number_list = []
idx = []
for i in range(len(proba)):
    
    #butun prob degerlerine bakiyorum ve en yuksek prob degerime bakiyorum
    max_prob = np.max(proba[i,:])
    #eger bu en yuksek prob deger benim istedigim thresholddan yuksek ise idx listem icine bu degeri ekliyorum
    if max_prob > 0.95:
        idx.append(i)
        number_list.append(np.argmax(proba[i]))
    
#gorsellestirme. rect ciziyoruz.
for i in range(len(number_list)):
    
    j = idx[i]
    cv2.rectangle(image, (boxes[j,0], boxes[j,1]), (boxes[j,2],boxes[j,3]),[0,0,255],2)
    cv2.putText(image, str(np.argmax(proba[j])),(boxes[j,0] + 5, boxes[j,1] +5 ), cv2.FONT_HERSHEY_COMPLEX,1.5,(0,255,0))
    
    cv2.imshow("Image",image)
    
    
    
    































































