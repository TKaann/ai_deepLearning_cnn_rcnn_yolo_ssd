import numpy as np
import cv2


#burada boxesleri aliyoruz cunku bizim icin en uygun olan box i secmemiz gerek
#probs ise kutularin dogru yerde olmasi ve dogru nesneyi secme olasilik degeridir,
#overlapThresh ise birden fazla kutu cizdirildiginde buyuk olcude ortusenleri elemeye yarar.
def non_max_suppression(boxes, probs = None, overlapThresh=0.3):
    
    #boxed gelmezse bunu bos bir liste olarak donduruyoruz.
    if len(boxes) == 0:
        return []
    
    #burda ise gelen kutularin type i int ise float a donduruyoruz.
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    #buradaki x1 ve y1  noktasi kutucugumuzun sol ust noktasi 
    #x2 ve y2  degerleri ise kutucugumuzun sag alt noktasini temsil ediyor.
    #buradlarda koseleri buluyoruz.
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    #buralarda ise alanı buluyoruz.
    area = (x2 - x1 + 1)*(y2 - y1 + 1)
    
    idxs = y2
    
    #olasılık degerleri yani yukarda tanimladigimiz probs degerlerimiz. olasilik degerlerine gore kutucuklari siralayacaz.
    if probs is not None:
        
        #kutucuklarimizin indeksini olasiklara gore siralayacagimiz icin olasiliga esitliyoruz.
        idxs = probs
        
    #indeksi siraliyoruz, (burdaki idxs i zaten olasiliga esitledik yukarda, yani olasiliga gore siraliyoruz kutularimizi.)
    idxs = np.argsort(idxs)
    
    #secilen kutulari buraya yazicaz.
    pick = [] 
    
    while len(idxs) > 0:
        
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        #en buyuk ve en küçük x ve y noktalarini buluyoruz.
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        #rect in yukseklik ve genislik yani w ve h degerlerini buluyoruz.
        w = np.maximum(0,xx2 - xx1 + 1)
        h = np.maximum(0,yy2 - yy1 + 1)
        
        #overlap yani iou
        overlap = (w*h)/area[idxs[:last]]
        
        #threshold degerimizin altinda olan idxs leri siliyoruz.
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
    return boxes[pick].astype("int")


























































