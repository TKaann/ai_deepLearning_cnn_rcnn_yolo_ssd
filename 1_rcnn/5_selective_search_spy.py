import cv2
import random


#selective search fonksiyonumuz pyramid ve selected window fonksiyonlarina alternatif olarak yapacagimiz fonksiyondur.


image = cv2.imread("pyramid.jpg")
image = cv2.resize(image, dsize = (600,600))
cv2.imshow("image",image)

#selective search algoritmamizi ilklendiriyoruz. burada segmenatation algoritmasiyla yapiyoruz.
#burada aslinda selective search algoritmamizi iceriye aktariyoruz.
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality()

#burada ise algoritmamizi kosturuyoruz.
print("start")
rects = ss.process()

output = image.copy()

#burada bizim rects algoritmamiz 9k kusur tane rectangle olusturdu biz burada ilk 50 tanesine bakiyoruz.
for (x,y,w,h) in rects[:50]:
    color = [random.randint(0,255) for j in range(0,3)]
    cv2.rectangle(output, (x,y),(x+w,y+h),color,2)
    
cv2.imshow("output",output)



#burada yapilan yani nesne tespiti icin secmeli arama. burada yapilan is super piksek algoritmasi kullanarak bir goseli
#asiri bolumlere ayirma yontemidir. yani bir segmentasyon yapiyoruz.

#super piksel, ortak ozellikleri paylasan bir piksel grubu olarak tanimlanabilir. yani bir kume olarak dusunebiliriz.

#burada 5 temel benzelik olcusu vardir bunlar
#1 - renk benzerlikleri 
#2 - doku benzelikleri
#3 - boyut benzerlikleri
#4 - sekil benzerlikleri
#5 - yukaridaki benzerliklerin dogrusal kombinasyonu 

#secmeli arama sinf etiketleri degil bolgeler olusturur. secmeli arama dedigimiz sey bir nesne tespit yontemi degildir.
#secmeli arama daha onceden ogrenmis oldugumuz resim piramidi ve sliding window (kayan pencere) algoritmalarina alternatif
#bir algoritmadir. yani buarad herhangi bir sinif etiketi olusmuyor. 
#secmeli aramada sadece burada bir nesne olabilir diye pencereler olustuurluyor.
#bu pencereler ise yukardaki maddeller dogrultusunda belirleniyor.
























































