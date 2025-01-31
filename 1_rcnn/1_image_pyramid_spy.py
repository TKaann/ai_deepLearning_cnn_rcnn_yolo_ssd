import cv2
import matplotlib.pyplot as plt



#burada minsize i tuple yapiyoruz cunku resimler 2 boyutlu oldugu icin 2 boyuta da ayri ayri bakmamiz gerekecek.
def image_pyramid(image, scale = 1.5, minSize=(224,224)):
    #burada resim generate edecegimiz icin cok resmimiz olacak bu resimleri uretip bir listenin icine koymak ve sonrasinda
    #siniflandirmaya calismak zorlu bir surec ve memory de cok yer kaplar cunku resim dedigimiz sey matris oldugu icin
    #cok fazla matris bizim istedigimiz bir durum degil. bu yuzden de yield komutunu kullanicaz ve resimleri generate edicez.
    
    yield image
    
    while True:
        #fonksiyon floati kabul etmedigi icin inte ceviriyoruz scale ile kuculttugumuz resimleri.
        w = int(image.shape[1]/scale)
        #burada resmi kucultme islemini yapmaya devam ediyoruz.
        image = cv2.resize(image, dsize=(w,w))
        
        #bunu eger resmimizin shape inin 0. indexi yani genisligi yukarda belirledigimiz minsize dan kucuk olana kadar devam.
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        #yield image diyoruz cunku icerdee generate ettigimiz imageler generate icinde return edilmis oluyor.
        yield image
        
# img = cv2.imread("husky.jpg")
# im = image_pyramid(img,1.5, (10,10))
# for i, image in enumerate(im):
#     print(i)
#     if i == 5:
#         plt.imshow(image)




























































