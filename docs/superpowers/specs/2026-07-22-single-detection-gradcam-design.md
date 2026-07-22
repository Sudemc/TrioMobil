# Tek Detection Grad-CAM Tasarımı

## Amaç

YOLOv7'nin seçilmiş tek bir detection kutusu için hedef sınıf skorunu hangi görüntü bölgelerinin desteklediğini göstermek. Bu, ileride gerçek false positive örneklerinde "model neden bunu hedef sınıf olarak gördü?" sorusunu araştırmak için kullanılacak.

## Kapsam

- Model: Resmi pretrained `models/yolov7.pt`
- CAM yöntemi: `GradCAM`
- Hedef: Seçilen tek detection'in class score'u
- Target layer'lar: Detect katmanını besleyen `102`, `103`, `104`
- Girdi: Tek görüntü
- Çıktı: Heatmap bindirilmiş görüntü, seçilen kutu ve deney ayarları

## Veri Akışı

1. YOLOv7 görüntü için ham aday tahminler üretir.
2. NMS, nihai detection kutularını oluşturur.
3. Kullanıcı veya otomatik seçim ile tek bir nihai kutu belirlenir.
4. Bu kutu, sınıfı ve IoU ilişkisi kullanılarak NMS-oncesi ham adayla eşleştirilir.
5. Yalnızca bu adayın hedef class score'u Grad-CAM target scalar'ı olur.
6. Grad-CAM, layer 102-104 aktivasyonları ve gradientleriyle heatmap üretir.
7. Heatmap, ilk deneyde kutu-ici yeniden normalizasyon olmadan (`renormalize=False`) kaydedilir.

## Ayarlar

```text
weight          models/yolov7.pt
device          cuda:0
method          GradCAM
target_mode     class
target_layers   [102, 103, 104]
conf_threshold  0.25
show_box        True
renormalize     False
```

## Neden Bu Tasarım?

- False positive sorusu, kutunun konumundan once sinif kararini aciklamayi gerektirir.
- Tek kutu hedefi, birden fazla detection skorunun toplamini aciklayan genel heatmap'ten daha denetlenebilirdir.
- 102-104 layer'lari, modelin Detect katmanina gercekten girdi veren uc olcekteki feature map'lerdir.
- Detect katmanina dogrudan hook baglanmaz; bu, YOLOv7'de bilinen in-place gradient sorunlarini azaltir.
- Kutu-ici yeniden normalizasyon ilk sonucu daha guzel gosterebilir, ancak kutu disindaki dikkat alanlarini gizleyebilir.

## Basari Olcutleri

- Script tek bir secilmis detection icin heatmap uretir.
- Secilen kutu, sinifi ve confidence degeri ciktiya yazilir.
- Ayni girdi ve ayarlarla calistirildiginda sonuc tekrar uretilebilir olur.
- Heatmap, kesin neden olarak degil, test edilmesi gereken bir dikkat hipotezi olarak belgelenir.

## Sinirliliklar

- CAM, modelin nereden etkilendigini gosterir; hatanin nedensel ve tek aciklamasini kanitlamaz.
- NMS sonrasi kutuyu ham adayla eslestirme, benzer veya cok cakisan kutularda belirsizlik icerebilir.
- Target layer secimi deneyle dogrulanmalidir; bu ilk secim nihai cevap degildir.
