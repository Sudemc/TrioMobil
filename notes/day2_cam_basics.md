# Gün 2 — YOLOv7 için CAM temelleri

## Normal inference ve CAM farkı

Normal inference, tespit sonucunu hızlı üretmek için gradyan hesaplamaz.
CAM ise seçilmiş bir detection skorunun hangi görüntü bölgeleriyle ilişkili olduğunu bulmak için gradyan hesaplar.

## CAM için gerekli üç karar

1. Hangi tespit skoru açıklanacak?
2. Hangi CAM yöntemi kullanılacak?
3. Hangi ara katmandan heatmap üretilecek?

## Referans koddan ilk gözlemler

- `ActivationsAndGradients`, seçilen katmanların aktivasyonlarını ve gradyanlarını kaydediyor.
- `yolov7_target`, class, box veya all hedefini seçiyor.
- Varsayılan layer indeksleri başka bir model için yazılmış olabilir.
- `renormalize` görseli daha okunur yapabilir; ancak heatmap yorumunu etkileyebilir.

## Grad-CAM'in temel kavramları

- Activation: Ara katmanların görüntüden çıkardığı feature map'lerdir.
- Gradient: Seçilmiş target skorunun bir activation kanalına ne kadar duyarlı olduğunu gösterir.
- Hook: Forward pass sırasında activation, backward pass sırasında gradient kaydetmek için seçilen katmana bağlanır.
- Target: Heatmap ile açıklanacak model kararıdır.

False positive analizi için temel soru:
"Model neden bu bölgeyi hedef sınıf olarak gördü?"

Bu nedenle başlangıç hedefi olarak `class` seçeneği anlamlıdır.

## Grad-CAM'in heatmap üretme mantığı

1. Seçilen katmanın feature map'leri alınır.
2. Seçilen target skorunun bu feature map'lere göre gradienti hesaplanır.
3. Her kanalın gradient ortalaması, o kanalın önem ağırlığı olarak kullanılır.
4. Feature map'ler bu ağırlıklarla toplanır.
5. Pozitif katkılar bırakılır, heatmap görüntü boyutuna büyütülür.

Temel fikir:
Heatmap, seçilmiş bir detection skorunu destekleyen ara katman aktivasyonlarını görselleştirir.
## CAM yöntemi seçimi

İlk yöntem olarak Grad-CAM kullanılacak.

Gerekçe:
- Gradient tabanlıdır.
- Seçilmiş detection/class skoruna bağlıdır.
- False positive analizi için "model neden bu sınıfı seçti?" sorusuna uygundur.
- Sonraki yöntemlerle karşılaştırmak için anlaşılır bir baseline oluşturur.

Karşılaştırma için değerlendirilecek yöntemler:
- Grad-CAM++
- XGradCAM
- LayerCAM
- EigenCAM

Önemli not:
Görsel olarak daha çekici bir heatmap, otomatik olarak daha doğru veya daha açıklayıcı değildir.
## Referans script ayar değerlendirmesi

Referans scriptin varsayılan weight, layer ve dosya yolu değerleri başka bir modele aittir; doğrudan kullanılmayacaktır.

İlk deney için planlanan seçimler:
- weight: `models/yolov7.pt`
- device: `cuda:0`
- method: `GradCAM`
- backward_type: `class`
- conf_threshold: `0.25`
- show_box: `True`
- renormalize: `False`

`ratio` parametresi, tek bir detection yerine yüksek sıralı birden fazla adayın skorunu birlikte hedefleyebilir. Bu nedenle referans scriptin ilk heatmap'i, tek bir kutuya tamamen özgü olmayabilir.