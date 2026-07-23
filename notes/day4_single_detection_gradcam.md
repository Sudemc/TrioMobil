# Gün 4 — Tek Tespit İçin Grad-CAM

## Bugünkü hedef

YOLOv7'nin NMS sonrasında kalan tek bir tespitini, doğru ham model adayıyla eşleştirmek ve seçilen adayın sınıf skoruna göre Grad-CAM heatmap üretmek.

## Başlangıç durumu

- Yöntem: `GradCAM`
- Model: `models/yolov7.pt`
- Görüntü: `third_party/yolov7/inference/images/horses.jpg`
- Cihaz: `cuda:0`
- Confidence threshold: `0.25`
- Açıklanan çıktı: seçilen tespitin class skoru
- Hedef feature katmanları: `[102, 103, 104]`
- `Detect` katmanı: `105`; doğrudan hook bağlanmadı
- Heatmap yalnızca kutu içinde yeniden normalize edilmedi

## 1. Ham aday eşleştirme yardımcıları

Grad-CAM'in hangi skoru açıklayacağını belirlemek için NMS sonrasında seçilen kutuyu, NMS öncesindeki doğru ham YOLO adayıyla eşleştiren yardımcı fonksiyonlar yazıldı:

- `xywh_to_xyxy`: Ham kutu biçimini merkez-genişlik-yükseklik gösteriminden köşe koordinatlarına dönüştürür.
- `box_iou`: İki kutu grubunun örtüşme oranını hesaplar.
- `select_detection`: NMS sonrasında kalan kutulardan istenen indeksteki tespiti seçer.
- `match_raw_candidate`: Sınıf, IoU ve confidence uyumuna göre seçilen tespitin ham adayını bulur.

### Neden test beklentisi 0 oldu?

Test verisinde aynı konumu ve sınıfı paylaşan iki ham aday vardı:

| Ham aday | Objectness | Sınıf skoru | Objectness × sınıf skoru |
|---:|---:|---:|---:|
| 0 | 0.90 | 0.95 | 0.855 |
| 2 | 0.80 | 0.96 | 0.768 |

NMS sonrasındaki kutunun confidence değeri yaklaşık `0.86` olduğu için doğru kaynak aday `0`dır. Eski plandaki `2` beklentisi `0` olarak düzeltildi.

### Test sonuçları

```text
test_detection_index_is_selected ... ok
test_empty_detections_raise_index_error ... ok
test_iou_of_identical_boxes_is_one ... ok
test_raw_candidate_matches_nms_confidence ... ok
test_xywh_to_xyxy ... ok

Ran 5 tests
OK
```

Öğrendiğim: Test sonucunun anlamlı olması için beklenen değeri rastgele yazmamak gerekir. Beklenen değer modelin matematiğinden ve gerçek veri akışından türetilmelidir.

## 2. Tek-tespit Grad-CAM komutu

`scripts/yolov7_single_detection_cam.py` oluşturuldu. Komut şu işlemleri yapar:

1. YOLOv7 modelini yükler ve `eval()` moduna alır.
2. Görüntüyü letterbox ile modele hazırlar.
3. Normal inference ve NMS çalıştırır.
4. `--detection-index` ile tek bir nihai kutu seçer.
5. Bu kutuyu doğru ham adayla eşleştirir.
6. Ham adayın tahmin edilen sınıfa ait skorunu Grad-CAM target olarak kullanır.
7. `[102, 103, 104]` katmanlarından heatmap hesaplar.
8. Kutulu heatmap ile deney metadata'sını kaydeder.

Komut arayüzü şu parametrelerle doğrulandı:

```text
--weights
--source
--output-dir
--detection-index
--conf-threshold
--device
```

## Karşılaşılan sorunlar ve çözümleri

### Sorun 1 — Yanlış Python ortamı

İlk test komutu sistem Python'u ile çalıştı ve şu hata oluştu:

```text
ModuleNotFoundError: No module named 'torch'
```

Kök neden: PyTorch sistem Python'unda değil, projenin `.venv` sanal ortamında kuruluydu.

Çözüm:

```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_yolov7_cam_utils.py -v
```

Öğrendiğim: Bir import hatası kodun eksik olduğu anlamına gelmeyebilir; yanlış Python yorumlayıcısı kullanılıyor olabilir.

### Sorun 2 — Proje içi modül bulunamadı

Komut dosyası doğrudan çalıştırıldığında şu hata oluştu:

```text
ModuleNotFoundError: No module named 'scripts.yolov7_cam_utils'
```

Kök neden: `python scripts/...` çalıştırıldığında Python arama yolunda proje kökü bulunmuyordu.

Çözüm: Komut dosyası, proje kökünü `sys.path` içine ekleyecek şekilde düzenlendi.

Öğrendiğim: Bir Python dosyasının hangi klasörden ve hangi biçimde çalıştırıldığı import davranışını etkiler.

### Sorun 3 — Grad-CAM target tensor boyutu

İlk Grad-CAM denemesinde şu hata oluştu:

```text
IndexError: too many indices for tensor of dimension 2
```

Kök neden:

- YOLOv7 wrapper çıktısı `[1, 25200, 85]` biçimindeydi.
- `grad-cam==1.4.8`, target fonksiyonuna batch içindeki tek görüntünün `[25200, 85]` çıktısını gönderiyordu.
- Target fonksiyonu hâlâ batch boyutunu indekslemeye çalışıyordu.

Çözüm:

```text
predictions[0, raw_index, class_column]
```

yerine:

```text
predictions[raw_index, class_column]
```

kullanıldı.

Öğrendiğim: Bir model wrapper'ının çıktısıyla kütüphanenin target fonksiyonuna aktardığı tensor aynı boyut sayısına sahip olmayabilir. Aradaki kütüphane davranışı kaynak koddan doğrulanmalıdır.

### Sorun 4 — YOLOv7 in-place gradient hatası

İkinci Grad-CAM denemesinde şu hata oluştu:

```text
RuntimeError: one of the variables needed for gradient computation
has been modified by an inplace operation
```

Kök neden: YOLOv7 `Detect` başlığı, inference sırasında sigmoid çıktısının kutu koordinatlarını tensor üzerinde yerinde değiştiriyordu. Grad-CAM backward aşamasında tensorun önceki sürümüne ihtiyaç duyduğu için gradyan hesabı bozuldu.

Yapılan doğrulama:

```text
YOLOv7 çıktı şekli: (1, 25200, 85)
Non-inplace backward: passed
```

Çözüm: `third_party/yolov7` kaynak kodu değiştirilmeden, kendi model wrapper'ımızın forward çağrısında YOLOv7'nin non-inplace decode dalı geçici olarak kullanıldı. Grad-CAM hook'ları yine head öncesindeki `[102, 103, 104]` feature katmanlarında kaldı.

Öğrendiğim: `eval()` modu gradyanları kapatmaz. Fakat gradient tabanlı açıklama yöntemlerinde forward sırasında yapılan in-place işlemler backward zincirini bozabilir.

## İlk başarılı Grad-CAM deneyi

Kullanılan komut:

```powershell
.\.venv\Scripts\python.exe scripts/yolov7_single_detection_cam.py --source third_party/yolov7/inference/images/horses.jpg --output-dir outputs/day2_single_detection --detection-index 0 --conf-threshold 0.25 --device cuda:0
```

Sonuç:

- Yöntem: `GradCAM`
- Açıklanan tespit indeksi: `0`
- Sınıf: `horse`
- Sınıf ID: `17`
- Detection confidence: `0.899893`
- Eşleşen ham aday: `400213`
- Hedef katmanlar: `[102, 103, 104]`
- Renormalize: `False`
- Heatmap: `outputs/day2_single_detection/heatmap.png`
- Metadata: `outputs/day2_single_detection/metadata.json`

### Görüntü boyutu gözlemi

İlk denemede görüntü boyutu `model.stride.max() × 80` olarak hesaplandı:

```text
32 × 80 = 2560
```

Bu nedenle ilk heatmap `2560 × 2560` işlenmiş görüntü üzerinde üretildi, kutu koordinatları büyük çıktı ve Grad-CAM normal `640 × 640` inference'a göre daha uzun sürdü. Sonraki adımda bu seçimin `640` ile karşılaştırılması ve deneylerin aynı giriş boyutunda tekrarlanabilir hâle getirilmesi gerekir.

## Giriş boyutunun 640 olarak sabitlenmesi

Komuta `--img-size` parametresi eklendi ve varsayılan değer `640` yapıldı. Önce varsayılan değeri kontrol eden test yazıldı.

Testin ilk sonucu:

```text
AttributeError: 'Namespace' object has no attribute 'img_size'
```

Bu beklenen başarısızlıktan sonra parametre ve preprocessing kullanımı eklendi. Tüm testlerin son durumu:

```text
Ran 6 tests
OK
```

Öğrendiğim: Giriş boyutu preprocessing'in bir parçasıdır ve deney metadata'sı kadar açık biçimde kontrol edilmelidir. Farklı giriş boyutları confidence, kutu koordinatları, ham aday sayısı ve heatmap görünümünü değiştirebilir.

## 640 × 640 tek-tespit deneyleri

İki deneyde yalnızca `detection-index` değiştirildi. Model, kaynak görüntü, eşik, giriş boyutu, target ve hedef katmanlar aynı tutuldu.

### Deney A — detection index 0

```powershell
.\.venv\Scripts\python.exe scripts/yolov7_single_detection_cam.py --source third_party/yolov7/inference/images/horses.jpg --output-dir outputs/day4_single_detection_index0_640 --detection-index 0 --conf-threshold 0.25 --img-size 640 --device cuda:0
```

- Sınıf: `horse`
- Confidence: `0.957332`
- Ham aday indeksi: `24224`
- Kutu: `[0.044, 266.302, 260.786, 448.565]`
- Heatmap: `outputs/day4_single_detection_index0_640/heatmap.png`
- Metadata: `outputs/day4_single_detection_index0_640/metadata.json`

### Deney B — detection index 1

```powershell
.\.venv\Scripts\python.exe scripts/yolov7_single_detection_cam.py --source third_party/yolov7/inference/images/horses.jpg --output-dir outputs/day4_single_detection_index1_640 --detection-index 1 --conf-threshold 0.25 --img-size 640 --device cuda:0
```

- Sınıf: `horse`
- Confidence: `0.943739`
- Ham aday indeksi: `24213`
- Kutu: `[358.194, 284.159, 492.544, 395.512]`
- Heatmap: `outputs/day4_single_detection_index1_640/heatmap.png`
- Metadata: `outputs/day4_single_detection_index1_640/metadata.json`

### İlk karşılaştırma

- İki deney farklı nihai kutuları seçti.
- İki kutu farklı ham adaylara eşleşti: `24224` ve `24213`.
- Her iki deney de aynı `horse` class-score target'ını kullandı.
- Bu sonuç, komutun detection index'e göre farklı bir açıklama hedefi seçtiğini metadata seviyesinde doğruladı.

### Görsel karşılaştırma

**Detection index 0:**

- Yeşil kutu soldaki büyük, beyaz atı kapsıyor.
- En sıcak kırmızı/turuncu alan seçili kutunun bir bölümünü kapsıyor.
- Ancak sıcaklık yalnızca seçili atta kalmıyor; ortadaki ve sağdaki atlara da yatay bir bant halinde yayılıyor.

**Detection index 1:**

- Yeşil kutu sağdaki gri atı kapsıyor.
- Seçili at üzerinde sarı/turuncu aktivasyon bulunuyor.
- Buna rağmen en güçlü kırmızı alan seçilen kutunun dışında, soldaki ve ortadaki atların üzerinde kalıyor.

**Yorum:**

- Detection index değiştiğinde yeşil kutu ve ham aday değişiyor; fakat heatmap'in genel şekli belirgin biçimde değişmiyor.
- Her iki heatmap de tek bir at yerine görüntüdeki bütün at sırasını ve atların bulunduğu yatay bölgeyi vurguluyor.
- Index 1 heatmap'i seçilen sağ atı tamamen görmezden gelmiyor, ancak en güçlü aktivasyon seçili kutuya ait değil.
- Bu nedenle mevcut sonuç metadata seviyesinde detection-specific olsa da görsel seviyede güçlü biçimde detection-specific değildir.
- Muhtemel nedenler; iki hedefin de aynı `horse` sınıfına ait olması, Grad-CAM'in düşük uzamsal çözünürlüğü, üç katmanın ortalamasının alınması ve kutu dışı alanların yeniden normalize edilmemesidir.
- Bu sonuç “model bu tek ata tam olarak neden horse dedi?” sorusunu kesin olarak cevaplamaz. Daha güvenli yorum, modelin `horse` sınıfı için görüntüdeki at benzeri yapılar ve atların bulunduğu genel yatay bölgeyle ilişkili aktivasyon ürettiğidir.
- Bu gözlem, yönergede belirtilen “güzel görünen heatmap tek başına güvenilir açıklama değildir” sınırlılığına gerçek bir örnektir.
## Ham CAM sayısal karşılaştırması

Renkli overlay, yeşil kutu ve yazının karşılaştırmayı etkilememesi için normalize edilmiş ham CAM matrisi iki biçimde kaydedildi:

- `cam_grayscale.npy`: Sayısal karşılaştırma için kayıpsız matris
- `cam_grayscale.png`: Görsel inceleme için gri görüntü

Karşılaştırma sonuçları:

| Metrik | Sonuç |
|---|---:|
| Pearson korelasyonu | `0.9999999999998445` |
| Kosinüs benzerliği | `0.9999999403953552` |
| Ortalama mutlak fark | `0.0000002022` |
| En sıcak %10 bölgenin Jaccard örtüşmesi | `1.0` |

Kutu içi ölçümler:

| Ölçüm | Sonuç |
|---|---:|
| CAM 0'ın kendi kutusundaki ortalaması | `0.681458` |
| CAM 0'ın diğer kutudaki ortalaması | `0.695974` |
| CAM 1'in kendi kutusundaki ortalaması | `0.695974` |
| CAM 1'in diğer kutudaki ortalaması | `0.681459` |
| CAM 0'ın en sıcak %10 piksellerinin kendi kutusundaki oranı | `0.616040` |
| CAM 1'in en sıcak %10 piksellerinin kendi kutusundaki oranı | `0.182935` |

Yorum:

- İki farklı ham aday hedeflenmesine rağmen iki CAM pratik olarak aynıdır.
- CAM 0'ın aktivasyonu kendi kutusundan çok diğer kutuda biraz daha yüksektir.
- CAM 1 için en sıcak piksellerin yalnızca yaklaşık `%18.3`ü kendi kutusundadır.
- Bu nedenle mevcut klasik Grad-CAM sonucu tek tespit açıklaması olarak yeterince ayrıştırıcı değildir.
- Olası teknik neden, klasik Grad-CAM'in her kanalın gradyanını uzamsal olarak ortalayıp tek kanal ağırlığına dönüştürmesidir. Aynı sınıfa ait iki konumun benzer kanal ağırlıkları üretmesi, farklı adaylar hedeflense bile aynı aktivasyon haritasının oluşmasına yol açabilir.
- Sonraki deneyde önce katman ortalamasının etkisi ayrıştırılmalı; her katman tek başına denenmelidir. Sonuç değişmezse uzamsal gradyanı koruyan `LayerCAM` veya `HiResCAM` gibi yöntemler karşılaştırılmalıdır.
## Tek katman Grad-CAM deneyleri

Katman ortalamasının iki detection için aynı CAM'i üretip üretmediğini anlamak amacıyla `[102]`, `[103]` ve `[104]` ayrı ayrı test edildi.

| Target layer | Index 0 CAM | Index 1 CAM | Karşılaştırma |
|---:|---|---|---|
| `102` | Tamamen sıfır | Tamamen sıfır | Benzerlik tanımsız |
| `103` | Tamamen sıfır | Tamamen sıfır | Benzerlik tanımsız |
| `104` | Dolu | Dolu | Pearson ≈ `1.0`, Jaccard `1.0` |

Layer 104 sonuçları:

- Pearson korelasyonu: `0.9999999999998496`
- Kosinüs benzerliği: `1.0`
- Ortalama mutlak fark: `0.0000001896`
- En sıcak %10 örtüşmesi: `1.0`

Yorum:

- Ham aday indeksleri `24224` ve `24213`, concatenated YOLO çıktısının üçüncü ölçeğine düşmektedir.
- Bu nedenle bu adayların class skorlarına giden gradyan yolu layer `104` üzerinden geçer; layer `102` ve `103` için gradyan sıfırdır.
- Üç katman birlikte kullanıldığında görülen CAM, pratikte yalnızca layer `104` CAM'idir. Diğer iki boş CAM ortalamaya katılsa da son normalizasyon genel şekli değiştirmez.
- Layer `104` tek başına da iki detection için aynı haritayı üretmektedir. Dolayısıyla benzerliğin nedeni katman ortalaması değildir.
- Bulgular klasik Grad-CAM'in uzamsal gradyanı kanal başına tek sayıya indirgemesinin, aynı sınıftaki farklı konumları ayırmada yetersiz kaldığı hipotezini desteklemektedir.
- Sonraki yöntem karşılaştırmasında uzamsal gradyanı koruyan `HiResCAM` veya `LayerCAM` kullanılmalıdır.
## HiResCAM karşılaştırması

Klasik Grad-CAM'in uzamsal ortalama nedeniyle iki detection'ı ayıramadığı hipotezini test etmek için layer `104` üzerinde `HiResCAM` çalıştırıldı. HiResCAM, kanal başına global gradyan ortalaması kullanmak yerine her konumda `gradient × activation` hesaplar.

Her iki deneyde model, görüntü, class-score target, layer ve preprocessing aynı tutuldu; yalnızca detection index değiştirildi.

| Metrik | HiResCAM sonucu |
|---|---:|
| Pearson korelasyonu | `-0.005660` |
| Kosinüs benzerliği | `0.0` |
| Ortalama mutlak fark | `0.005160` |
| En sıcak %10 bölgenin Jaccard örtüşmesi | `0.0` |
| CAM 0'ın diğer kutudaki ortalaması | `0.0` |
| CAM 1'in diğer kutudaki ortalaması | `0.0` |
| CAM 0 sıcak piksellerinin kendi kutusundaki oranı | `1.0` |
| CAM 1 sıcak piksellerinin kendi kutusundaki oranı | `1.0` |

Yorum:

- İki HiResCAM haritası birbirinden belirgin biçimde ayrılmıştır.
- En sıcak bölgeler iki detection arasında hiç çakışmamaktadır.
- Her haritanın en sıcak %10 piksellerinin tamamı kendi seçili kutusunda kalmıştır.
- Diğer detection kutusunda ortalama aktivasyon sıfırdır.
- Bu sonuç, ham aday eşleştirme ve class-score target yaklaşımının çalıştığını; klasik Grad-CAM'deki aynı-harita probleminin büyük ölçüde yöntemin uzamsal gradyan ortalamasından kaynaklandığını destekler.
- Bu örnek için HiResCAM, klasik Grad-CAM'e göre çok daha detection-specific bir açıklama üretmektedir.
- Yine de kutu içinde aktivasyon bulunması modelin hatasının kesin nedenini kanıtlamaz; yöntem model skoruyla ilişkili konumu gösterir.
### HiResCAM görsel yorumu

**Detection index 0:**

- Yeşil kutu soldaki büyük beyaz atı kapsamaktadır.
- Tek güçlü sıcak bölge seçili atın gövdesinin alt-orta kısmında, ön bacaklara yakın bölgede oluşmuştur.
- Diğer atlarda ve arka planda belirgin sıcak aktivasyon görünmemektedir.

**Detection index 1:**

- Yeşil kutu sağdaki gri atı kapsamaktadır.
- Güçlü sıcak bölge yalnızca seçili gri atın gövdesinin merkezinde oluşmuştur.
- Soldaki ve ortadaki diğer atlarda belirgin sıcak aktivasyon görünmemektedir.

**Karşılaştırmalı yorum:**

- Detection index değiştiğinde sıcak bölge de doğru seçili ata taşınmaktadır.
- Bu davranış, HiResCAM'in bu örnekte klasik Grad-CAM'e göre çok daha detection-specific olduğunu görsel olarak doğrulamaktadır.
- Her iki harita da baş, kuyruk veya bütün siluet yerine gövdenin küçük bir bölümünü vurgulamaktadır. Bu, modelin class skoruyla en güçlü ilişkili yerel bölgenin gövde olduğunu düşündürür; ancak modelin yalnızca bu özelliğe dayanarak karar verdiğini kanıtlamaz.
- Sıcak bölgelerin küçük ve elmas/piksel bloklu görünmesi layer `104` feature map'inin düşük uzamsal çözünürlüğünün `640 × 640` görüntüye büyütülmesinden kaynaklanır.
- Sonuç konum açısından başarılıdır; fakat ayrıntılı anatomik yorum için layer `104` çözünürlüğü sınırlıdır.
## Aynı detection için GradCAM ve HiResCAM

Aynı seçili detection üzerinde yalnızca CAM yöntemi değiştirilerek GradCAM ve HiResCAM ham matrisleri karşılaştırıldı.

| Detection | Pearson | Kosinüs | Ortalama mutlak fark | En sıcak %10 örtüşmesi |
|---:|---:|---:|---:|---:|
| Index `0` | `0.183199` | `0.189083` | `0.217568` | `0.096226` |
| Index `1` | `0.155063` | `0.167310` | `0.217579` | `0.071079` |

Yorum:

- Aynı detection hedeflenmesine rağmen GradCAM ve HiResCAM haritaları düşük benzerliğe sahiptir.
- En sıcak %10 bölgelerin örtüşmesi index 0 için yaklaşık `%9.6`, index 1 için yaklaşık `%7.1`dir.
- HiResCAM, GradCAM sonucunu yalnızca görsel olarak daraltmamış; uzamsal olarak farklı bir açıklama üretmiştir.
- GradCAM geniş bir at sırası bandını vurgularken HiResCAM seçilen atın gövdesinde küçük ve ayrışmış bir bölge üretmiştir.
- Her yöntem kendi CAM matrisini `0–1` aralığında bağımsız normalize ettiği için ortalama aktivasyon büyüklükleri yöntemler arasında doğrudan “daha güçlü” veya “daha zayıf” olarak yorumlanmamalıdır.
- Yöntem seçimi, heatmap sonucunu ve yapılacak yorumu ciddi biçimde değiştirmektedir. Bu nedenle raporda kullanılan yöntem mutlaka açıkça belirtilmelidir.
## Kutu içi yeniden normalizasyon deneyi

Klasik `GradCAM`, layer `104`, detection index `1` deneyi aynı ayarlarla tekrarlandı. Tek değişiklik `--renormalize-within-box` seçeneğinin açılmasıydı.

Yöntem:

1. Orijinal ham CAM değiştirilmeden `cam_grayscale.npy` olarak kaydedildi.
2. Görselleştirme için ayrı bir kopya oluşturuldu.
3. Seçili kutunun dışındaki bütün değerler sıfırlandı.
4. Kutunun içindeki minimum değer `0`, maksimum değer `1` olacak şekilde yeniden ölçeklendi.
5. Renkli overlay bu değiştirilmiş görselleştirme kopyasından üretildi.

Doğrulama:

| Kontrol | Sonuç |
|---|---:|
| Eski ve yeni ham CAM birebir aynı mı? | `True` |
| Ham CAM maksimum mutlak farkı | `0.0` |
| Yeniden normalize edilmiş CAM kutu dışı maksimumu | `0.0` |
| Kutu içi minimum | `0.0` |
| Kutu içi maksimum | `0.9999998` |

Yorum:

- Yeniden normalizasyon modelin gradyanını, aktivasyonunu veya ham açıklamasını değiştirmemiştir.
- Yalnızca hangi bölgelerin kullanıcıya gösterileceğini ve renk ölçeğini değiştirmiştir.
- Bu nedenle yeniden normalize edilmiş görüntü daha detection-specific görünebilir; fakat bu görünüm tek başına yöntemin daha doğru olduğu anlamına gelmez.
- Ham CAM'in korunması, görsel sunum ile modelden gelen asıl sinyali birbirinden ayırmamızı sağlar.
- Raporlamada `renormalize: true/false` bilgisi mutlaka belirtilmelidir.
- Yeniden normalize edilmiş overlay'in görsel yorumu henüz tamamlanmadı.
## Bugünkü genel çıkarımlar

- Grad-CAM'in bir kutuyu açıklayabilmesi için önce NMS sonrası kutunun doğru ham adayla eşleştirilmesi gerekir.
- Class-score target, “Model bu bölgeyi neden horse olarak değerlendirdi?” sorusuna yöneliktir; objectness veya kutu koordinatlarını açıklamaz.
- Güzel görünen bir heatmap tek başına doğru açıklama olduğunu kanıtlamaz.
- Feature layer seçimi, tensor boyutları, gradyan akışı ve preprocessing ayarları heatmap'in anlamını doğrudan etkiler.
- Çalışan her deney; kullanılan model, target, katmanlar, eşik, kutu, ham aday ve normalizasyon ayarlarıyla birlikte kaydedilmelidir.

## Sonraki adımlar

- Kutu içi yeniden normalizasyonu ayrı bir deney olarak uygulayıp sonucu yanıltıcı biçimde güzelleştirip güzelleştirmediğini tartışmak.
- Deney sonuçlarını bu Gün 4 notuna eklemeye devam etmek.
