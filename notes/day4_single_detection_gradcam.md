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
- Heatmap'in gerçekten kendi yeşil kutusunu takip edip etmediğine dair görsel yorum henüz yazılmadı.
## Bugünkü genel çıkarımlar

- Grad-CAM'in bir kutuyu açıklayabilmesi için önce NMS sonrası kutunun doğru ham adayla eşleştirilmesi gerekir.
- Class-score target, “Model bu bölgeyi neden horse olarak değerlendirdi?” sorusuna yöneliktir; objectness veya kutu koordinatlarını açıklamaz.
- Güzel görünen bir heatmap tek başına doğru açıklama olduğunu kanıtlamaz.
- Feature layer seçimi, tensor boyutları, gradyan akışı ve preprocessing ayarları heatmap'in anlamını doğrudan etkiler.
- Çalışan her deney; kullanılan model, target, katmanlar, eşik, kutu, ham aday ve normalizasyon ayarlarıyla birlikte kaydedilmelidir.

## Sonraki adımlar

- İlk heatmap'in seçilen yeşil kutuyla ilişkisini görsel olarak yorumlamak.
- `--img-size 640` deneylerinin görsel sonuçlarını karşılaştırmak.
- İki heatmap'in gerçekten kendi seçili kutularını takip edip etmediğini karşılaştırmak.
- Deney sonuçlarını bu Gün 4 notuna eklemeye devam etmek.
