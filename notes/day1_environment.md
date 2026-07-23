# Gün 1 — Ortam ve normal YOLOv7 inference

## Hedef

YOLOv7'nin normal nesne tespit akışını çalıştırmak, GPU ortamını doğrulamak ve confidence threshold ile NMS'in sonuçlara etkisini gözlemlemek.

## Bilgisayar ve yazılım ortamı

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- VRAM: 8 GB
- NVIDIA sürücüsü: 581.42
- Python: 3.10.11
- PyTorch: 2.5.1+cu124
- PyTorch CUDA runtime: 12.4
- OpenCV: 4.11.0
- NumPy: 1.23.5
- Git: 2.51.0

## GPU doğrulaması

PyTorch, CUDA'yı ve RTX 4060 GPU'yu başarıyla gördü.

```text
GPU bulundu mu?: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

## Kullanılan kaynaklar

- Resmî YOLOv7 reposu: `third_party/yolov7`
- Model ağırlığı: `models/yolov7.pt`
- Örnek görüntü: `third_party/yolov7/inference/images/horses.jpg`

## İlk YOLOv7 inference sonucu

Kullanılan komut:

```powershell
python third_party\yolov7\detect.py --weights models\yolov7.pt --source third_party\yolov7\inference\images\horses.jpg --img-size 640 --conf-thres 0.25 --device 0 --project outputs --name day1_detection --exist-ok
```

Sonuç:

- Tespit edilen nesne: 5 horse
- Inference süresi: 56.2 ms
- NMS süresi: 292.4 ms
- Toplam süre: 0.732 saniye
- Çıktı görseli: `outputs/day1_detection/horses.jpg`

Not: Inference ve NMS süreleri sistem yükü, GPU hazırlığı ve tekrar çalıştırmalara göre değişebilir.

## Confidence threshold deneyi

Aynı görüntü, aynı model ve aynı NMS ayarlarıyla farklı confidence eşiklerinde çalıştırıldı.

| Confidence eşiği | Kalan tespit |
|---:|---:|
| 0.25 | 5 horse |
| 0.50 | 5 horse |
| 0.85 | 3 horse |

0.50 deneyi için kullanılan komut:

```powershell
python third_party\yolov7\detect.py --weights models\yolov7.pt --source third_party\yolov7\inference\images\horses.jpg --img-size 640 --conf-thres 0.50 --device 0 --project outputs --name day1_conf_050 --save-txt --save-conf
```

0.85 deneyi için kullanılan komut:

```powershell
python third_party\yolov7\detect.py --weights models\yolov7.pt --source third_party\yolov7\inference\images\horses.jpg --img-size 640 --conf-thres 0.85 --device 0 --project outputs --name day1_conf_085 --save-txt --save-conf
```

0.50 deneyinde NMS sonrasında kalan confidence değerleri:

```text
0.692, 0.803, 0.859, 0.943, 0.958
```

0.85 eşiğinde 0.692 ve 0.803 skorlu tespitler elendi; 0.859, 0.943 ve 0.958 skorlu üç tespit kaldı.

Sonuç: Confidence threshold, düşük güvenli tespitleri filtreler. Eşiği artırmak false positive sayısını azaltabilir; fakat gerçek nesnelerin de elenmesine neden olabilir.

## Öğrenme sorularının cevapları

### Bir görüntü modele girmeden önce nasıl hazırlanır?

Görüntü okunur, modelin beklediği boyuta getirilir, NumPy dizisinden PyTorch tensor'una dönüştürülür, GPU'ya aktarılır ve piksel değerleri 0-255 aralığından 0-1 aralığına normalize edilir.

### YOLOv7'nin ham çıktısı nedir?

YOLOv7, görüntü için çok sayıda aday kutu üretir. Her aday kutu; konum bilgisi, objectness skoru ve sınıf skorları içerir. Bu ham adaylar henüz nihai sonuç değildir.

### Objectness, sınıf skoru ve bounding box ne anlama gelir?

- Bounding box: Modelin nesnenin konumunu belirlediği dikdörtgen alan.
- Objectness: Kutuda herhangi bir nesne bulunduğuna dair güven.
- Class score: Nesnenin belirli bir sınıfa ait olduğuna dair güven.

### NMS neden gerekir?

Model aynı nesne için birbirini örten birçok aday kutu üretebilir. NMS, yüksek güvenli kutuyu korur ve onunla fazla çakışan kutuları eler. Görüntüde yalnızca NMS sonrasında kalan nihai kutular gösterilir.

## Karşılaşılan hatalar ve çözümler

### Otomatik model indirme hatası

`detect.py`, ilk çalıştırmada `models/yolov7.pt` dosyasını otomatik indirmeye çalıştı.

Ancak kod GitHub API yanıtında `assets` alanını bekledi. API yanıtı beklenen biçimde gelmediğinde kod yedek yöntem olarak `git tag` komutunu kullanmaya çalıştı.

YOLOv7 reposu `--depth 1` ile klonlandığı için yalnızca son commit indirildi; yerelde Git tag bilgisi bulunmadı. Bu nedenle kod boş tag listesinin son elemanına erişmeye çalışırken şu hata oluştu:

```text
IndexError: list index out of range
```

Çözüm: Model ağırlığı resmî YOLOv7 release bağlantısından doğrudan `models/yolov7.pt` konumuna indirildi.

### Eksik model ağırlığı

İlk indirilen `yolov7.pt` dosyası yalnızca yaklaşık 23.7 MB idi. Dosya eksik olduğu için PyTorch modeli yüklerken şu hatayı verdi:

```text
PytorchStreamReader failed reading zip archive:
failed finding central directory
```

Çözüm: Eksik dosya `yolov7.incomplete.pt` olarak saklandı ve model yeniden indirildi. Çalışan model ağırlığının boyutu 75,587,165 byte oldu.

## Öğrendiklerim

- Bir hata yalnızca model veya GPU kaynaklı olmayabilir; Git, GitHub API, paket sürümleri ve indirilen dosyanın bütünlüğü de sorun oluşturabilir.
- Normal inference sırasında model hızlı çalışması için gradyan hesaplamaz.
- Grad-CAM aşamasında ise gradyan gerekli olacağı için normal `detect.py` akışı doğrudan heatmap üretmek için yeterli değildir.
- Heatmap çalışmasına geçmeden önce modelin normal tespit, confidence threshold ve NMS davranışını anlamak gerekir.

## Git kaydı

İlk gün çalışması şu commit ile kaydedildi:

```text
893b65c docs: complete day 1 YOLOv7 baseline
```