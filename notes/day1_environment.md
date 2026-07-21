# Gün 1 — Ortam ve YOLOv7

## Bilgisayar
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- VRAM: 8 GB
- NVIDIA sürücüsü: 581.42
- Python: 3.10.11
- PyTorch: 2.5.1+cu124
- CUDA: 12.4
- OpenCV: 4.11.0
- NumPy: 1.23.5
- Git: 2.51.0

## Bugünkü hedef
YOLOv7'nin normal nesne tespit akışını çalıştırmak ve anlamak.

## Öğrenme soruları
1. Bir görüntü modele girmeden önce nasıl hazırlanır?
2. YOLOv7 modelinin ham çıktısı nedir?
3. Objectness, sınıf skoru ve bounding box ne anlama gelir?
4. NMS neden gerekir?

## Karşılaşılan hata — YOLOv7 ağırlık indirme

`detect.py` ilk çalıştırmada `models/yolov7.pt` dosyasını otomatik indirmeye çalıştı.

Ancak indirme adımı başarısız oldu:

- Kod GitHub API yanıtında `assets` alanını bekledi.
- API yanıtı beklenen biçimde gelmedi.
- Kodun yedek yöntemi `git tag` komutunu kullandı.
- YOLOv7 reposu `--depth 1` ile indirildiği için yerelde Git tag bilgisi yoktu.
- Bu nedenle `IndexError: list index out of range` hatası oluştu.

Çözüm:
Resmî YOLOv7 release bağlantısından model ağırlığını doğrudan `models/yolov7.pt` konumuna indirmek.

Öğrendiğim:
Bir hata sadece model veya GPU kaynaklı olmayabilir. Eski yardımcı kodlar, GitHub API'si veya Git geçmişiyle ilgili varsayımlar da hata oluşturabilir.

İkinci hata: İlk indirilen `yolov7.pt` yalnızca 23.7 MB olduğu için bozuktu. Model ağırlığı tamamlanmamış olduğunda PyTorch zip arşivinin merkezi dizinini okuyamadı. Eksik dosya saklanıp, model ağırlığı yeniden indirildi.

## İlk YOLOv7 inference sonucu

Komut:

```powershell
python third_party\yolov7\detect.py --weights models\yolov7.pt --source third_party\yolov7\inference\images\horses.jpg --img-size 640 --conf-thres 0.25 --device 0 --project outputs --name day1_detection --exist-ok```
Sonuç:

Sonuç:

- Tespit edilen nesne: 5 horse
- Inference süresi: 56.2 ms
- NMS süresi: 292.4 ms
- Toplam süre: 0.732 saniye
- Çıktı görseli: `outputs/day1_detection/horses.jpg`

## Ortam doğrulaması

PyTorch, CUDA'yı ve RTX 4060 GPU'yu başarıyla gördü.

```text
GPU bulundu mu?: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU ```

## Confidence threshold deneyi

Aynı görüntü, aynı model ve aynı NMS ayarlarıyla farklı confidence eşiklerinde çalıştırıldı.

| Confidence eşiği | Kalan tespit |
|---:|---:|
| 0.25 | 5 horse |
| 0.50 | 5 horse |
| 0.85 | 3 horse |

0.50 deneyinde NMS sonrasında kalan confidence değerleri:

```text
0.692, 0.803, 0.859, 0.943, 0.958

