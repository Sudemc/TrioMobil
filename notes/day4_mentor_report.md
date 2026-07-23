# Gün 4 — Kısa Mentor Raporu

- YOLOv7'de NMS sonrasında seçilen tek bir tespiti doğru ham model adayıyla eşleştiren ve bu adayın class skorunu açıklayan CAM pipeline'ını tamamladım.
- Tensor boyutu, Python import yolu ve YOLOv7 Detect katmanındaki in-place gradient sorunlarını inceleyip çözdüm; `third_party` kaynak kodlarını değiştirmedim.
- Deneyleri `640 × 640` giriş boyutunda, class-score target ve gerçek detection ölçeği olan layer `104` ile tekrarlanabilir hâle getirdim.
- Klasik GradCAM'in iki farklı horse tespiti için neredeyse aynı haritayı ürettiğini hem görsel hem sayısal olarak gösterdim: Pearson korelasyonu yaklaşık `1.0`, en sıcak %10 örtüşmesi `1.0` çıktı.
- Layer `102` ve `103` CAM'lerinin seçilen ham adaylar için boş kaldığını, bu adayların gradient yolunun layer `104` üzerinden geçtiğini doğruladım.
- HiResCAM ile uzamsal gradyanı koruduğumda iki tespit birbirinden ayrıldı: sıcak bölgelerin örtüşmesi `0.0` oldu ve her haritanın en sıcak pikselleri kendi seçili kutusunda kaldı.
- Kutu içi yeniden normalizasyonun ham CAM'i değiştirmediğini, yalnızca kutu dışındaki aktivasyonu gizleyerek sonucu daha ikna edici gösterebildiğini deneyle gösterdim.
- Bu örnek için tek-tespit açıklamasında HiResCAM + layer `104` yaklaşımının klasik GradCAM'den daha uygun olduğu sonucuna vardım; buna rağmen heatmap'in hatanın kesin nedenini değil, skorla ilişkili bölgeyi gösterdiğini not ettim.
- Pipeline ve karşılaştırma araçları için toplam `14` otomatik test çalıştırdım ve bütün testler geçti.

## Sonraki çalışma

Bir sonraki aşamada aynı pipeline'ı gerçek bir false-positive görüntü üzerinde çalıştıracağım. Normal görüntü, ham CAM, HiResCAM overlay, metadata ve kısa hipotezi birlikte kaydedeceğim. Bunun için gerçek false-positive örnek görüntüsüne ve varsa fine-tuned YOLOv7 ağırlığına ihtiyaç var.
