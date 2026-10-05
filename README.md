# ulke-kalkınma-kumeleme-analizi
# Ülkelerin Kalkınma Göstergelerine Göre Kümelendirme Analizi

Python kullanarak ülkeleri kalkınma göstergelerine (HDI, yaşam beklentisi, 
eğitim süresi, gelir, cinsiyet eşitliği, karbon emisyonu gibi) göre 
gruplandıran bir veri analizi projesi.

## Ne Yapıyor
- K-Means ve Hiyerarşik (Agglomerative) kümeleme algoritmalarıyla ülkeleri 
  4 kalkınma grubuna ayırıyor
- Optimal küme sayısını Elbow Method ve Silhouette Score ile belirliyor
- İki algoritmanın sonuçlarını istatistiksel metriklerle (Silhouette, 
  Davies-Bouldin, Adjusted Rand Index) karşılaştırıyor
- Sonuçları interaktif bir dünya haritası ve detaylı Excel raporu olarak sunuyor

## Kullanılan Teknolojiler
Python, pandas, scikit-learn, matplotlib, seaborn, plotly

## Çıktılar
- İnteraktif dünya haritası (ülkeler kalkınma grubuna göre renklendirilmiş)
- Kümeleme görselleştirmeleri (dendrogram, elbow curve, confusion matrix)
- Detaylı Excel raporu (küme özeti, güvenilirlik metrikleri)

- ## Veri
`hdr_general.csv` dosyası, ülkelerin yıllık kalkınma göstergelerini içerir 
(HDI, yaşam beklentisi, eğitim, gelir, cinsiyet eşitliği, karbon emisyonu vb.).
