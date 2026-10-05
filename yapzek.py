# type: ignore
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    adjusted_rand_score,
    confusion_matrix
)
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🌍 ÜLKELERİN KALKINMA GÖSTERGELERİNE GÖRE KÜMELENDİRME ANALİZİ".center(80))
print("="*80)

# ========================================
# 1. VERİYİ YÜKLE VE TEMİZLE
# ========================================
print("\n📂 Veri yükleniyor...")
df_full = pd.read_csv('hdr_general.csv', encoding='cp1252')
df_2022 = df_full[df_full['year'] == 2022].copy()
df_2022_clean = df_2022[~df_2022['country'].str.contains('SAR|Macao', case=False, na=False)].copy()
print(f"✅ 2022 yılı verisi yüklendi")
print(f"✅ SAR bölgeleri çıkarıldı")

features = [
    'hdi', 'life_expectancy', 'mean_yr_school', 'gross_inc_percap',
    'expec_yr_school', 'secondary_education_f_%', 'gender_inequality',
    'labour_participation_f_%', 'seats_in_parliament_f_%',
    'co2_emission_tons', 'mat_footprint_percap_tons', 'gender_development'
]

df = df_2022_clean[['country', 'iso3'] + features].dropna().copy()
print(f"✅ Toplam {len(df)} ülke (eksik veri yok)")

# ========================================
# 2. VERİYİ STANDARDİZE ET
# ========================================
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Veri standardize edildi")

# ========================================
# 3. ELBOW METHOD
# ========================================
print("\n🔍 Elbow Method çalışıyor...")
wcss = []
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(k_values, wcss, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Küme Sayısı (k)', fontsize=12)
ax1.set_ylabel('WCSS', fontsize=12)
ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=4, color='red', linestyle='--', alpha=0.5, label='Optimal k=4')
ax1.legend()

ax2.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.set_xlabel('Küme Sayısı (k)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score vs k', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=4, color='red', linestyle='--', alpha=0.5, label='Optimal k=4')
ax2.legend()

plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=300, bbox_inches='tight')
print("✅ Elbow + Silhouette: elbow_silhouette.png")

# ========================================
# 4. DENDOGRAM
# ========================================
print("🌳 Dendogram oluşturuluyor...")
plt.figure(figsize=(20, 8))
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix, labels=df['country'].values, leaf_font_size=6)
plt.axhline(y=30, color='red', linestyle='--', label='Kesim Noktası (k=4)')
plt.xlabel('Ülkeler', fontsize=12)
plt.ylabel('Uzaklık', fontsize=12)
plt.title('Dendogram - Hierarchical Clustering', fontsize=14, fontweight='bold')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('dendogram.png', dpi=300, bbox_inches='tight')
print("✅ Dendogram: dendogram.png")

# ========================================
# 5. K-MEANS & AGGLOMERATIVE
# ========================================
optimal_k = 4
print(f"\n🎯 Kümeleme (k={optimal_k})...")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

agg = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
df['Agg_Cluster'] = agg.fit_predict(X_scaled)

# Küme isimleri
cluster_means = df.groupby('KMeans_Cluster')['hdi'].mean().sort_values()
cluster_names = {
    cluster_means.index[0]: 'Düşük Kalkınma', 
    cluster_means.index[1]: 'Orta-Düşük Kalkınma',
    cluster_means.index[2]: 'Orta-Yüksek Kalkınma',
    cluster_means.index[3]: 'Yüksek Kalkınma'
}
df['Küme_Adı'] = df['KMeans_Cluster'].map(cluster_names)

color_map = {
    'Yüksek Kalkınma': '#10b981',
    'Orta-Yüksek Kalkınma': '#3b82f6',
    'Orta-Düşük Kalkınma': '#f59e0b',
    'Düşük Kalkınma': '#ef4444'
}

# ========================================
# 6. GÜVENİLİRLİK METRİKLERİ
# ========================================
print("\n" + "="*80)
print("📊 GÜVENİLİRLİK METRİKLERİ ANALİZİ")
print("="*80)

# Metrikler
kmeans_silhouette = silhouette_score(X_scaled, df['KMeans_Cluster'])
agg_silhouette = silhouette_score(X_scaled, df['Agg_Cluster'])

kmeans_davies = davies_bouldin_score(X_scaled, df['KMeans_Cluster'])
agg_davies = davies_bouldin_score(X_scaled, df['Agg_Cluster'])

kmeans_calinski = calinski_harabasz_score(X_scaled, df['KMeans_Cluster'])
agg_calinski = calinski_harabasz_score(X_scaled, df['Agg_Cluster'])

ari_score = adjusted_rand_score(df['KMeans_Cluster'], df['Agg_Cluster'])

agreement = (df['KMeans_Cluster'] == df['Agg_Cluster']).sum()
agreement_pct = (agreement / len(df)) * 100

# Tablo
metrics_data = {
    'Metrik': [
        'Silhouette Score',
        'Davies-Bouldin Index',
        'Calinski-Harabasz Score',
        'Adjusted Rand Index',
        'Algoritma Uyumu (%)'
    ],
    'K-Means': [
        f'{kmeans_silhouette:.3f}',
        f'{kmeans_davies:.3f}',
        f'{kmeans_calinski:.1f}',
        '-',
        f'{agreement_pct:.1f}%'
    ],
    'Agglomerative': [
        f'{agg_silhouette:.3f}',
        f'{agg_davies:.3f}',
        f'{agg_calinski:.1f}',
        '-',
        '-'
    ],
    'Karşılaştırma': [
        f'{ari_score:.3f}',
        '-',
        '-',
        f'{ari_score:.3f}',
        f'{agreement}/{len(df)} ülke'
    ],
    'Yorum': [
        '↑ İyi (0.5+ mükemmel)',
        '↓ İyi (0-1 arası)',
        '↑ İyi (>100 iyi)',
        '1.0 = Tam uyum',
        'İki algoritma uyumu'
    ]
}

metrics_df = pd.DataFrame(metrics_data)

print("\n" + metrics_df.to_string(index=False))

# ========================================
# 7. CONFUSION MATRIX
# ========================================
print("\n📊 Confusion Matrix oluşturuluyor...")

cm = confusion_matrix(df['KMeans_Cluster'], df['Agg_Cluster'])

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f'Agg-{i}' for i in range(optimal_k)],
            yticklabels=[f'KMeans-{i}' for i in range(optimal_k)],
            cbar_kws={'label': 'Ülke Sayısı'})
plt.title('Confusion Matrix: K-Means vs Agglomerative\n(Köşegen = Uyum)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Agglomerative Clustering', fontsize=12)
plt.ylabel('K-Means Clustering', fontsize=12)

# Uyum yüzdesi ekleme
for i in range(optimal_k):
    for j in range(optimal_k):
        value = cm[i, j]
        if value > 0:
            percentage = (value / cm.sum()) * 100
            plt.text(j + 0.5, i + 0.7, f'(%{percentage:.1f})', 
                    ha='center', va='center', fontsize=9, color='red')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion Matrix: confusion_matrix.png")

# ========================================
# 8. KÜME DAĞILIMI VE BOYUT ANALİZİ
# ========================================
print("\n📊 Küme dağılım analizi...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Küme boyutları
cluster_counts = df['Küme_Adı'].value_counts().sort_index()
axes[0, 0].bar(range(len(cluster_counts)), cluster_counts.values, 
               color=[color_map[name] for name in cluster_counts.index])
axes[0, 0].set_xticks(range(len(cluster_counts)))
axes[0, 0].set_xticklabels(cluster_counts.index, rotation=45, ha='right')
axes[0, 0].set_ylabel('Ülke Sayısı', fontsize=11)
axes[0, 0].set_title('Küme Büyüklükleri', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2. HDI dağılımı
for kume_adi in sorted(cluster_names.values()):
    df_kume = df[df['Küme_Adı'] == kume_adi]
    axes[0, 1].hist(df_kume['hdi'], bins=10, alpha=0.6, 
                    label=kume_adi, color=color_map[kume_adi])
axes[0, 1].set_xlabel('HDI', fontsize=11)
axes[0, 1].set_ylabel('Frekans', fontsize=11)
axes[0, 1].set_title('Kümelere Göre HDI Dağılımı', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Yaşam Beklentisi vs Gelir
for kume_adi in sorted(cluster_names.values()):
    df_kume = df[df['Küme_Adı'] == kume_adi]
    axes[1, 0].scatter(df_kume['life_expectancy'], df_kume['gross_inc_percap'],
                       label=kume_adi, alpha=0.6, s=80, color=color_map[kume_adi])
axes[1, 0].set_xlabel('Yaşam Beklentisi (yıl)', fontsize=11)
axes[1, 0].set_ylabel('Kişi Başı Gelir ($)', fontsize=11)
axes[1, 0].set_title('Yaşam Beklentisi vs Gelir', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Eğitim vs Cinsiyet Eşitsizliği
for kume_adi in sorted(cluster_names.values()):
    df_kume = df[df['Küme_Adı'] == kume_adi]
    axes[1, 1].scatter(df_kume['mean_yr_school'], df_kume['gender_inequality'],
                       label=kume_adi, alpha=0.6, s=80, color=color_map[kume_adi])
axes[1, 1].set_xlabel('Ortalama Eğitim Yılı', fontsize=11)
axes[1, 1].set_ylabel('Cinsiyet Eşitsizliği', fontsize=11)
axes[1, 1].set_title('Eğitim vs Cinsiyet Eşitsizliği', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kume_dagilimlari.png', dpi=300, bbox_inches='tight')
print("✅ Küme dağılımları: kume_dagilimlari.png")

# ========================================
# 9. KÜME İÇİ/ARASI MESAFE ANALİZİ
# ========================================
print("\n📏 Küme mesafe analizi...")

intra_distances = []
inter_distances = []

for k in range(optimal_k):
    cluster_data = X_scaled[df['KMeans_Cluster'] == k]
    if len(cluster_data) > 1:
        center = cluster_data.mean(axis=0)
        distances = np.sqrt(((cluster_data - center) ** 2).sum(axis=1))
        intra_distances.append(distances.mean())
    
    for k2 in range(k + 1, optimal_k):
        cluster_data2 = X_scaled[df['KMeans_Cluster'] == k2]
        if len(cluster_data) > 0 and len(cluster_data2) > 0:
            center1 = cluster_data.mean(axis=0)
            center2 = cluster_data2.mean(axis=0)
            dist = np.sqrt(((center1 - center2) ** 2).sum())
            inter_distances.append(dist)

avg_intra = np.mean(intra_distances)
avg_inter = np.mean(inter_distances)
separation_ratio = avg_inter / avg_intra

print(f"\n📍 Ortalama Küme İçi Mesafe: {avg_intra:.3f}")
print(f"📍 Ortalama Küme Arası Mesafe: {avg_inter:.3f}")
print(f"📍 Ayrım Oranı (Inter/Intra): {separation_ratio:.3f} (↑ Daha iyi)")

# ========================================
# 10. İNTERAKTİF HARİTA (AYNI)
# ========================================
print("\n🗺️ İnteraktif harita oluşturuluyor...")

df['Hover_Text'] = df.apply(lambda row: 
    f"<b>{row['country']}</b><br>" +
    f"<b>Küme:</b> {row['Küme_Adı']}<br><br>" +
    f"<b>📊 TEMEL:</b><br>" +
    f"   • HDI: {row['hdi']:.3f}<br>" +
    f"   • Yaşam: {row['life_expectancy']:.1f} yıl<br>" +
    f"   • Eğitim: {row['mean_yr_school']:.1f} yıl<br>" +
    f"   • Gelir: ${row['gross_inc_percap']:,.0f}<br><br>" +
    f"<b>🎓 EĞİTİM:</b><br>" +
    f"   • Beklenen: {row['expec_yr_school']:.1f} yıl<br>" +
    f"   • Kadın Lise: %{row['secondary_education_f_%']:.1f}<br><br>" +
    f"<b>♀️ CİNSİYET:</b><br>" +
    f"   • Eşitsizlik: {row['gender_inequality']:.3f}<br>" +
    f"   • Kadın İşgücü: %{row['labour_participation_f_%']:.1f}<br>" +
    f"   • Parlamentoda: %{row['seats_in_parliament_f_%']:.1f}<br><br>" +
    f"<b>🌱 ÇEVRE:</b><br>" +
    f"   • CO2: {row['co2_emission_tons']:.1f} ton<br>" +
    f"   • Ayak İzi: {row['mat_footprint_percap_tons']:.1f} ton"
, axis=1)

fig = go.Figure()

for kume_adi in sorted(cluster_names.values()):
    df_kume = df[df['Küme_Adı'] == kume_adi]
    fig.add_trace(go.Choropleth(
        locations=df_kume['iso3'],
        z=df_kume['KMeans_Cluster'],
        text=df_kume['Hover_Text'],
        colorscale=[[0, color_map[kume_adi]], [1, color_map[kume_adi]]],
        showscale=False,
        hovertemplate='%{text}<extra></extra>',
        name=kume_adi,
        marker_line_color='#ffffff',
        marker_line_width=1.5
    ))

fig.update_layout(
    title='<b>🌍 Dünya Ülkeleri Kalkınma Analizi - Güvenilirlik Doğrulanmış</b>',
    geo=dict(projection_type='natural earth'),
    width=1800, height=1000
)

fig.write_html('harita_FINAL_guvenilir.html')
print("✅ Harita: harita_FINAL_guvenilir.html")

# ========================================
# 11. EXCEL RAPORU
# ========================================
print("\n💾 Excel raporu oluşturuluyor...")

output_file = 'kumelendirme_GUVENILIRLIK_RAPORU.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 1. Güvenilirlik metrikleri
    metrics_df.to_excel(writer, sheet_name='Güvenilirlik Metrikleri', index=False)
    
    # 2. Confusion Matrix
    cm_df = pd.DataFrame(cm, 
                         columns=[f'Agg-{i}' for i in range(optimal_k)],
                         index=[f'KMeans-{i}' for i in range(optimal_k)])
    cm_df.to_excel(writer, sheet_name='Confusion Matrix')
    
    # 3. Tüm sonuçlar
    df[['country', 'iso3', 'Küme_Adı', 'KMeans_Cluster', 'Agg_Cluster'] + features].to_excel(
        writer, sheet_name='Tüm Sonuçlar', index=False)
    
    # 4. Küme istatistikleri
    summary_data = []
    for kume_adi in sorted(cluster_names.values()):
        df_kume = df[df['Küme_Adı'] == kume_adi]
        summary_data.append({
            'Küme': kume_adi,
            'Ülke Sayısı': len(df_kume),
            'Oran (%)': f"{(len(df_kume)/len(df))*100:.1f}",
            'Ort. HDI': f"{df_kume['hdi'].mean():.3f}",
            'Ort. Yaşam': f"{df_kume['life_expectancy'].mean():.1f}",
            'Ort. Eğitim': f"{df_kume['mean_yr_school'].mean():.1f}",
            'Ort. Gelir': f"${df_kume['gross_inc_percap'].mean():,.0f}"
        })
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Küme Özeti', index=False)

print(f"✅ {output_file}")

# ========================================
# 12. KONSOL SONUÇ
# ========================================
print("\n" + "="*80)
print("✅ GÜVENİLİRLİK ANALİZİ TAMAMLANDI!")
print("="*80)
print("\n📁 Oluşturulan Dosyalar:")
print("   1. elbow_silhouette.png       - Optimal k belirleme")
print("   2. dendogram.png               - Hiyerarşik kümeleme")
print("   3. confusion_matrix.png        - Algoritma karşılaştırması")
print("   4. kume_dagilimlari.png        - Küme analizi görselleri")
print("   5. harita_FINAL_guvenilir.html - İnteraktif harita")
print("   6. kumelendirme_GUVENILIRLIK_RAPORU.xlsx - Detaylı rapor")

print("\n🎯 Güvenilirlik Özeti:")
print(f"   ✓ Silhouette Score: {kmeans_silhouette:.3f} (0.5+ mükemmel)")
print(f"   ✓ Davies-Bouldin: {kmeans_davies:.3f} (düşük = iyi)")
print(f"   ✓ Calinski-Harabasz: {kmeans_calinski:.1f} (yüksek = iyi)")
print(f"   ✓ Algoritma Uyumu: %{agreement_pct:.1f}")
print(f"   ✓ Adjusted Rand Index: {ari_score:.3f}")
print(f"   ✓ Ayrım Oranı: {separation_ratio:.3f}")

print("\n💡 Yorum:")
if kmeans_silhouette > 0.5 and agreement_pct > 80:
    print("   🟢 Kümeleme çok güvenilir! Tüm metrikler mükemmel.")
elif kmeans_silhouette > 0.3 and agreement_pct > 60:
    print("   🟡 Kümeleme güvenilir. Metrikler kabul edilebilir seviyede.")
else:
    print("   🟠 Kümeleme orta güvenilirlikte. Daha fazla analiz gerekebilir.")

print("\n" + "="*80 + "\n")