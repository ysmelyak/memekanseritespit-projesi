import os
import cv2
import numpy as np
import pandas as pd
import sys

try:
    import pywt
except ImportError:
    print("UYARI: 'pywt' (PyWavelets) yüklü değil. Wavelet özellikleri atlanacak.")

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from scipy.stats import skew, kurtosis

# ==========================================
# AYARLAR
# ==========================================
BASE_PATH = r"C:\Users\user\Desktop\goruntuproje"
DATASET_PATH = os.path.join(BASE_PATH, "veri_seti") 
OUTPUT_PATH  = os.path.join(BASE_PATH, "final_outputs")

if not os.path.exists(DATASET_PATH):
    alt_path = os.path.join(BASE_PATH, "Dataset")
    if os.path.exists(alt_path):
        DATASET_PATH = alt_path

GLCM_DISTANCES = [1]
GLCM_ANGLES    = [0, np.pi/4, np.pi/2, 3*np.pi/4] 
GLCM_LEVELS    = 32  
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
HOG_RESIZE = (64, 64)

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ==========================================
# ÖN İŞLEME VE YARDIMCI FONKSİYONLAR
# ==========================================
def preprocess_to_gray(img):
    if img is None: return None, None
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    gray = cv2.resize(gray, (128, 128)) 
    gray = cv2.medianBlur(gray, 3) 
    gray = gray.astype(np.uint8)

    if GLCM_LEVELS < 256:
        gray_q = (gray / (256 / GLCM_LEVELS)).astype(np.uint8)
    else:
        gray_q = gray.copy()
    return gray, gray_q

def write_arff(data_list, filename, relation="Project"):
    if not data_list: return
    df = pd.DataFrame(data_list).fillna(0)
    
    if "class" in df.columns:
        cols = [c for c in df.columns if c != "class"] + ["class"]
        df = df[cols]
    else:
        cols = df.columns.tolist()

    numeric_cols = [c for c in cols if c != "class"]
    
    for c in numeric_cols:
        mn, mx = df[c].min(), df[c].max()
        if mx > mn: df[c] = (df[c] - mn) / (mx - mn)
    
    path = os.path.join(OUTPUT_PATH, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"@RELATION {relation}\n\n")
        for c in numeric_cols: f.write(f"@ATTRIBUTE {c} NUMERIC\n")
        
        if "class" in df.columns:
            classes = sorted(df["class"].unique())
            f.write(f"@ATTRIBUTE class {{{','.join(classes)}}}\n\n@DATA\n")
        else:
             f.write(f"\n@DATA\n")

        for _, row in df.iterrows():
            line = [f"{row[c]:.6f}" for c in numeric_cols]
            if "class" in df.columns:
                line.append(str(row["class"]))
            f.write(",".join(line) + "\n")
    print(f"✅ KAYDEDİLDİ: {filename}")

# ==========================================
# ÖZELLİK ÇIKARMA FONKSİYONLARI
# ==========================================

def extract_glcm_features(gray_q):
    feats = {}
    try:
        glcm = graycomatrix(gray_q, distances=GLCM_DISTANCES, angles=GLCM_ANGLES, 
                            symmetric=True, normed=True, levels=GLCM_LEVELS)
        props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
        for p in props:
            val = graycoprops(glcm, p).mean() 
            feats[f"glcm_{p}"] = float(val)
    except: pass
    return feats

def extract_lbp_features(gray):
    feats = {}
    try:
        lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        feats["lbp_energy"] = float(np.sum(hist ** 2))
        feats["lbp_entropy"] = float(-np.sum(hist * np.log2(hist + 1e-7)))
        feats["lbp_mean"] = float(np.mean(lbp))
        feats["lbp_std"] = float(np.std(lbp))
    except: pass
    return feats

def extract_hog_compact(gray):
    feats = {}
    try:
        g = cv2.resize(gray, HOG_RESIZE)
        vec = hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True, channel_axis=None)
        feats["hog_mean"] = float(np.mean(vec))
        feats["hog_std"] = float(np.std(vec))
        feats["hog_skew"] = float(skew(vec))
        feats["hog_kurtosis"] = float(kurtosis(vec))
    except: pass
    return feats

def extract_lcp_features(gray):
    feats = {}
    try:
        kernel = np.ones((3,3), np.uint8)
        local_min = cv2.erode(gray, kernel)
        local_max = cv2.dilate(gray, kernel)
        contrast_img = cv2.subtract(local_max, local_min).astype(np.float32)
        feats["lcp_mean"] = float(np.mean(contrast_img))
        feats["lcp_std"]  = float(np.std(contrast_img))
        feats["lcp_energy"] = float(np.sum(contrast_img**2) / (gray.size + 1e-7))
        feats["lcp_entropy"] = float(-np.sum((contrast_img/255.0) * np.log2((contrast_img/255.0) + 1e-7)))
    except: pass
    return feats

def extract_shape_features(gray):
    feats = {'shape_area': 0.0, 'shape_perimeter': 0.0, 'shape_circularity': 0.0, 'shape_aspect_ratio': 0.0}
    try:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            x,y,w,h = cv2.boundingRect(c)
            
            feats['shape_area'] = float(area)
            feats['shape_perimeter'] = float(perimeter)
            if perimeter > 0:
                feats['shape_circularity'] = float((4 * np.pi * area) / (perimeter * perimeter))
            if h > 0:
                feats['shape_aspect_ratio'] = float(w / h)
    except: pass
    return feats

def extract_phog_features(gray):
    feats = {}
    try:
        g = cv2.resize(gray, HOG_RESIZE)
        vec0 = hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True, channel_axis=None)
        feats["phog_L0_mean"] = float(np.mean(vec0))
        
        h, w = g.shape
        h2, w2 = h//2, w//2
        quadrants = [g[0:h2, 0:w2], g[0:h2, w2:w], g[h2:h, 0:w2], g[h2:h, w2:w]]
        stats = []
        for q in quadrants:
            vec_q = hog(q, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True, channel_axis=None)
            stats.append(np.mean(vec_q))
        
        feats["phog_L1_mean"] = float(np.mean(stats))
        feats["phog_L1_std"] = float(np.std(stats))
    except: pass
    return feats

def extract_soft_histogram(gray):
    feats = {}
    try:
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist = hist.flatten()
        
        for i, val in enumerate(hist):
            feats[f"softhist_bin_{i}"] = float(val)
            
        feats["softhist_mean"] = float(np.mean(hist))
        feats["softhist_std"] = float(np.std(hist))
        feats["softhist_energy"] = float(np.sum(hist**2))
        feats["softhist_entropy"] = float(-np.sum(hist * np.log2(hist + 1e-7)))
    except: pass
    return feats

def extract_statistical_features(gray):
    feats = {}
    try:
        pixels = gray.flatten().astype(np.float32)
        feats["stat_skew"] = float(skew(pixels))
        feats["stat_kurtosis"] = float(kurtosis(pixels))
        feats["stat_mean"] = float(np.mean(pixels))
        feats["stat_std"]  = float(np.std(pixels))
    except: pass
    return feats

def extract_fourier_features(gray):
    feats = {}
    try:
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1e-7)
        feats["fft_mean"] = float(np.mean(mag))
        feats["fft_std"] = float(np.std(mag))
        feats["fft_max"] = float(np.max(mag))
    except: pass
    return feats

def get_features_for_image(img_gray, img_gray_q, config):
    f = {}
    if config.get("use_glcm", False):     f.update(extract_glcm_features(img_gray_q))
    if config.get("use_lbp", False):      f.update(extract_lbp_features(img_gray))
    if config.get("use_lcp", False):      f.update(extract_lcp_features(img_gray))
    if config.get("use_shape", False):    f.update(extract_shape_features(img_gray))
    if config.get("use_hog", False):      f.update(extract_hog_compact(img_gray))
    if config.get("use_phog", False):     f.update(extract_phog_features(img_gray))
    if config.get("use_softhist", False): f.update(extract_soft_histogram(img_gray))
    if config.get("use_stat", False):     f.update(extract_statistical_features(img_gray))
    if config.get("use_fft", False):      f.update(extract_fourier_features(img_gray))
    return f

# ==========================================
# DOMAIN TRANSFORM FONKSİYONLARI
# ==========================================
def domain_fourier(gray):
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def domain_wavelet(gray):
    try:
        coeffs = pywt.dwt2(gray, 'bior1.3')
        LL, _ = coeffs
        return cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    except: return gray

# ==========================================
# ANA ÇALIŞMA (MAIN)
# ==========================================
def main():
    print(f"Veri Yolu: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        print("HATA: Dataset bulunamadı!")
        return

    images = []
    print("Resimler taranıyor...")
    klasorler = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    for cls in klasorler:
        d_path = os.path.join(DATASET_PATH, cls)
        files = [f for f in os.listdir(d_path) if f.lower().endswith(('png','jpg','jpeg','bmp'))]
        print(f" - {cls}: {len(files)} resim")
        for fn in files:
            img = cv2.imread(os.path.join(d_path, fn))
            if img is not None:
                g, gq = preprocess_to_gray(img)
                images.append({"gray": g, "gray_q": gq, "class": cls})

    print(f"Toplam {len(images)} görüntü işlenmek üzere hazır.\n")

    # --- 1. MEVCUT GLCM ve VARYASYONLARI ---
    
    data = []
    for item in images:
        ft = get_features_for_image(item["gray"], item["gray_q"], {"use_glcm": True})
        ft["class"] = item["class"]
        data.append(ft)
    write_arff(data, "glcm.arff", "GLCM_ONLY")

    data = []
    for item in images:
        ft = get_features_for_image(item["gray"], item["gray_q"], {"use_glcm": True, "use_lbp": True})
        ft["class"] = item["class"]
        data.append(ft)
    write_arff(data, "glcm_lbp.arff", "GLCM_LBP")

    data = []
    for item in images:
        ft = get_features_for_image(item["gray"], item["gray_q"], {"use_glcm": True, "use_lbp": True, "use_hog": True})
        ft["class"] = item["class"]
        data.append(ft)
    write_arff(data, "glcm_lbp_hog.arff", "GLCM_LBP_HOG")
    
    data = []
    for item in images:
        ft = get_features_for_image(item["gray"], item["gray_q"], {"use_glcm": True, "use_lcp": True})
        ft["class"] = item["class"]
        data.append(ft)
    write_arff(data, "glcm_lcp.arff", "GLCM_LCP")
    
    data = []
    for item in images:
        ft = get_features_for_image(item["gray"], item["gray_q"], {"use_glcm": True, "use_hog": True})
        ft["class"] = item["class"]
        data.append(ft)
    write_arff(data, "glcm_hog.arff", "GLCM_HOG")
    
    data = []
    for item in images:
        ft = get_features_for_image(item["gray"], item["gray_q"], {"use_glcm": True, "use_phog": True})
        ft["class"] = item["class"]
        data.append(ft)
    write_arff(data, "glcm_phog.arff", "GLCM_PHOG")
    
    data = []
    for item in images:
        ft = get_features_for_image(item["gray"], item["gray_q"], {"use_glcm": True, "use_shape": True})
        ft["class"] = item["class"]
        data.append(ft)
    write_arff(data, "glcm_shape.arff", "GLCM_SHAPE")
    
    data = []
    for item in images:
        ft = get_features_for_image(item["gray"], item["gray_q"], {"use_glcm": True, "use_softhist": True})
        ft["class"] = item["class"]
        data.append(ft)
    write_arff(data, "glcm_softhist.arff", "GLCM_SOFTHIST")

    data = []
    for item in images:
        ft = get_features_for_image(item["gray"], item["gray_q"], {"use_glcm": True, "use_stat": True})
        ft["class"] = item["class"]
        data.append(ft)
    write_arff(data, "glcm_stat.arff", "GLCM_STAT")

    data = []
    for item in images:
        ft = get_features_for_image(item["gray"], item["gray_q"], {"use_glcm": True, "use_fft": True})
        ft["class"] = item["class"]
        data.append(ft)
    write_arff(data, "glcm_fft.arff", "GLCM_FFT")

    # --- 2. TEKİL ÖZELLİKLER ---

    write_arff([ {**get_features_for_image(i["gray"], i["gray_q"], {"use_lbp": True}), "class": i["class"]} for i in images], "LBP_only.arff", "LBP")
    write_arff([ {**get_features_for_image(i["gray"], i["gray_q"], {"use_lcp": True}), "class": i["class"]} for i in images], "lcp_only.arff", "LCP")
    write_arff([ {**get_features_for_image(i["gray"], i["gray_q"], {"use_shape": True}), "class": i["class"]} for i in images], "Shape_only.arff", "SHAPE")
    write_arff([ {**get_features_for_image(i["gray"], i["gray_q"], {"use_hog": True}), "class": i["class"]} for i in images], "HOG_only.arff", "HOG")
    write_arff([ {**get_features_for_image(i["gray"], i["gray_q"], {"use_phog": True}), "class": i["class"]} for i in images], "PHOG_only.arff", "PHOG")
    write_arff([ {**get_features_for_image(i["gray"], i["gray_q"], {"use_softhist": True}), "class": i["class"]} for i in images], "SoftHist_only.arff", "SOFTHIST")
    write_arff([ {**get_features_for_image(i["gray"], i["gray_q"], {"use_stat": True}), "class": i["class"]} for i in images], "stat_only.arff", "STAT")
    write_arff([ {**get_features_for_image(i["gray"], i["gray_q"], {"use_fft": True}), "class": i["class"]} for i in images], "fft_only.arff", "FFT")

    # --- 3. BİRLEŞİK SENARYOLAR (FUSION) ---

    data = []
    for item in images:
        ft = get_features_for_image(item["gray"], item["gray_q"], 
            {"use_lbp": True, "use_lcp": True, "use_shape": True, "use_hog": True, "use_phog": True, "use_softhist": True})
        ft["class"] = item["class"]
        data.append(ft)
    write_arff(data, "Fusion_lbp+lcp+shape+hog+phog+softhist_List.arff", "FUSION_LIST")

    data = []
    for item in images:
        ft = get_features_for_image(item["gray"], item["gray_q"], 
            {"use_glcm":True, "use_lbp":True, "use_lcp":True, "use_shape":True, "use_hog":True, "use_phog":True, "use_softhist":True, "use_stat":True, "use_fft":True})
        ft["class"] = item["class"]
        data.append(ft)
    write_arff(data, "fusion_ALL.arff", "FUSION_WITH_STATS")
    
    data = []
    for item in images:
        ft = get_features_for_image(item["gray"], item["gray_q"], {"use_stat": True, "use_fft": True})
        ft["class"] = item["class"]
        data.append(ft)
    write_arff(data, "stat_fft_fusion.arff", "STAT_FFT_FUSION")

    # --- 4. DOMAIN TRANSFORMLARI ---
    
    domains = [
        ("wavelet_glcm.arff", domain_wavelet),
        ("fourier_glcm.arff", domain_fourier)
    ]

    for fname, d_func in domains:
        print(f"Domain İşleniyor: {fname}")
        d_data = []
        for item in images:
            d_img = d_func(item["gray"])
            if d_img is None: continue
            d_img_q = (d_img / (256/32)).astype(np.uint8)
            ft = extract_glcm_features(d_img_q)
            ft["class"] = item["class"]
            d_data.append(ft)
        write_arff(d_data, fname, "DOMAIN_TRANSFORM")

    # --- 5. TEKİL ÖZELLİKLER (NORMAL VE WAVELET) ---
    print("\n--- Tekil GLCM ve Wavelet+GLCM özellikleri çıkarılıyor ---")
    glcm_props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

    for prop in glcm_props:
        single_data = []
        for item in images:
            full_glcm = extract_glcm_features(item["gray_q"])
            prop_key = f"glcm_{prop}"
            val = full_glcm.get(prop_key, 0.0)
            single_data.append({prop_key: val, "class": item["class"]})
        write_arff(single_data, f"only_glcm_{prop}.arff", f"SINGLE_GLCM_{prop.upper()}")

    for prop in glcm_props:
        wavelet_single_data = []
        for item in images:
            d_img = domain_wavelet(item["gray"])
            d_img_q = (d_img / (256/32)).astype(np.uint8)
            full_glcm_wave = extract_glcm_features(d_img_q)
            prop_key = f"glcm_{prop}"
            val = full_glcm_wave.get(prop_key, 0.0)
            wavelet_single_data.append({f"wavelet_{prop}": val, "class": item["class"]})
        write_arff(wavelet_single_data, f"wavelet_glcm_{prop}.arff", f"WAVELET_SINGLE_GLCM_{prop.upper()}")

    # =========================================================================
    #  WAVELET ALT BANTLARI (LL, LH, HL, HH) AYRI KAYIT
    # =========================================================================
    print("\n--- Wavelet Alt Bant Özellikleri Çıkarılıyor (LL, LH, HL, HH) ---")
    
    # Her bant için listeler hazırlayalım
    wavelet_results = {"LL": [], "LH": [], "HL": [], "HH": []}
    all_bands_fusion = [] # Tüm bantların birleştiği büyük özellik seti için

    for item in images:
        try:
            # 2D Discrete Wavelet Transform (DWT)
            # cA = LL (Approx), (cH, cV, cD) = (LH, HL, HH detay bantları)
            coeffs2 = pywt.dwt2(item["gray"].astype(np.float32), 'bior1.3')
            LL, (LH, HL, HH) = coeffs2
            
            bands = {"LL": LL, "LH": LH, "HL": HL, "HH": HH}
            fusion_row = {} # Birleştirilmiş satır için
            
            for band_name, band_data in bands.items():
                # Bandı normalize et ve GLCM için 32 seviyeye kuantize et
                band_norm = cv2.normalize(band_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                band_q = (band_norm / (256/32)).astype(np.uint8)
                
                # GLCM özelliklerini çıkar
                band_feats = extract_glcm_features(band_q)
                
                # Özellik isimlerini banta göre güncelle 
                renamed_feats = {f"{band_name}_{k}": v for k, v in band_feats.items()}
                
                # Ayrı kayıt listesine ekle
                row_with_class = renamed_feats.copy()
                row_with_class["class"] = item["class"]
                wavelet_results[band_name].append(row_with_class)
                
                # Fusion (birleştirme) satırına ekle
                fusion_row.update(renamed_feats)
            
            fusion_row["class"] = item["class"]
            all_bands_fusion.append(fusion_row)
            
        except Exception as e:
            print(f"Wavelet bant hatası: {e}")

    # Her bandı ayrı ARFF dosyası olarak kaydet
    for band_name, data_list in wavelet_results.items():
        write_arff(data_list, f"wavelet_band_{band_name}.arff", f"WAVELET_BAND_{band_name}")

    # Tüm bantların bir arada olduğu dosyayı kaydet
    write_arff(all_bands_fusion, "wavelet_all_bands_fusion.arff", "WAVELET_ALL_BANDS_COMBINED")
# =========================================================================
    #  HER BANT İÇİN TEKİL GLCM ÖZELLİKLERİ
    # =========================================================================
    print("\n--- Bant Bazlı Tekil GLCM Özellikleri Kaydediliyor ---")
    
    # Mevcut glcm_props listesini kullanıyoruz
    # wavelet_results sözlüğü zaten yukarıdaki döngüde dolmuş durumda
    
    for band_name in ["LL", "LH", "HL", "HH"]:
        for prop in glcm_props:
            single_prop_data = []
            # Yukarıda her bant için hazırlanan tüm GLCM özelliklerini içeren listeden ilgili olanı ayıklıyoruz
            for entry in wavelet_results[band_name]:
                # Özellik ismi yukarıdaki blokta "LL_glcm_contrast" formatında kaydedilmişti
                prop_key = f"{band_name}_glcm_{prop}"
                
                if prop_key in entry:
                    # Yeni bir sözlük oluştur: { "HL_energy": değer, "class": sınıf }
                   
                    clean_name = f"{band_name}_{prop}" 
                    single_prop_data.append({
                        clean_name: entry[prop_key],
                        "class": entry["class"]
                    })
            
            # Her birini ayrı dosya olarak kaydet
            if single_prop_data:
                filename = f"wavelet_{band_name}_{prop}.arff"
                write_arff(single_prop_data, filename, f"SINGLE_{band_name}_{prop.upper()}")
    print("\n✅ TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")

if __name__ == "__main__":
    main()
