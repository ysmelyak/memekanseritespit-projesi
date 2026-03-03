Meme Kanseri Tespiti İçin Medikal Görüntü Analizi ve Öznitelik Çıkarımı
Bu proje, meme kanseri veri setindeki görüntüleri işleyerek makine öğrenmesi modelleri için anlamlı öznitelikler (features) çıkaran kapsamlı bir görüntü işleme hattıdır. Kod, görüntüleri sadece piksel olarak değil; dokusal, istatistiksel ve frekans bazlı olarak analiz eder.

Projenin Yaptığı İşlemler
Görüntü Ön İşleme: Görüntülerin gri seviyeye dönüştürülmesi, 128x128 boyutuna yeniden boyutlandırılması ve gürültü giderme için Median Blur filtresi uygulanması.
Doku Analizi (Texture Analysis): * GLCM: Kontrast, enerji ve homojenlik gibi dokusal özelliklerin hesaplanması.
LBP (Local Binary Pattern): Yerel doku desenlerinin tespiti.
Kenar ve Şekil Analizi: * HOG (Histogram of Oriented Gradients): Nesne kenarlarının ve yönelimlerinin tespiti.
Şekil Özellikleri: Alan, çevre ve dairesellik (circularity) hesaplamaları.
Frekans Alanı Analizi: * Fourier Transform (FFT): Görüntünün frekans bazlı özelliklerinin çıkarılması.
Wavelet Transform (DWT): Görüntünün LL, LH, HL ve HH alt bantlarına ayrılarak analiz edilmesi.

Veri Çıktısı: Tüm bu özelliklerin makine öğrenmesi yazılımlarında (Weka vb.) kullanılmak üzere .arff formatında kaydedilmesi.

Kullanılan Teknolojiler
Python 3
OpenCV: Görüntü manipülasyonu.
Scikit-Image: Gelişmiş öznitelik çıkarımı (GLCM, LBP, HOG).
PyWavelets: Dalgacık (Wavelet) dönüşümleri.
Pandas & SciPy: Veri yönetimi ve istatistiksel hesaplamalar.

Dosya Yapısı
yeşim_elyak_..._final.py: Ana işlem kodu.
/veri_seti: Analiz edilen ham görüntüler.
/final_outputs: Kodun ürettiği .arff formatındaki veri dosyaları.
