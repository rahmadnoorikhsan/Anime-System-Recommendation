# Laporan Proyek Machine Learning - Rahmad Noor Ikhsan

## Project Overview

Dalam era digital, konsumsi konten hiburan mengalami lonjakan yang signifikan, terutama pada animasi Jepang atau anime. Dengan ratusan hingga ribuan judul yang tersedia, pengguna sering kali kesulitan memilih anime yang sesuai dengan preferensi mereka. Untuk itu, dibutuhkan sistem rekomendasi yang dapat membantu pengguna menemukan tontonan yang relevan, personal, dan menarik secara otomatis.

Sistem rekomendasi bukan hanya memberikan pengalaman pengguna yang lebih baik, tetapi juga menjadi strategi penting dalam industri hiburan untuk meningkatkan engagement dan retensi pengguna. Platform besar seperti Netflix dan Crunchyroll juga telah menerapkan sistem rekomendasi serupa untuk meningkatkan kepuasan pengguna.

Menurut riset yang dilakukan oleh Statista (2021), industri anime Jepang bernilai lebih dari 2 triliun yen, mencerminkan peningkatan signifikan dalam permintaan konten animasi. Ini memperkuat pentingnya keberadaan sistem rekomendasi dalam ekosistem digital, terutama untuk mempertahankan keterlibatan pengguna dan meningkatkan pengalaman eksplorasi konten.

Studi lain oleh Soni et al. (2021) mengembangkan engine rekomendasi anime yang disebut RikoNet, dan membuktikan bahwa penggunaan fitur berbasis genre dan skor penilaian dapat membantu sistem menyarankan anime secara efektif. Penelitian ini menjadi dasar penting dalam menggabungkan pendekatan konten dan perilaku pengguna untuk sistem rekomendasi hiburan.

Dari serangkaian penelitian tersebut, dapat disimpulkan bahwa pengembangan sistem rekomendasi anime bukan hanya relevan, tetapi juga memiliki potensi besar untuk meningkatkan engagement pengguna melalui pendekatan machine learning yang tepat.

## Business Understanding

### Problem Statements
- Bagaimana memberikan rekomendasi bagi pengguna yang memiliki kesamaan pola kesukaan dengan anime tertentu?
- Bagaimana merancang model yang mampu memahami pola tersembunyi dalam perilaku pengguna untuk memberikan rekomendasi yang lebih relevan?
### Goals
- Membangun model rekomendasi yang mampu mengenali pola kesukaan pengguna berdasarkan anime yang mirip dengan tontonan sebelumnya.
- Mengembangkan model machine learning yang dapat beradaptasi dan terus memperbaiki rekomendasinya seiring waktu dan perubahan selera pengguna.
### Solution statements
- Content-Based Filtering
  
    Sistem mempelajari karakteristik anime yang sudah disukai pengguna, seperti genre, tema, dan rating. Dengan memahami ciri-ciri tersebut, sistem merekomendasikan anime lain yang memiliki kemiripan, sehingga rekomendasi terasa sangat personal dan sesuai dengan selera individu pengguna.
- Collaborative Filtering
  
  Pendekatan ini memanfaatkan pola interaksi dan preferensi dari banyak pengguna untuk menemukan kemiripan perilaku. Melalui model deep learning, sistem dapat menggali pola tersembunyi dalam data yang kompleks dan memberikan rekomendasi yang tidak hanya berdasarkan konten, tetapi juga berdasarkan kesamaan preferensi antar pengguna. Dengan demikian, rekomendasi menjadi lebih variatif dan akurat, bahkan untuk anime yang belum dikenal oleh pengguna.

## Data Understanding

### Informasi Datasets

| Jenis | Keterangan |
| ------ | ------ |
| Title | Anime Recommendations Database |
| Source | [Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) |
| Maintainer | [CooperUnion](https://www.kaggle.com/organizations/CooperUnion)  |
| License | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) |
| Visibility | Public |
| Tags | Movies and TV Shows, Anime and Manga, Comics and Animation, Popular Culture |
| Usability | 8.24 |

Dataset ini terdiri dari dua file utama, yaitu `animes.csv` dan `ratings.csv`, yang secara bersama-sama digunakan untuk membangun sistem rekomendasi berbasis **Content-Based Filtering** dan **Collaborative Filtering**.

### Data Loading
Pada tahap ini, dilakukan proses pemuatan data dari dua file utama yang digunakan dalam analisis, yaitu dataset `animes` dan `ratings`. Kedua dataset ini memberikan gambaran lengkap mengenai konten anime serta interaksi pengguna dengan anime tersebut.

Dataset yang dimuat memiliki karakteristik sebagai berikut:
- Dataset `animes` berisi **12.294 judul anime unik** yang mencerminkan keberagaman konten anime yang ada di dalam dataset.
- Dataset `ratings` memuat data dari **73.515 pengguna unik** yang telah memberikan penilaian terhadap anime-anime tersebut.

### Exploratory Data Analysis (EDA)
Exploratory Data Analysis atau sering disingkat EDA merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. Teknik ini biasanya menggunakan bantuan statistik dan representasi grafis atau visualisasi.

#### Analisis Dataset `animes`
Dataset `animes` merupakan kumpulan data yang menyajikan informasi mendetail mengenai berbagai judul anime yang tersedia dalam sistem.
| anime\_id | name                                                      | genre                                                        | type  | episodes | rating | members |
| --------- | --------------------------------------------------------- | ------------------------------------------------------------ | ----- | -------- | ------ | ------- |
| 32281     | Kimi no Na wa.                                            | Drama, Romance, School, Supernatural                         | Movie | 1        | 9.37   | 200630  |
| 5114      | Fullmetal Alchemist: Brotherhood                          | Action, Adventure, Drama, Fantasy, Magic, Military, Shounen  | TV    | 64       | 9.26   | 793665  |
| 32935     | Haikyuu!!: Karasuno Koukou VS Shiratorizawa Gakuen Koukou | Comedy, Drama, School, Shounen, Sports                       | TV    | 10       | 9.15   | 93351   |
| 11061     | Hunter x Hunter (2011)                                    | Action, Adventure, Shounen, Super Power                      | TV    | 148      | 9.13   | 425855  |
| 820       | Ginga Eiyuu Densetsu                                      | Drama, Military, Sci-Fi, Space                               | OVA   | 110      | 9.11   | 80679   |

Dataset `animes` terdiri dari **12.294 baris data** dan memiliki **7 kolom**. Setiap baris merepresentasikan satu judul anime, lengkap dengan berbagai atribut informatif. Berikut ini adalah penjelasan masing-masing kolom yang terdapat dalam dataset:

- **anime_id** : Merupakan ID unik yang mengidentifikasi setiap judul anime secara individual.

- **name** : Nama atau judul dari anime.

- **genre** : Kategori atau genre anime, seperti Action, Comedy, Romance, dan sebagainya. Beberapa judul dapat memiliki lebih dari satu genre.

- **type** : Format dari anime tersebut, seperti TV series, Movie, OVA, dll.

- **episodes** : Jumlah episode dari anime. Nilai ini tidak selalu berupa angka karena beberapa judul mungkin belum selesai atau tidak memiliki informasi jumlah episode yang lengkap.

- **rating** : Skor rata-rata yang diberikan oleh pengguna terhadap anime tersebut.

- **members** : Jumlah pengguna yang telah menambahkan anime ini ke daftar mereka (watchlist, watching, completed, dll).

#### Visualisasi Distribusi Genre
Untuk memahami lebih dalam seberapa sering masing-masing genre muncul dalam dataset, dilakukan visualisasi distribusi genre. Visualisasi ini memberikan gambaran yang lebih intuitif mengenai genre-genre yang mendominasi serta membantu dalam mengidentifikasi pola umum dalam data.

![Distribusi Genre](https://raw.githubusercontent.com/rahmadnoorikhsan/Anime-System-Recommendation/main/resource/genres_distribution.png)

Distribusi genre dalam dataset `animes` menunjukkan bahwa **Comedy** merupakan genre yang paling sering muncul, dengan total **4.645** anime. Hal ini menunjukkan bahwa komedi menjadi elemen yang dominan dan populer dalam dunia anime. Genre-genre populer lainnya seperti **Action (2845)**, **Adventure (2348)**, dan **Fantasy (2309)** juga memiliki frekuensi yang tinggi dan menempati posisi teratas setelah Comedy.

#### Analisis Dataset `ratings`
Dataset `ratings` berisi informasi mengenai interaksi pengguna terhadap judul anime tertentu dalam bentuk penilaian (rating). Data ini penting dalam membangun sistem rekomendasi berbasis collaborative filtering, karena merepresentasikan preferensi individual setiap pengguna terhadap anime yang telah mereka tonton.

| user\_id | anime\_id | rating |
| -------- | --------- | ------ |
| 1        | 20        | 9.0    |
| 5        | 24        | 10.0   |
| 8        | 79        | 8.0    |
| 8        | 226       | 5.0    |
| 9        | 241       | 5.0    |

Dataset ini terdiri dari lebih dari **7 juta entri**, dengan total **73.515 pengguna unik**. Jumlah data yang besar ini menjadi potensi besar untuk mengeksplorasi hubungan antar pengguna dan preferensi mereka terhadap berbagai genre atau judul anime tertentu. Berikut ini adalah penjelasan masing-masing kolom yang terdapat dalam dataset:

- **user_id**: Merupakan ID unik dari masing-masing pengguna yang melakukan penilaian terhadap anime.

- **anime_id**: Merupakan ID dari judul anime yang diberi penilaian. ID ini merujuk pada data yang ada di dataset animes.

- **rating**: Nilai yang diberikan pengguna terhadap anime. Skor berkisar antara 1 hingga 10, dengan nilai -1 menandakan bahwa pengguna telah menonton anime tersebut namun tidak memberikan penilaian (missing rating).

#### Visualisasi Distribusi Rating
Distribusi  rating menggambarkan seberapa aktif setiap pengguna dalam memberikan penilaian terhadap anime. Analisis ini penting untuk memahami pola interaksi pengguna dengan dataset dan memastikan model rekomendasi dapat bekerja efektif dengan memperhatikan perilaku pengguna.

![Distribusi Rating](https://raw.githubusercontent.com/rahmadnoorikhsan/Anime-System-Recommendation/main/resource/rating_distribution.png)

Distribusi user rating menunjukkan bahwa **rating -1 (18,9%)** mendominasi sebagai nilai tertinggi kedua, menandakan banyak pengguna menonton tanpa memberikan rating. **Rating 8 (21,1%)** adalah nilai yang paling sering diberikan, diikuti rating 7, 9, dan 10, yang menunjukkan kecenderungan memberi nilai tinggi. Rating rendah (1–4) sangat jarang, hanya **kurang dari 2,5%,** memperlihatkan bias positif pengguna.

## Data Preparation
Data preparation merupakan proses penting yang dilakukan sebelum masuk ke tahap analisis atau pemodelan. Tujuannya adalah memastikan bahwa data yang digunakan dalam kondisi bersih, konsisten, dan siap digunakan. Berikut ini adalah tahapan-tahapan data preparation yang diterapkan dalam proyek ini:

### Pengurutan Dataset
Langkah pertama dalam persiapan data adalah mengurutkan dataset animes berdasarkan kolom anime_id dan dataset ratings berdasarkan kolom user_id. Pengurutan ini tidak memengaruhi isi data, namun berguna untuk menjaga keteraturan dan mempermudah proses integrasi antar dataset.
```python
animes_df = animes.sort_values('anime_id')
ratings_df = ratings.sort_values('user_id')
```

### Penyesuaian Nama Kolom
Jika diperhatikan pada tahapan EDA, dataset `animes` dan `ratings` sama-sama memiliki kolom bernama rating. Untuk menghindari konflik dan kebingungan saat proses penggabungan, kolom rating pada dataset animes diubah menjadi official_rating.
```python
animes_df.rename(columns={"rating": "official_rating"}, inplace=True)
```

### Penggabungan Dataset
Dataset `animes_df` dan `ratings_df` kemudian digabungkan menggunakan teknik inner join berdasarkan kolom `anime_id`. Hasil penggabungan ini menghasilkan dataset baru yang berisi informasi lengkap mengenai anime dan rating yang diberikan oleh masing-masing pengguna.
```python
full_anime_data = pd.merge(ratings_df, animes_df, on="anime_id", how='inner')
```

Setelah dilakukan penggabungan dataset, terdapat **7.813.727 data** dan **9 kolom**. Dengan tampilan seperti berikut
| user\_id | anime\_id | rating | name                              | genre                                                   | type  | episodes | official\_rating | members |
| -------- | --------- | ------ | --------------------------------- | ------------------------------------------------------- | ----- | -------- | ---------------- | ------- |
| 1        | 936       | -1     | Naruto Movie 2: Dai Gekitotsu!... | Adventure, Comedy, Drama, Fantasy, Shounen, Super Power | Movie | 1        | 6.99             | 97,308  |
| 1        | 20        | -1     | Naruto                            | Action, Comedy, Martial Arts, Shounen, Super Power      | TV    | 220      | 7.81             | 683,297 |
| 1        | 4744      | -1     | Akaneiro ni Somaru Saka           | Comedy, Harem, Romance, School                          | TV    | 12       | 6.69             | 91,453  |
| 1        | 4581      | -1     | Shikabane Hime: Aka               | Action, Horror, Martial Arts                            | TV    | 13       | 7.38             | 71,502  |
| 1        | 4224      | -1     | Toradora!                         | Comedy, Romance, School, Slice of Life                  | TV    | 25       | 8.45             | 633,817 |

### Menangani Anomali Nilai Rating
Dalam dataset `ratings`, nilai -1 pada kolom `rating` menunjukkan bahwa pengguna telah menonton anime namun tidak memberikan rating eksplisit. Nilai ini dianggap sebagai anomali dan diubah menjadi NaN (missing value) agar tidak disalahartikan sebagai rating valid.
```python
full_anime_clean_data = full_anime_data.copy()

full_anime_clean_data.replace({"rating": -1}, np.nan, inplace=True)
```

### Menangani Missing Value
Setelah mengganti nilai -1 menjadi NaN, dilakukan pengecekan terhadap seluruh kolom untuk mengetahui keberadaan missing values. Kolom-kolom yang mengandung nilai kosong seperti rating, genre, type, dan official_rating kemudian dibersihkan dengan metode dropna().
| Nama Kolom        | Jumlah Missing Value |
| ----------------- | -------------------- | 
| user_id           | 0                    | 
| anime_id          | 0                    | 
| rating            | 1.476.488            | 
| name              | 0                    | 
| genre             | 110                  | 
| type              | 4                    | 
| episodes          | 0                    | 
| official_rating   | 6                    | 
| members           | 0                    | 

```python
full_anime_clean_data = full_anime_clean_data.dropna()
```

Setelah melakukan drop pada missing values, saat ini dataset memiliki **6.337.146 data**

### Menangani Duplikasi Data
Tahapan terakhir adalah memastikan bahwa tidak terdapat data yang duplikat. Duplikasi dapat memengaruhi hasil analisis atau model karena memberi bobot lebih pada data yang berulang. Setelah dicek, ditemukan **satu baris data** yang duplikat dan langsung dihapus.

```python
full_anime_clean_data = full_anime_clean_data.drop_duplicates()
```

Setelah menghapus data duplikat, dataset akhir yang bersih terdiri dari **6.337.145 baris data** dan telah siap digunakan untuk proses analisis lebih lanjut, termasuk dalam pengembangan sistem rekomendasi. Perlu dicatat bahwa setiap metode pemodelan memiliki karakteristik dan kebutuhan data yang berbeda, sehingga tahapan data preparation dapat mengalami penyesuaian sesuai dengan pendekatan yang digunakan.

## Data Preparation untuk Content-Based Filtering
Pada bagian ini, proses persiapan data difokuskan untuk memenuhi kebutuhan metode Content-Based Filtering yang akan digunakan untuk menghitung kemiripan antar item, khususnya pada kolom `genre` guna membangun profil item yang akan digunakan dalam sistem rekomendasi.

### Penghapusan Duplikat Berdasarkan Nama Anime
Untuk memastikan setiap anime unik, data duplikat berdasarkan kolom name dihapus.
```python
df_content_based = full_anime_clean_data.copy()
df_content_based.drop_duplicates(subset="name", keep="first", inplace=True)
df_content_based.reset_index(drop=True, inplace=True)
```
### Pemformatan Kolom Genre
Kolom `genre` yang berisi beberapa genre dalam satu string dipisah menjadi list agar dapat diproses oleh model.
```python
genres = df_content_based["genre"].str.split(", | , | ,").astype(str)
```

### Ekstraksi Fitur dengan TF-IDF
TF-IDF (Term Frequency - Inverse Document Frequency) digunakan untuk mengubah data genre menjadi representasi numerik. Hal ini memudahkan model dalam menghitung kemiripan antar anime berdasarkan fitur genre.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_content_based = TfidfVectorizer(
    min_df=3,
    max_features=None,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    ngram_range=(1, 3),
    stop_words="english"
)

tf_matrix_content_based = tf_idf_content_based.fit_transform(genres)
```
Matriks TF-IDF ini memiliki **9.892 baris (jumlah anime unik)** dan **1.480 kolom (fitur unik hasil tokenisasi)**.

### Konversi Matriks TF-IDF ke Bentuk Dense
Matriks hasil TF-IDF diubah menjadi bentuk dense matrix agar mudah dianalisis atau diolah dalam tahap berikutnya.
```python
tf_matrix_dense = tf_matrix_content_based.todense()
```
## Data Preparation untuk Collaborative Filtering
## Modeling

## Evaluation
