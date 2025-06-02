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

##### Struktur Dataset :
- **Jumlah entri**: 12.294 baris
- **Jumlah Kolom**: 7 kolom
- **Jumlah anime unik**: 12.294 anime

##### Struktur Kolom :
- **anime_id** : Merupakan ID unik yang mengidentifikasi setiap judul anime secara individual.

- **name** : Nama atau judul dari anime.

- **genre** : Kategori atau genre anime, seperti Action, Comedy, Romance, dan sebagainya. Beberapa judul dapat memiliki lebih dari satu genre.

- **type** : Format dari anime tersebut, seperti TV series, Movie, OVA, dll.

- **episodes** : Jumlah episode dari anime. Nilai ini tidak selalu berupa angka karena beberapa judul mungkin belum selesai atau tidak memiliki informasi jumlah episode yang lengkap.

- **rating** : Skor rata-rata yang diberikan oleh pengguna terhadap anime tersebut.

- **members** : Jumlah pengguna yang telah menambahkan anime ini ke daftar mereka (watchlist, watching, completed, dll).


##### Contoh Data :
| anime\_id | name                                                      | genre                                                        | type  | episodes | rating | members |
| --------- | --------------------------------------------------------- | ------------------------------------------------------------ | ----- | -------- | ------ | ------- |
| 32281     | Kimi no Na wa.                                            | Drama, Romance, School, Supernatural                         | Movie | 1        | 9.37   | 200630  |
| 5114      | Fullmetal Alchemist: Brotherhood                          | Action, Adventure, Drama, Fantasy, Magic, Military, Shounen  | TV    | 64       | 9.26   | 793665  |
| 32935     | Haikyuu!!: Karasuno Koukou VS Shiratorizawa Gakuen Koukou | Comedy, Drama, School, Shounen, Sports                       | TV    | 10       | 9.15   | 93351   |
| 11061     | Hunter x Hunter (2011)                                    | Action, Adventure, Shounen, Super Power                      | TV    | 148      | 9.13   | 425855  |
| 820       | Ginga Eiyuu Densetsu                                      | Drama, Military, Sci-Fi, Space                               | OVA   | 110      | 9.11   | 80679   |


##### Temuan :
- **Missing Values**: Ditemukan beberapa nilai yang hilang (missing values) pada kolom-kolom berikut:

  - `genre`: **62 missing values**

  - `type`: **25 missing values**

  - `rating`: **230 missing values**

- **Duplikat** : Tidak ditemukan data duplikat dalam dataset ini. Setiap anime memiliki ID unik dan tidak ada entri yang identik secara keseluruhan.

#### Visualisasi Distribusi Genre
Untuk memahami lebih dalam seberapa sering masing-masing genre muncul dalam dataset, dilakukan visualisasi distribusi genre. Visualisasi ini memberikan gambaran yang lebih intuitif mengenai genre-genre yang mendominasi serta membantu dalam mengidentifikasi pola umum dalam data.

![Distribusi Genre](https://raw.githubusercontent.com/rahmadnoorikhsan/Anime-System-Recommendation/main/resource/genres_distribution.png)

Distribusi genre dalam dataset `animes` menunjukkan bahwa **Comedy** merupakan genre yang paling sering muncul, dengan total **4.645** anime. Hal ini menunjukkan bahwa komedi menjadi elemen yang dominan dan populer dalam dunia anime. Genre-genre populer lainnya seperti **Action (2845)**, **Adventure (2348)**, dan **Fantasy (2309)** juga memiliki frekuensi yang tinggi dan menempati posisi teratas setelah Comedy.

#### Analisis Dataset `ratings`
Dataset `ratings` berisi informasi mengenai interaksi pengguna terhadap judul anime tertentu dalam bentuk penilaian (rating). Data ini penting dalam membangun sistem rekomendasi berbasis collaborative filtering, karena merepresentasikan preferensi individual setiap pengguna terhadap anime yang telah mereka tonton.

##### Struktur Dataset :
- **Jumlah entri**: 7.813.737 baris
- **Jumlah kolom**: 3 kolom
- **Jumlah pengguna unik**: 73.515 user

##### Struktur Kolom :
- **user_id**: Merupakan ID unik dari masing-masing pengguna yang melakukan penilaian terhadap anime.

- **anime_id**: Merupakan ID dari judul anime yang diberi penilaian. ID ini merujuk pada data yang ada di dataset animes.

- **rating**: Nilai yang diberikan pengguna terhadap anime. Skor berkisar antara 1 hingga 10, dengan nilai -1 menandakan bahwa pengguna telah menonton anime tersebut namun tidak memberikan penilaian (missing rating).

##### Contoh Data :

| user\_id | anime\_id | rating |
| -------- | --------- | ------ |
| 1        | 20        | 9.0    |
| 5        | 24        | 10.0   |
| 8        | 79        | 8.0    |
| 8        | 226       | 5.0    |
| 9        | 241       | 5.0    |


##### Temuan :
- **Missing Values**: Tidak ditemukan missing values secara eksplisit dalam dataset ini. Namun, perlu dicatat bahwa nilai -1 dalam kolom rating merepresentasikan ketidakhadiran rating dari pengguna, dan dapat diperlakukan sebagai missing value tergantung pada konteks analisis.

- **Duplikat**: Ditemukan 1 entri duplikat dalam dataset ini. Duplikat ini dapat mengganggu akurasi analisis dan model rekomendasi, sehingga sebaiknya dihapus atau dikonsolidasikan.

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

Dengan proses ini, data telah siap untuk digunakan dalam pelatihan model Content-Based Filtering yang akan dibahas pada tahap Modelling.

## Data Preparation untuk Collaborative Filtering
Sebelum membangun model Collaborative Filtering, dilakukan tahap persiapan data untuk memastikan bahwa data yang digunakan bersih, optimal, dan sesuai dengan kebutuhan algoritma. Proses ini mencakup penyaringan terhadap user dan anime yang paling aktif maupun populer, serta proses encoding untuk mengonversi ID menjadi format numerik.

### Penyaringan User Teraktif dan Anime Terpopuler
Dataset awal berukuran sangat besar dan mengandung banyak entri dengan sedikit interaksi. Hal ini dapat mempengaruhi efisiensi dan performa model. Oleh karena itu, dilakukan penyaringan dengan kriteria sebagai berikut:

- Memilih 100 pengguna dengan jumlah rating terbanyak.

- Memilih 500 anime yang memiliki rating terbanyak dari pengguna yang telah difilter.

Langkah ini mampu mengurangi kompleksitas data, mempercepat proses pelatihan model, dan tetap mempertahankan pola interaksi yang representatif.

```python
# Membuat salinan dataset untuk Collaborative Filtering
df_collab = full_anime_clean_data.copy()

# Memilih 100 user paling aktif berdasarkan jumlah rating
top_n_users = df_collab["user_id"].value_counts().index[:100]
df_collab = df_collab[df_collab["user_id"].isin(top_n_users)]

# Memilih 500 anime terpopuler berdasarkan jumlah rating
top_m_anime = df_collab["name"].value_counts().index[:500]
df_collab = df_collab[df_collab["name"].isin(top_m_anime)]

# Menampilkan jumlah data hasil filter
print("Jumlah data setelah filtering:", len(df_collab))
```
Hasil penyaringan ini menghasilkan **34.501 baris data**, yang mencerminkan interaksi antara **100 user teraktif dan 500 anime terpopuler.

### Encoding anime_id dan user_id
Langkah selanjutnya adalah melakukan encoding pada atribut `anime_id` dan `user_id` agar data dapat digunakan dalam algoritma Collaborative Filtering berbasis matrix factorization. Encoding ini mengubah setiap ID menjadi angka urut yang dimulai dari 0.

```python
# Mengambil daftar unik dari anime_id dan user_id
anime_ids = df_collab['anime_id'].unique().tolist()
user_ids = df_collab['user_id'].unique().tolist()

# Melakukan encoding anime_id dan user_id
anime_to_anime_encoded = {x: i for i, x in enumerate(anime_ids)}
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}

# Reverse dictionary
anime_encoded_to_anime = {i: x for x, i in anime_to_anime_encoded.items()}
user_encoded_to_user = {i: x for x, i in user_to_user_encoded.items()}

# Menambahkan kolom hasil encoding ke dalam dataframe
df_collab['anime'] = df_collab['anime_id'].map(anime_to_anime_encoded)
df_collab['user'] = df_collab['user_id'].map(user_to_user_encoded)
```

Proses ini menghasilkan dua kolom baru, yaitu user dan anime, yang merepresentasikan hasil transformasi numerik dari user_id dan anime_id. Data ini akan menjadi input utama pada proses pelatihan model Collaborative Filtering di tahap selanjutnya.

### Pembagian Data untuk Training dan Validasi
Setelah proses encoding selesai, tahap selanjutnya adalah membagi data menjadi bagian training dan validasi. Hal ini bertujuan untuk mempersiapkan data agar dapat digunakan pada proses pelatihan dan evaluasi model Collaborative Filtering, serta untuk menghindari overfitting terhadap data pelatihan.

#### Pengacakan Dataset
Sebelum data dibagi, dilakukan proses pengacakan guna memastikan distribusi data lebih merata dan tidak mengikuti urutan aslinya yang dapat menyebabkan bias dalam proses pelatihan.

```python
df_collab = df_collab.sample(frac=1, random_state=42)
```

#### Membentuk Input (x) dan Target (y)

Untuk membentuk data input dan target, diperlukan beberapa langkah penyesuaian:

- Variabel `x` dibentuk dari pasangan user dan anime, masing-masing sudah dalam bentuk numerik hasil encoding. Kombinasi ini akan menjadi input bagi model.

- Variabel `y` merupakan nilai rating yang telah dinormalisasi ke dalam skala 0 hingga 1, menggunakan rumus:

$$
\text{rating\_scaled} = \frac{\text{rating} - \text{min\_rating}}{\text{max\_rating} - \text{min\_rating}}
$$

Tujuannya adalah menyederhanakan rentang nilai target untuk mempercepat konvergensi saat pelatihan model.

```python
# Membuat variabel x untuk mencocokkan data user dan anime menjadi satu value
x = df_collab[['user', 'anime']].values

# Mengatur nilai minimum dan maksimum rating
min_rating = df_collab['rating'].min()
max_rating = df_collab['rating'].max()

# Membuat variabel y sebagai target rating dalam skala 0–1
y = df_collab['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
```

#### Pembagian Data Training dan Validasi
Setelah data input dan target terbentuk, proses selanjutnya adalah membagi data menjadi **80% untuk pelatihan (training set)** dan **20% untuk validasi (validation set)**. Validasi dilakukan untuk mengevaluasi performa model terhadap data yang tidak dilatih secara langsung.

```python
# Menentukan indeks pemisah antara data training dan validasi
train_indices = int(0.8 * df_collab.shape[0])

# Membagi data menjadi train dan validation
x_train, x_val = x[:train_indices], x[train_indices:]
y_train, y_val = y[:train_indices], y[train_indices:]
```

Hasil dari proses ini adalah dua pasang dataset:

- `x_train` dan `y_train` digunakan untuk pelatihan model.

- `x_val` dan `y_val` digunakan untuk validasi model.

Dengan proses ini, data telah siap untuk digunakan dalam pelatihan model Collaborative Filtering yang akan dibahas pada tahap Modeling.
## Modeling 
Permasalahan utama dalam penelitian ini adalah membantu pengguna menemukan anime yang relevan berdasarkan preferensi mereka. Untuk menyelesaikan permasalahan tersebut, sistem rekomendasi dikembangkan menggunakan dua pendekatan utama:

- Content-Based Filtering: Menganalisis kesamaan konten antar anime berdasarkan fitur genre.

- Collaborative Filtering: Memprediksi minat pengguna berdasarkan pola interaksi pengguna lain dengan anime.

Dengan dua pendekatan ini, sistem mampu merekomendasikan top-N anime kepada pengguna baik yang baru maupun yang sudah memiliki histori interaksi (cold-start dan warm-start problem).

### Content-Based Filtering dengan Cosine Similarity
Setelah membentuk matriks TF-IDF, sistem menghitung kemiripan antar anime menggunakan Cosine Similarity. Cosine Similarity merupakan metode yang umum digunakan dalam sistem rekomendasi berbasis konten karena mampu mengukur seberapa mirip dua vektor (dalam ruang fitur) berdasarkan arah atau sudut antar vektor, bukan berdasarkan besar nilainya. Nilai cosine similarity berkisar antara 0 (tidak mirip sama sekali) hingga 1 (sangat mirip atau identik).

Dalam implementasi ini, digunakan fungsi cosine_similarity dari library sklearn.
 
```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tf_matrix_content_based)
```

Berikut ini adalah sebagian dari hasil matriks cosine similarity dalam bentuk array:

```python
array([[1.        , 0.02228665, 0.04886245, ..., 0.        , 0.09304222,
        0.09304222],
       [0.02228665, 1.        , 0.02139413, ..., 0.04750064, 0.        ,
        0.        ],
       [0.04886245, 0.02139413, 1.        , ..., 0.        , 0.        ,
        0.        ],
       ...,
       [0.        , 0.04750064, 0.        , ..., 1.        , 0.        ,
        0.        ],
       [0.09304222, 0.        , 0.        , ..., 0.        , 1.        ,
        1.        ],
       [0.09304222, 0.        , 0.        , ..., 0.        , 1.        ,
        1.        ]])
```

Setelah berhasil membangun matriks kesamaan antar anime berdasarkan genre menggunakan metode Cosine Similarity, langkah selanjutnya adalah membuat fungsi untuk memberikan rekomendasi. Fungsi ini dikembangkan untuk menerima input berupa nama anime, kemudian mengembalikan top-N rekomendasi berdasarkan nilai kemiripan tertinggi dalam matriks cosine similarity.

```python
def anime_recommendations(nama_anime, similarity_data=cosine_sim_df, items=df_content_based[['name', 'genre']], k=10) -> pd.DataFrame:
    index = similarity_data.loc[:, nama_anime].to_numpy().argpartition(
        range(-1, -k, -1))

    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(nama_anime, errors='ignore')

    recommendations = pd.DataFrame(closest).merge(items).head(k)

    pd.set_option('display.max_colwidth', None)

    return recommendations
```
Untuk memastikan sistem rekomendasi bekerja dengan baik, dilakukan pengujian menggunakan anime populer, yaitu Naruto, sebagai referensi. Langkah pertama yang dilakukan untuk pengujian sistem rekomendasi adalah Menampilkan Data Genre Naruto.

```python
df_content_based[df_content_based.name.eq("Naruto")]
```
| user_id | anime_id | rating | name   | genre                                                           | type | episodes | official_rating | members |
|---------|----------|--------|--------|------------------------------------------------------------------|------|----------|------------------|---------|
| 25      | 3        | 20     | Naruto | Action, Comedy, Martial Arts, Shounen, Super Power              | TV   | 220      | 7.81             | 683297  |

Dari hasil tersebut, diketahui bahwa Naruto memiliki genre sebagai berikut:

**Action**, **Comedy**, **Martial Arts**, **Shounen**, dan **Super Power**.

Genre ini menjadi dasar bagi sistem untuk mencari anime dengan konten yang mirip.

Langkah selanjutnya adalah menjalankan fungsi rekomendasi dengan menjalankan code seperti berikut untuk menampilkan hasil rekomendasi yang relevan dengan anime naruto.
```python
anime_recommendations('Naruto')
```
| name                                                                                      | genre                                              |
|-------------------------------------------------------------------------------------------|----------------------------------------------------|
| Naruto x UT                                                                               | Action, Comedy, Martial Arts, Shounen, Super Power |
| Boruto: Naruto the Movie - Naruto ga Hokage ni Natta Hi                                   | Action, Comedy, Martial Arts, Shounen, Super Power |
| Naruto: Shippuuden Movie 3 - Hi no Ishi wo Tsugu Mono                                     | Action, Comedy, Martial Arts, Shounen, Super Power |
| Naruto Shippuuden: Sunny Side Battle                                                      | Action, Comedy, Martial Arts, Shounen, Super Power |
| Boruto: Naruto the Movie                                                                  | Action, Comedy, Martial Arts, Shounen, Super Power |
| Naruto Soyokazeden Movie: Naruto to Mashin to Mitsu no Onegai Dattebayo!!                 | Action, Comedy, Martial Arts, Shounen, Super Power |
| Naruto: Shippuuden Movie 4 - The Lost Tower                                               | Action, Comedy, Martial Arts, Shounen, Super Power |
| Battle Spirits: Ryuuko no Ken                                                             | Action, Comedy, Martial Arts, Shounen              |
| Kyutai Panic Adventure!                                                                   | Action, Martial Arts, Shounen, Super Power         |
| Ben-To                                                                                    | Action, Comedy, Martial Arts                       |

Dari hasil yang ditampilkan menunjukkan beberapa judul anime yang memiliki genre serupa dengan Naruto. Ini mengindikasikan bahwa sistem rekomendasi telah berhasil bekerja dengan baik. Judul-judul yang muncul dalam daftar rekomendasi umumnya juga bergenre Action, Comedy, Martial Arts, Shounen, atau Super Power yang merupakan ciri khas dari anime Naruto itu sendiri.

### Collaborative Filtering
Pada tahap modeling ini, sistem rekomendasi dibangun dengan pendekatan Collaborative Filtering menggunakan teknik embedding untuk merepresentasikan hubungan antara pengguna dan anime.

Model dikembangkan menggunakan API dari TensorFlow, dengan mendefinisikan kelas `RecommenderNet`. Model ini terdiri atas:

- **User Embedding**: Untuk merepresentasikan preferensi masing-masing pengguna.

- **Anime Embedding**: Untuk merepresentasikan karakteristik tiap anime.

- **User Bias dan Anime Bias**: Untuk memperhitungkan kecenderungan rating pengguna atau popularitas anime tertentu.

- **Dot Product**: Digunakan untuk menghitung skor kecocokan antara pengguna dan anime.

- **Sigmoid Activation**: Digunakan untuk menormalkan skor ke dalam skala [0, 1].

Berikut merupakan implementasi arsitektur modelnya.

```python
class RecommenderNet(Model):
    def __init__(self, num_users, num_anime, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=l2(1e-6))
        self.user_bias = Embedding(num_users, 1)
        self.anime_embedding = Embedding(num_anime, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=l2(1e-6))
        self.anime_bias = Embedding(num_anime, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        anime_vector = self.anime_embedding(inputs[:, 1])
        anime_bias = self.anime_bias(inputs[:, 1])

        dot_user_anime = tf.tensordot(user_vector, anime_vector, 2)
        x = dot_user_anime + user_bias + anime_bias
        return tf.nn.sigmoid(x)
```

Selanjutnya Model dikompilasi dengan parameter sebagai berikut:

- **Loss Function**: Binary Crossentropy

- **Optimizer**: Adam dengan learning rate 0.001

- **Metrics**: Root Mean Squared Error (RMSE)

```python
model = RecommenderNet(total_users, total_animes, 50)
model.compile(
    loss=BinaryCrossentropy(),
    optimizer=Adam(learning_rate=0.001),
    metrics=[RootMeanSquaredError()]
)
```

Untuk proses pelatihan, digunakan parameter:

- **batch_size** = 16

- **epochs** = 50

- **validation_data** = (x_val, y_val)

- **verbose** = 1

Callback EarlyStopping juga digunakan untuk menghentikan pelatihan jika tidak ada peningkatan performa model selama 10 epoch berturut-turut.

```python
callbacks = EarlyStopping(
    min_delta=0.0001,
    patience=10,
    restore_best_weights=True,
)

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=16,
    epochs=50,
    validation_data=(x_val, y_val),
    verbose=1,
    callbacks=[callbacks]
)
```
Setelah melakukan training pada model, selanjutnya adalah melakukan pengujian. Pengujian dilakukan untuk mengevaluasi kemampuan model dalam memberikan rekomendasi anime berdasarkan pola preferensi pengguna. Dalam pengujian ini, sistem akan memilih secara acak satu pengguna dari dataset, lalu memprediksi skor kecocokan untuk semua anime yang belum ditonton oleh pengguna tersebut.

Langkah-langkah yang dilakukan dalam melakukan pengujian adalah sebagai berikut : 
- Ambil satu ID pengguna secara acak.

- Identifikasi anime yang telah dan belum ditonton oleh pengguna.

- Prediksi skor rating terhadap semua anime yang belum ditonton.

- Mengambil 10 anime dengan skor tertinggi sebagai rekomendasi.

- Menampilkan hasil rekomendasi dan anime favorit sebelumnya dari pengguna.

Hasilnya, sistem menampilkan:

- Top-5 anime favorit pengguna berdasarkan rating tertinggi yang telah diberikan.

- Top-10 rekomendasi anime terbaik yang diprediksi cocok untuk pengguna.

```python
Rekomendasi Anime untuk Pengguna dengan ID: 47849

--------------------------------------------------------------------------------
Anime dengan rating tertinggi untuk pengguna:
--------------------------------------------------------------------------------
Hikaru no Go : Comedy, Game, Shounen, Supernatural
Toaru Kagaku no Railgun : Action, Sci-Fi, Super Power
Sono Hanabira ni Kuchizuke wo: Anata to Koibito Tsunagi : Hentai, School, Yuri
Himouto! Umaru-chan : Comedy, School, Seinen, Slice of Life
Haikyuu!! Second Season : Comedy, Drama, School, Shounen, Sports

--------------------------------------------------------------------------------
Rekomendasi 10 Anime Terbaik:
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
No         : 1
Nama Anime  : Neon Genesis Evangelion: The End of Evangelion
Genre      : Dementia, Drama, Mecha, Psychological, Sci-Fi
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
No         : 2
Nama Anime  : Kanon (2006)
Genre      : Drama, Romance, Slice of Life, Supernatural
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
No         : 3
Nama Anime  : Tengen Toppa Gurren Lagann
Genre      : Action, Adventure, Comedy, Mecha, Sci-Fi
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
No         : 4
Nama Anime  : Evangelion: 2.0 You Can (Not) Advance
Genre      : Action, Mecha, Sci-Fi
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
No         : 5
Nama Anime  : Clannad: After Story
Genre      : Drama, Fantasy, Romance, Slice of Life, Supernatural
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
No         : 6
Nama Anime  : Kara no Kyoukai 7: Satsujin Kousatsu (Kou)
Genre      : Action, Mystery, Romance, Supernatural, Thriller
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
No         : 7
Nama Anime  : Summer Wars
Genre      : Comedy, Sci-Fi
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
No         : 8
Nama Anime  : Suzumiya Haruhi no Shoushitsu
Genre      : Comedy, Mystery, Romance, School, Sci-Fi, Supernatural
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
No         : 9
Nama Anime  : Mawaru Penguindrum
Genre      : Comedy, Drama, Mystery, Psychological
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
No         : 10
Nama Anime  : Ookami Kodomo no Ame to Yuki
Genre      : Fantasy, Slice of Life
--------------------------------------------------------------------------------
```
## Evaluation

### Content-Based Filtering
Pada tahap ini, dilakukan proses evaluasi untuk mengukur performa sistem rekomendasi yang telah dikembangkan. Evaluasi difokuskan pada kemampuan sistem dalam memberikan rekomendasi anime yang relevan terhadap preferensi pengguna, khususnya pengguna yang menyukai anime Naruto. Hal ini dilakukan dengan mengukur sejauh mana model mampu mengidentifikasi dan mengurutkan anime dengan genre yang serupa dengan Naruto, yaitu Action, Comedy, Martial Arts, Shounen, dan Super Power.

Untuk mengevaluasi kualitas rekomendasi, digunakan metrik evaluasi berbasis Top-K relevance, yaitu:

#### 1. Precision@K
  
Precision@K mengukur proporsi item yang direkomendasikan di antara Top-K yang benar-benar relevan.

**Rumus:**

$$
\text{Precision@K} = \frac{|\text{Relevant Items in Top-K}|}{K}
$$

**Penjelasan:**

- *Relevant Items in Top-K*: Jumlah item relevan dalam daftar rekomendasi sebanyak *K* teratas.
- *K*: Jumlah rekomendasi yang dievaluasi.

#### 2. Recall@K
Recall@K mengukur sejauh mana sistem berhasil menemukan item-item relevan dari seluruh item yang relevan.
**Rumus:**

$$
\text{Recall@K} = \frac{|\text{Relevant Items in Top-K}|}{|\text{All Relevant Items}|}
$$

**Penjelasan:**

- *All Relevant Items*: Jumlah keseluruhan item relevan dalam ground truth.
- *Relevant Items in Top-K*: Item relevan yang muncul dalam Top-K hasil rekomendasi.
#### 3. Normalized Discounted Cumulative Gain (NDCG@K)
NDCG@K tidak hanya memperhatikan apakah item relevan masuk dalam rekomendasi, tetapi juga memperhatikan posisi kemunculannya dalam daftar. Semakin tinggi posisi item relevan, semakin besar kontribusinya terhadap skor.
**Rumus:**

$$
\text{NDCG@K} = \frac{DCG@K}{IDCG@K}
$$

**Penjelasan:**

- *DCG@K*: Skor DCG dari urutan rekomendasi aktual.
- *IDCG@K*: Skor DCG maksimum yang bisa dicapai jika semua item relevan berada di posisi teratas (urutan ideal).
- Nilai NDCG berada antara 0 dan 1, di mana 1 berarti rekomendasi sempurna.

Ketiga metrik ini dipilih karena sesuai dengan konteks sistem rekomendasi yang berfokus pada urutan rekomendasi terbaik berdasarkan relevansi, bukan pada nilai rating numerik semata. Metrik ini sangat relevan untuk menilai apakah sistem mampu menyarankan item yang relevan di antara Top-K pilihan teratas.

Evaluasi dilakukan menggunakan data ground truth yang disaring berdasarkan anime yang memiliki kelima genre utama yang identik dengan anime Naruto. Daftar ini diasumsikan sebagai item paling ideal untuk direkomendasikan kepada pengguna dengan preferensi serupa.

Hasil evaluasi Top-10 rekomendasi terhadap pengguna yang menyukai anime Naruto adalah sebagai berikut:
![Evaluation Model Content-Based Filtering](https://raw.githubusercontent.com/rahmadnoorikhsan/Anime-System-Recommendation/main/resource/evaluation-content-based.png)

- **Precision@10 sebesar 70%** menunjukkan bahwa sebagian besar rekomendasi yang dihasilkan sistem sudah tepat sasaran. Hal ini membuktikan bahwa sistem cukup andal dalam memilih item relevan.

- **Recall@10 sebesar 33%** mengindikasikan bahwa meskipun sistem belum mencakup semua item relevan dari seluruh daftar ground truth, namun performa dalam menemukan sebagian besar rekomendasi terbaik masih dapat diterima untuk konteks rekomendasi Top-K.

- **NDCG@10 sebesar 80%** menjadi bukti bahwa sistem tidak hanya memberikan anime yang relevan, namun juga mengurutkannya secara optimal. Ini penting karena pengguna cenderung melihat rekomendasi pada posisi atas terlebih dahulu.

#### Dampak terhadap Business Understanding:

- Model ini berhasil menjawab **Problem Statement Pertama**, karena mampu memberikan rekomendasi yang relevan berdasarkan kemiripan konten, dan sesuai dengan pola kesukaan pengguna terhadap anime tertentu seperti Naruto.

- **Goal Membangun model rekomendasi yang mampu mengenali pola kesukaan pengguna berdasarkan anime yang mirip dengan tontonan sebelumnya** Terpenuhi melalui Content-Based Filtering yang secara efektif memanfaatkan fitur konten seperti genre dan tema.

- **Solution Statement dengan model Content-based filtering berdampak** langsung pada peningkatan kepuasan pengguna karena mereka mendapatkan rekomendasi personal yang sesuai minatnya.

### Collaborative Filtering
Pada tahap ini, dilakukan evaluasi untuk mengukur performa sistem rekomendasi berbasis Collaborative Filtering. Berbeda dengan pendekatan Content-Based Filtering yang mengandalkan kemiripan fitur antar item, Collaborative Filtering memanfaatkan pola interaksi antar pengguna untuk memberikan rekomendasi.

Evaluasi dilakukan dengan fokus pada seberapa baik model dalam memprediksi rating pengguna terhadap anime, yang nantinya digunakan sebagai dasar pemeringkatan dan rekomendasi. Metrik utama yang digunakan adalah **Root Mean Squared Error (RMSE)** karena pendekatan ini bersifat prediktif terhadap nilai rating.

RMSE mengukur seberapa besar rata-rata kesalahan kuadrat dari prediksi terhadap nilai aktual. Nilai RMSE yang lebih rendah menunjukkan bahwa model memiliki prediksi yang lebih akurat.

**Rumus:**

$$
\text{RMSE} = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (\hat{r}_i - r_i)^2 }
$$

**Penjelasan:**

- $$\( \hat{r}_i \)$$: Rating yang diprediksi oleh model.
- $$\( r_i \)$$: Rating aktual dari pengguna.
- $$\( N \)$$: Jumlah total prediksi yang dievaluasi.

Berikut ini merupakan hasil training yang divisualisasikan pada grafik berikut:

![Evaluation Model Collaborative Filtering](https://raw.githubusercontent.com/rahmadnoorikhsan/Anime-System-Recommendation/main/resource/evaluation-collaborative.png)

Dari grafik tersebut, dapat disimpulkan bahwa model menunjukkan konvergensi yang baik. Setelah melalui proses pelatihan hingga epoch ke-25, diperoleh hasil sebagai berikut:

- **RMSE pada data pelatihan** stabil di kisaran **0.126**.
- **RMSE pada data validasi** stabil di kisaran **0.133**.

Nilai yang rendah dan stabil ini menunjukkan bahwa model memiliki tingkat generalisasi yang baik serta tidak mengalami overfitting secara signifikan. Dengan demikian, sistem rekomendasi berbasis Collaborative Filtering yang dibangun mampu memberikan prediksi rating yang cukup akurat untuk digunakan dalam menyusun rekomendasi bagi pengguna.

#### Dampak terhadap Business Understanding:
- Model Collaborative Filtering ini **berhasil menjawab Problem Statement Kedua**, karena mampu membangun pemahaman terhadap perilaku pengguna secara kolektif dan menghasilkan rekomendasi yang tetap relevan meskipun pengguna belum pernah menonton anime tertentu sebelumnya.
- **Goals Mengembangkan model machine learning yang dapat beradaptasi dan terus memperbaiki rekomendasinya seiring waktu dan perubahan selera pengguna** telah terpenuhi melalui Collaborative Filtering berbasis deep learning yang mempelajari interaksi antar pengguna dan dapat diperbarui secara berkala untuk menyesuaikan tren selera.
- **Solution statement dengan model Collaborative Filtering** memberikan dampak nyata dalam meningkatkan efektivitas personalisasi dan memperluas eksplorasi pengguna terhadap anime lain yang belum pernah mereka temui, meskipun berbeda genre.

### Kesimpulan dan Relevansi terhadap Tujuan Bisnis
Baik Content-Based Filtering maupun Collaborative Filtering telah berhasil menjawab seluruh problem statement yang diajukan di awal proyek. Setiap model memiliki kekuatan yang saling melengkapi:

- Content-Based unggul dalam memahami fitur dan preferensi eksplisit pengguna.

- Collaborative Filtering unggul dalam menangkap pola kolektif antar pengguna.

Gabungan keduanya dapat digunakan dalam strategi hybrid recommendation system untuk memaksimalkan performa sistem secara menyeluruh, sehingga selaras dengan tujuan bisnis untuk meningkatkan retensi pengguna, interaksi terhadap konten, dan kepuasan penggunaan aplikasi secara keseluruhan.

## Referensi
- Statista. (2021). Anime market size in Japan from 2002 to 2021. Retrieved from https://www.statista.com/statistics/1109105/japan-anime-market-size
- Soni, B., Thakuria, D., Nath, N., Das, N., & Boro, B. (2021). RikoNet: A Novel Anime Recommendation Engine. arXiv preprint arXiv:2106.12970. Retrieved from https://arxiv.org/abs/2106.12970
  
