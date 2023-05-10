import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import time
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from PIL import Image




st.set_page_config(page_title="Hitters Salary Prediction", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "EDA & Visualizations", "Contact"])



if page == "Prediction":
    st.title("Hitters Salary Prediction")

    # Modeli yükleyin

    from joblib import load

    # Modelin tam yolunu kullanarak modeli yükleyin
    model_path = '724linear_regression_hitters.joblib'
    model = load(model_path)

    # Kullanıcıdan girdi alın
    # Özellikleri belirtin ve değerlerini isteyin
    # Örnek olarak 'AtBat', 'Hits' ve 'HmRun' özelliklerini kullanalım

    def user_input_features():
        League_mm = st.sidebar.selectbox('League', ('A', 'N'))
        Division_mm = st.sidebar.selectbox('Division', ('E', 'W'))
        #NewLeague_mm = st.sidebar.selectbox('NewLeague', ('A', 'N'))
        NewLeague_mm = st.sidebar.radio("What\'s players New League",('A', 'N'))
        AtBat_mm = st.sidebar.slider('AtBat', 32.1, 380.93, 687.9)
        Hits_mm = st.sidebar.slider('Hits', 13.1, 101.02, 238.00)
        HmRun_mm = st.sidebar.slider('HmRun', 0.00, 10.77, 40.00)
        Runs_mm = st.sidebar.number_input('Runs')
        RBI_mm = st.sidebar.slider('RBI', 0.00, 48.03, 121.00)
        Walks_mm = st.sidebar.slider('Walks ', 0.00, 38.74, 105.00)
        Years_mm = st.sidebar.slider('Years', 1.00, 7.44, 24.00)
        CAtBat_mm = st.sidebar.slider('CAtBat', 19.00, 15000.00, 2648.68)
        CHits_mm = st.sidebar.slider('CHits', 4.00, 717.57, 4256.00)
        CHmRun_mm = st.sidebar.slider('CHmRun', 0.00, 69.49, 548.00)
        CRuns_mm = st.sidebar.slider('CRuns', 1.00, 358.80, 2165.00)
        CRBI_mm = st.sidebar.slider('CRBI', 0.00, 330.12, 1659.00)
        CWalks_mm = st.sidebar.slider('CWalks', 0.00, 260.24, 1566.00)
        PutOuts_mm = st.sidebar.slider('PutOuts', 0.00, 288.94, 1378.00)
        Assists_mm = st.sidebar.slider('Assists', 0.00, 106.91, 492.00)
        Errors_mm = st.sidebar.number_input('Errors', min_value=0, step=1)
        data = {'AtBat': AtBat_mm,
                'Hits': Hits_mm,
                'HmRun': HmRun_mm,
                'Runs': Runs_mm,
                'RBI': RBI_mm,
                'Walks': Walks_mm,
                'Years': Years_mm,
                'CAtBat': CAtBat_mm,
                'CHits': CHits_mm,
                'CHmRun': CHmRun_mm,
                'CRuns': CRuns_mm,
                'CRBI': CRBI_mm,
                'CWalks': CWalks_mm,
                'League': League_mm,
                'Division': Division_mm,
                'PutOuts': PutOuts_mm,
                'Assists': Assists_mm,
                'Errors': Errors_mm,
                'NewLeague':NewLeague_mm}

        features = pd.DataFrame(data, index=[0])
        return features


    input_df = user_input_features()

    st.dataframe(input_df)

    # Veri setini yükle
    url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Hitters.csv"
    df = pd.read_csv(url)
    df = df.drop("Unnamed: 0", axis=1)

    # NA değerlerini kaldırın
    df = df.dropna()


    birlesmis = pd.concat([input_df, df], axis=0)
    st.write("Kulalnıcı bilgilerini ve veri setini birleştirelim.")
    st.write(birlesmis.head())


    def outlier_thresholds(df, col_name, q1=0.25, q3=0.75):
        quartile1 = df[col_name].quantile(q1)
        quartile3 = df[col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit


    def replace_with_thresholds(df, variable):
        low_limit, up_limit = outlier_thresholds(df, variable)
        df.loc[(df[variable] < low_limit), variable] = low_limit
        df.loc[(df[variable] > up_limit), variable] = up_limit


    def check_outlier(df, col_name):
        low_limit, up_limit = outlier_thresholds(df, col_name)
        if df[(df[col_name] > up_limit) | (df[col_name] < low_limit)].any(axis=None):
            return True
        else:
            return False


    def grab_col_names(df, cat_th=10, car_th=20):
        """

        Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
        Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

        Parameters
        ------
            df: df
                    Değişken isimleri alınmak istenilen df
            cat_th: int, optional
                    numerik fakat kategorik olan değişkenler için sınıf eşik değeri
            car_th: int, optinal
                    kategorik fakat kardinal değişkenler için sınıf eşik değeri

        Returns
        ------
            cat_cols: list
                    Kategorik değişken listesi
            num_cols: list
                    Numerik değişken listesi
            cat_but_car: list
                    Kategorik görünümlü kardinal değişken listesi

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            print(grab_col_names(df))


        Notes
        ------
            cat_cols + num_cols + cat_but_car = toplam değişken sayısı
            num_but_cat cat_cols'un içerisinde.
            Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

        """

        # cat_cols, cat_but_car
        cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
        num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and
                       df[col].dtypes != "O"]
        cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and
                       df[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # num_cols
        num_cols = [col for col in df.columns if df[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        print(f"Observations: {df.shape[0]}")
        print(f"Variables: {df.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')
        return cat_cols, num_cols, cat_but_car


    cat_cols, num_cols, cat_but_car = grab_col_names(birlesmis)

    #############################################
    # 1. Outliers (Aykırı Değerler)
    #############################################

    for col in num_cols:
        print(col, check_outlier(birlesmis, col))

    for col in num_cols:
        if check_outlier(birlesmis, col):
            replace_with_thresholds(birlesmis, col)

    #############################################
    # 3. Feature Extraction (Özellik Çıkarımı)
    #############################################
    birlesmis.columns = [col.upper() for col in birlesmis.columns]
    cat_cols, num_cols, cat_but_car = grab_col_names(birlesmis)

    birlesmis[num_cols] = birlesmis[num_cols] + 0.0000000000001


    birlesmis["NEW_C_RUNS_RATIO"] = birlesmis["RUNS"] / birlesmis["CRUNS"]
    # CAREER BAT RATIO
    birlesmis["NEW_C_ATBAT_RATIO"] = birlesmis["ATBAT"] / birlesmis["CATBAT"]
    # CAREER HITS RATIO
    birlesmis["NEW_C_HITS_RATIO"] = birlesmis["HITS"] / birlesmis["CHITS"]
    # CAREER HMRUN RATIO
    birlesmis["NEW_C_HMRUN_RATIO"] = birlesmis["HMRUN"] / birlesmis["CHMRUN"]
    # CAREER RBI RATIO
    birlesmis["NEW_C_RBI_RATIO"] = birlesmis["RBI"] / birlesmis["CRBI"]
    # CAREER WALKS RATIO
    birlesmis["NEW_C_WALKS_RATIO"] = birlesmis["WALKS"] / birlesmis["CWALKS"]
    birlesmis["NEW_C_HIT_RATE"] = birlesmis["CHITS"] / birlesmis["CATBAT"]
    # PLAYER TYPE : RUNNER
    birlesmis["NEW_C_RUNNER"] = birlesmis["CRBI"] / birlesmis["CHITS"]
    # PLAYER TYPE : HIT AND RUN
    birlesmis["NEW_C_HIT-AND-RUN"] = birlesmis["CRUNS"] / birlesmis["CHITS"]
    # MOST VALUABLE HIT RATIO IN HITS
    birlesmis["NEW_C_HMHITS_RATIO"] = birlesmis["CHMRUN"] / birlesmis["CHITS"]
    # MOST VALUABLE HIT RATIO IN ALL SHOTS
    birlesmis["NEW_C_HMATBAT_RATIO"] = birlesmis["CATBAT"] / birlesmis["CHMRUN"]

    # Annual Averages
    birlesmis["NEW_CATBAT_MEAN"] = birlesmis["CATBAT"] / birlesmis["YEARS"]
    birlesmis["NEW_CHITS_MEAN"] = birlesmis["CHITS"] / birlesmis["YEARS"]
    birlesmis["NEW_CHMRUN_MEAN"] = birlesmis["CHMRUN"] / birlesmis["YEARS"]
    birlesmis["NEW_CRUNS_MEAN"] = birlesmis["CRUNS"] / birlesmis["YEARS"]
    birlesmis["NEW_CRBI_MEAN"] = birlesmis["CRBI"] / birlesmis["YEARS"]
    birlesmis["NEW_CWALKS_MEAN"] = birlesmis["CWALKS"] / birlesmis["YEARS"]

    # PLAYER LEVEL
    birlesmis.loc[(birlesmis["YEARS"] <= 2), "NEW_YEARS_LEVEL"] = "Junior"
    birlesmis.loc[(birlesmis["YEARS"] > 2) & (birlesmis['YEARS'] <= 5), "NEW_YEARS_LEVEL"] = "Mid"
    birlesmis.loc[(birlesmis["YEARS"] > 5) & (birlesmis['YEARS'] <= 10), "NEW_YEARS_LEVEL"] = "Senior"
    birlesmis.loc[(birlesmis["YEARS"] > 10), "NEW_YEARS_LEVEL"] = "Expert"

    # PLAYER LEVEL X DIVISION

    birlesmis.loc[(birlesmis["NEW_YEARS_LEVEL"] == "Junior") & (birlesmis["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Junior-East"
    birlesmis.loc[(birlesmis["NEW_YEARS_LEVEL"] == "Junior") & (birlesmis["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Junior-West"
    birlesmis.loc[(birlesmis["NEW_YEARS_LEVEL"] == "Mid") & (birlesmis["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Mid-East"
    birlesmis.loc[(birlesmis["NEW_YEARS_LEVEL"] == "Mid") & (birlesmis["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Mid-West"
    birlesmis.loc[(birlesmis["NEW_YEARS_LEVEL"] == "Senior") & (birlesmis["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Senior-East"
    birlesmis.loc[(birlesmis["NEW_YEARS_LEVEL"] == "Senior") & (birlesmis["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Senior-West"
    birlesmis.loc[(birlesmis["NEW_YEARS_LEVEL"] == "Expert") & (birlesmis["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Expert-East"
    birlesmis.loc[(birlesmis["NEW_YEARS_LEVEL"] == "Expert") & (birlesmis["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Expert-West"

    # Player Promotion to Next League
    birlesmis.loc[(birlesmis["LEAGUE"] == "N") & (birlesmis["NEWLEAGUE"] == "N"), "NEW_PLAYER_PROGRESS"] = "StandN"
    birlesmis.loc[(birlesmis["LEAGUE"] == "A") & (birlesmis["NEWLEAGUE"] == "A"), "NEW_PLAYER_PROGRESS"] = "StandA"
    birlesmis.loc[(birlesmis["LEAGUE"] == "N") & (birlesmis["NEWLEAGUE"] == "A"), "NEW_PLAYER_PROGRESS"] = "Descend"
    birlesmis.loc[(birlesmis["LEAGUE"] == "A") & (birlesmis["NEWLEAGUE"] == "N"), "NEW_PLAYER_PROGRESS"] = "Ascend"



    cat_cols, num_cols, cat_but_car = grab_col_names(birlesmis)


    # Label Encoding
    def label_encoder(dataframe, binary_col):
        labelencoder = LabelEncoder()
        dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
        return dataframe


    binary_cols = [col for col in birlesmis.columns if
                   birlesmis[col].dtype not in [int, float] and birlesmis[col].nunique() == 2]

    for col in binary_cols:
        birlesmis = label_encoder(birlesmis, col)


    def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
        dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
        return dataframe


    ohe_cols = [col for col in birlesmis.columns if 10 >= birlesmis[col].nunique() > 2]
    birlesmis = one_hot_encoder(birlesmis, ohe_cols, drop_first=True)
    cat_cols, num_cols, cat_but_car = grab_col_names(birlesmis)

    # 7. Robust-Scaler
    for col in num_cols:
        transformer = RobustScaler().fit(birlesmis[[col]])
        birlesmis[col] = transformer.transform(birlesmis[[col]])




    #inputa fe uyguladıktan sonra çekelim predict için

    tahmin_edilecek = birlesmis[:1]
    #model_icin = birlesmis[1:]

    st.write("------------------------------")



    tahmin_edilecek.drop("SALARY",axis=1,inplace=True)


    # Hedef değişkeni ve özellikleri belirleyin
    #X = model_icin.drop('SALARY', axis=1)
    #y = model_icin['SALARY']

    # Verileri eğitim ve test setlerine ayırın
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #reg_model = LinearRegression()
    #reg_model.fit(X_train, y_train)

    #prediction = reg_model.predict(tahmin_edilecek)
    #st.title("Maaş Tahmini")
    #st.write(prediction)
    #st.write(tahmin_edilecek.columns)

    prediction = model.predict(tahmin_edilecek)
    #st.subheader('Tahmin Edilen Maaş')
    #st.write(st.write(prediction))


    st.title('İstatistikleri verilen sporcu için maaş tahmini :baseball:')
    tahminler_df = pd.DataFrame({'Tahmin': prediction})
    st.write(tahminler_df)


    #başa al
    import requests
    import xml.etree.ElementTree as ET
    import streamlit as st

    # Merkez Bankası XML dosyasının URL'si
    url = "https://www.tcmb.gov.tr/kurlar/today.xml"

    # XML dosyasını alın
    response = requests.get(url)
    root = ET.fromstring(response.content)

    # dolar kurunu bulma
    dolar_kuru = ""
    for child in root:
        if child.tag == "Currency":
            if child.attrib['Kod'] == "USD":
                dolar_kuru = child.find('ForexBuying').text

    # dolar kuru ile prediction değerini çarpma
    tl_degeri = float(dolar_kuru) * prediction

    int_degeri = int(tl_degeri[0])


    st.subheader("Türkiye'deki Dolar Kuru 💲: " + dolar_kuru )
    st.subheader("🇹🇷 Cinsinden Maaş 👉" + str(float(int_degeri)))
















elif page == "EDA & Visualizations":
    st.title("EDA & Visualizations")

    # Add your EDA and visualizations here
    




    # Veri kümesini yükleyin
    url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Hitters.csv"
    data = pd.read_csv(url)
    data = data.drop("Unnamed: 0", axis=1)

    # Streamlit başlığı
    st.title("Hitters Veri Kümesi İstatistiksel Raporlar ve Görselleştirmeler")

    # Veri kümesini göster
    st.write("### Veri Kümesi:")
    st.write(data.head())

    show_profile_report = st.checkbox("Pandas Profiling Raporunu Göster")

    if show_profile_report:
        st.write("### Pandas Profiling Raporu:")
        profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)
        st_profile_report(profile)

        # Veri kümesinin özellik dağılımlarını göster
        st.write("### Özellik Dağılımları:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=data, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Sütunlar arasındaki ilişkileri göster
        st.write("### Sütunlar Arası İlişkiler:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.pairplot(data=data, diag_kind="kde")
        st.pyplot(fig)

        # Korelasyon matrisini göster
        st.write("### Korelasyon Matrisi:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", linewidths=.5, ax=ax)
        st.pyplot(fig)






else:
    st.title("Contact")




    col1, col2, = st.columns(2)

    with col1:
        st.title("Miuul")
        image_url = "https://miuul.com/image/theme/logo-dark.png"
        st.image(image_url, use_column_width=True)

    with col2:
        st.title("Veri Bilimi Okulu")
        image_url = "https://www.veribilimiokulu.com/wp-content/uploads/2020/12/veribilimiokulu_logo-crop.png"
        st.image(image_url, use_column_width=True)

    col1, col2 = st.columns(2)

    with col1:
        link = "[Miuul](https://miuul.com)"
        st.markdown(link, unsafe_allow_html=True)

    with col2:
        link = "[Veri Bilimi Okulu](https://bootcamp.veribilimiokulu.com/bootcamp-programlari/veri-bilimci-yetistirme-programi/)"
        st.markdown(link, unsafe_allow_html=True)

    col1, = st.columns(1)

    with col1:
        st.title("Kaynakça 🎤")
        video_url = "https://www.youtube.com/watch?v=Ww9M0WJfGN8"
        st.video(video_url)

    col3, = st.columns(1)
    with col3:
        st.title("Linkedin")
        st.write("[Linkedin](https://www.linkedin.com/in/furkanbyc/)")

    #Data Vaders :)

    st.title("Best Group :)")


    video_path = 'vaders_video.mp4'
    

    #video_file = open(video_path,'rb')
    #video_bytes = video_file.read()
    #st.video(video_bytes)









