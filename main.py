import pandas as pd
import numpy as np
pd.plotting.register_matplotlib_converters()


filepath = "data/amazon_products_sales_data_uncleaned.csv"
data = pd.read_csv(filepath)


'''0. Посмотрим на количество пропусков'''
# print(data.isna().sum())
'''
title                           0
rating                       1024
number_of_reviews            1024
bought_in_last_month         3217
current/discounted_price    11749
price_on_variant                0
listed_price                    0
is_best_seller                  0
is_sponsored                    0
is_couponed                     0
buy_box_availability        14653
delivery_details            11720
sustainability_badges       39267
image_url                       0
product_url                  2069
collected_at                    0
'''


'''1. Подготовим данные (приведём к нужному формату)'''
# + title – Полное название товара (готовая строка)
data["title"] = data["title"].str.lower()

# + rating – Средний рейтинг покупателей по пятибалльной шкале (преобразуем в число с плавающей точкой, с точностью до десятых)
data["rating"] = (
    data["rating"]
    .str.extract(r"(\d+(?:\.\d+)?)")
    .astype(float)
    .round(1)
)
median_val = data["rating"].median()
data["rating"] = data["rating"].fillna(median_val)

# + number_of_reviews – Общее количество отзывов покупателей (преобразуем в число)
data["number_of_reviews"] = (
    data["number_of_reviews"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .replace("nan", np.nan)
    .astype(float)
)
median_val = data["number_of_reviews"].median()
data["number_of_reviews"] = data["number_of_reviews"].fillna(median_val)
data["number_of_reviews"] = data["number_of_reviews"].astype(int)

# + bought_in_last_month – Количество единиц, купленных за последний месяц (преобразуем в натуральное число)
bought_raw = data["bought_in_last_month"].astype(str)
numbers = bought_raw.str.extract(r"(\d+\.?\d*)([KM]?)")[0]
suffixes = bought_raw.str.extract(r"(\d+\.?\d*)([KM]?)")[1]
data["bought_in_last_month"] = pd.to_numeric(numbers, errors="coerce")
data["bought_in_last_month"] = np.where(
    suffixes == "K", data["bought_in_last_month"] * 1000,
    np.where(suffixes == "M", data["bought_in_last_month"] * 1_000_000, data["bought_in_last_month"])
)
median_val = data["bought_in_last_month"].median()
data["bought_in_last_month"] = data["bought_in_last_month"].fillna(median_val).astype(int)

# + current/discounted_price – Текущая цена после скидки (число с плавающей точкой, с точностью до сотых)
data["current/discounted_price"] = pd.to_numeric(data["current/discounted_price"], errors="coerce").round(2)
median_val = data["current/discounted_price"].median()
data["current/discounted_price"] = data["current/discounted_price"].fillna(median_val)

# - price_on_variant – (очень странный столбец с непонятными данными. в описании нет пояснений к нему. убираем)
data = data.drop("price_on_variant", axis=1)

# + listed_price – Изначальная цена (есть строки со значением "No Discount". Такие надо заменить на значения из current/discounted_price)
data["listed_price"] = data["listed_price"].replace("No Discount", np.nan)
data["listed_price"] = data["listed_price"].str.replace("[$,]", "", regex=True).astype(float)
data["listed_price"] = data["listed_price"].fillna(data["current/discounted_price"]).round(2)

# + is_best_seller – Указывает, помечен ли товар как «Бестселлер» (имеется несколько строковых значений)
data["is_best_seller"] = data["is_best_seller"].str.lower()

# + is_sponsored – Является ли товар рекламным или попал в рекомендации естественным образом (два значения "Organic" - False, "Sponsored" - True)
data["is_sponsored"] = data["is_sponsored"].map(lambda x: True if x == "Sponsored" else False)

# + is_couponed – Наличие специальных скидочных купонов, если есть то стоимость (где "No Coupon" заменим на 0, остальное на номинал купона)
is_percent = data["is_couponed"].str.contains("%")
numbers = data["is_couponed"].str.extract(r"(\d+\.?\d*)")[0].astype(float)
data["is_couponed"] = np.where(
    data["is_couponed"].str.contains("No Coupon"),
    0,
    np.where(
        is_percent,
        (data["current/discounted_price"] * numbers / 100).round(2),
        numbers.round(2)
    )
)

# + buy_box_availability – Наличие кнопки BuyBox на странице поиска Amazon (в описании null означает False, "Add to cart" - True. На них и заменим)
data["buy_box_availability"] = data["buy_box_availability"].map(lambda x: True if x == "Add to cart" else False)

# - delivery_details – Ожидаемая дата доставки (зависит от даты заказа. убираем)
# - sustainability_badges – Теги, связанные с экологичностью и устойчивым развитием (слишком много пропусков. заполнить их нереально: это строки с описанием. убираем)
# - image_url – Прямая ссылка на изображение товара (не представляет интереса для анализа. убираем)
# - product_url – Официальная страница товара на Amazon (не представляет интереса для анализа. убираем)
# - collected_at – Дата, когда были собраны данные (не представляет интереса для анализа. убираем)
data = data.drop(columns=[
    "delivery_details",
    "sustainability_badges",
    "image_url",
    "product_url",
    "collected_at"
])

'''Проверим количество пропусков'''
# print(data.isna().sum())
'''
title                       0
rating                      0
number_of_reviews           0
bought_in_last_month        0
current/discounted_price    0
listed_price                0
is_best_seller              0
is_sponsored                0
is_couponed                 0
buy_box_availability        0
'''

data.to_csv("data/amazon_products_sales_data_full.csv", index=False)


'''2. Нормализуем данные'''
from sklearn.preprocessing import MinMaxScaler

numeric_cols = [
    "rating",
    "number_of_reviews",
    "bought_in_last_month",
    "current/discounted_price",
    "listed_price",
    "is_couponed"
]

scaler = MinMaxScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

data.to_csv("data/amazon_products_sales_data_normalized.csv", index=False)


'''3. Удалим дубликаты'''
# print(data.duplicated().sum())
# print(data.shape[0])

# Удалить только точные копии строк (по всем столбцам):
data = data.drop_duplicates()

# Удалить дубликаты только по названию (title):
# data = data.drop_duplicates(subset=["title"])

# Удалить дубликаты по названию и цене:
# data = data.drop_duplicates(subset=["title", "listed_price"])

# print(data.shape[0])

data.to_csv("data/amazon_products_sales_data_cleaned.csv", index=False)


'''4. Визуализируем данные'''
import matplotlib.pyplot as plt
import seaborn as sns


# Диаграмма рассеяния
plt.figure(figsize=(10,7))
plt.scatter(
    data["current/discounted_price"],
    data["number_of_reviews"],
    alpha=0.6, edgecolor='k'
)
plt.xlabel("Discounted Price (normalized)")
plt.ylabel("Number of Reviews (normalized)")
plt.title("Price vs Reviews")
plt.show()

# Ящик с усами
plt.figure(figsize=(8,6))
sns.boxplot(y=data["bought_in_last_month"])
plt.ylabel("Bought in Last Month (normalized)")
plt.title("Distribution of Units Bought in Last Month")
plt.show()

# Гистограмма
plt.figure(figsize=(8,6))
plt.hist(data["current/discounted_price"], bins=30, color="skyblue", edgecolor="black")
plt.xlabel("Discounted Price (normalized)")
plt.ylabel("Number of Products")
plt.title("Distribution of Discounted Price")
plt.show()


'''5. Статистический анализ'''
# Первичная матрица pairplot
sns.pairplot(data[numeric_cols])
plt.suptitle("Pairplot: первичные данные", y=1.02)
plt.show()

# Функция для поиска выбросов по 3 сигмам
def outliers_indices(df, feature):
    mean = df[feature].mean()
    std = df[feature].std()
    return df[(df[feature] < mean - 3*std) | (df[feature] > mean + 3*std)].index
outliers_sets = [outliers_indices(data, col) for col in numeric_cols]
outliers_all = set().union(*outliers_sets)
# print(f"Количество выбросов, которые будут удалены: {len(outliers_all)}")
data_clean = data.drop(outliers_all)

# Матрица pairplot без выбросов
sns.pairplot(data_clean[numeric_cols])
plt.suptitle("Pairplot: данные без выбросов", y=1.02)
plt.show()

# Корреляционная матрица
plt.figure(figsize=(8,6))
sns.heatmap(data_clean[numeric_cols].corr(method='spearman'), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Spearman Correlation Matrix")
plt.show()