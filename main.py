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
data = data.rename(columns={"current/discounted_price": "total_price"})

# - price_on_variant – (очень странный столбец с непонятными данными. в описании нет пояснений к нему. убираем)
data = data.drop("price_on_variant", axis=1)

# + listed_price – Изначальная цена (есть строки со значением "No Discount". Такие надо заменить на значения из total_price)
data["listed_price"] = data["listed_price"].replace("No Discount", np.nan)
data["listed_price"] = data["listed_price"].str.replace("[$,]", "", regex=True).astype(float)
data["listed_price"] = data["listed_price"].fillna(data["total_price"]).round(2)

# - is_best_seller – Указывает, помечен ли товар как «Бестселлер» (не представляет интереса. убираем)
data = data.drop("is_best_seller", axis=1)

# + is_sponsored – Является ли товар рекламным или попал в рекомендации естественным образом (два значения "Organic" - 0, "Sponsored" - 1)
data["is_sponsored"] = data["is_sponsored"].map(lambda x: 1 if x == "Sponsored" else 0)

# + is_couponed – Наличие специальных скидочных купонов, если есть то стоимость (где "No Coupon" заменим на 0, остальное на номинал купона)
coupon = data["is_couponed"].astype(str)
mask_no = coupon.str.contains("No Coupon", case=False, na=False)
mask_dollar = coupon.str.contains(r"\$", na=False)
dollar_val = (coupon.str.extract(r"(\d+\.?\d*)")[0].astype(float))
mask_dollar_ge10 = mask_dollar & (dollar_val >= 10)
mask_dollar_lt10 = mask_dollar & (dollar_val < 10)
mask_percent = coupon.str.contains(r"%", na=False)
percent_val = (coupon.str.extract(r"(\d+\.?\d*)")[0].astype(float))
mask_percent_ge25 = mask_percent & (percent_val >= 25)
mask_percent_lt25 = mask_percent & (percent_val < 25)
data["is_couponed"] = np.select(
    [
        mask_no,
        mask_dollar_ge10,
        mask_dollar_lt10,
        mask_percent_ge25,
        mask_percent_lt25
    ],
    [
        "no",
        ">=10$",
        "<10$",
        ">=25%",
        "<25%"
    ],
    default="no"
)

# + buy_box_availability – Наличие кнопки BuyBox на странице поиска Amazon (в описании null означает False, "Add to cart" - True. На них и заменим)
data["buy_box_availability"] = data["buy_box_availability"].map(lambda x: 1 if x == "Add to cart" else 0)

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
title                   0
rating                  0
number_of_reviews       0
bought_in_last_month    0
total_price             0
listed_price            0
is_sponsored            0
is_couponed             0
buy_box_availability    0
'''

data.to_csv("data/amazon_products_sales_data_full.csv", index=False)


'''2. Нормализуем данные'''
from sklearn.preprocessing import MinMaxScaler

numeric_cols = [
    "rating",
    "number_of_reviews",
    "bought_in_last_month",
    "total_price",
    "listed_price"
]

scaler = MinMaxScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Создадим более объективную характеристики товара "Репутация" (Среднее арифметическое от нормализованных значений двух столбцов)
data["reputation"] = data[["rating", "number_of_reviews"]].mean(axis=1)
data = data.drop(columns=["rating", "number_of_reviews"])

# Сгруппируем столбец
def demand_group(x):
    if 0 <= x <= 0.01:
        return "low"
    elif 0.01 < x <= 0.1:
        return "middle"
    else:
        return "high"

data["demand"] = data["bought_in_last_month"].apply(demand_group)
data = data.drop(columns=["bought_in_last_month"])

data.to_csv("data/amazon_products_sales_data_normalized.csv", index=False)


'''3. Удалим дубликаты'''
# print(data.duplicated().sum())
'''33413'''
# print(data.shape[0])
'''42675'''

# Удалить только точные копии строк (по всем столбцам):
data = data.drop_duplicates()

# print(data.shape[0])
'''9262'''

data.to_csv("data/amazon_products_sales_data_cleaned.csv", index=False)


'''4. Визуализируем данные'''
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


# Диаграммы рассеяния
plt.figure(figsize=(10,7))
plt.scatter('total_price', 'listed_price', data=data)
plt.xlabel("Total Price")
plt.ylabel("Listed Price")
plt.savefig("images/scatter_total_vs_listed.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10,7))
plt.scatter('total_price', 'reputation', data=data)
plt.xlabel("Total Price")
plt.ylabel("Reputation")
plt.savefig("images/scatter_total_vs_reputation.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10,7))
plt.scatter('listed_price', 'reputation', data=data)
plt.xlabel("Listed Price")
plt.ylabel("Reputation")
plt.savefig("images/scatter_listed_vs_reputation.png", dpi=300, bbox_inches="tight")
plt.close()


# Ящики с усами
fig = go.Figure()
for group, color in zip(["low", "middle", "high"], ["blue", "green", "red"]):
    fig.add_trace(go.Box(
        y=data.loc[data["demand"] == group, "total_price"],
        name=group,
        marker_color=color
    ))
fig.update_layout(title="Boxplot: Total Price по Demand", yaxis_title="Total Price")
fig.write_image("images/box_total_price_by_demand.png")

fig = go.Figure()
for group, color in zip(["low", "middle", "high"], ["blue", "green", "red"]):
    fig.add_trace(go.Box(
        y=data.loc[data["demand"] == group, "reputation"],
        name=group,
        marker_color=color
    ))
fig.update_layout(title="Boxplot: Reputation по Demand", yaxis_title="Reputation")
fig.write_image("images/box_reputation_by_demand.png")

fig = go.Figure()
for group, color in zip(["low", "middle", "high"], ["blue", "green", "red"]):
    fig.add_trace(go.Box(
        y=data.loc[data["demand"] == group, "listed_price"],
        name=group,
        marker_color=color
    ))
fig.update_layout(title="Boxplot: Listed Price по Demand", yaxis_title="Listed Price")
fig.write_image("images/box_listed_price_by_demand.png")

fig = go.Figure()
for group, color in zip(data["is_couponed"].unique(), ["blue", "green", "red", "orange", "purple"]):
    fig.add_trace(go.Box(
        y=data.loc[data["is_couponed"] == group, "total_price"],
        name=group,
        marker_color=color
    ))
fig.update_layout(title="Boxplot: Total Price по Is Couponed", yaxis_title="Total Price")
fig.write_image("images/box_total_price_by_couponed.png")

fig = go.Figure()
for group, color in zip(data["is_couponed"].unique(), ["blue", "green", "red", "orange", "purple"]):
    fig.add_trace(go.Box(
        y=data.loc[data["is_couponed"] == group, "reputation"],
        name=group,
        marker_color=color
    ))
fig.update_layout(title="Boxplot: Reputation по Is Couponed", yaxis_title="Reputation")
fig.write_image("images/box_reputation_by_couponed.png")

fig = go.Figure()
for group, color in zip(data["is_couponed"].unique(), ["blue", "green", "red", "orange", "purple"]):
    fig.add_trace(go.Box(
        y=data.loc[data["is_couponed"] == group, "listed_price"],
        name=group,
        marker_color=color
    ))
fig.update_layout(title="Boxplot: Listed Price по Is Couponed", yaxis_title="Listed Price")
fig.write_image("images/box_listed_price_by_couponed.png")


# Гистограммы
plt.figure(figsize=(10, 7))
sns.countplot(data=data, x="demand", hue="demand", palette="Set1", order=["low", "middle", "high"])
plt.xlabel("Demand")
plt.ylabel("Count")
plt.savefig("images/hist_demand.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 7))
sns.countplot(data=data, x="is_couponed", hue="is_couponed", palette="Set2", order=data["is_couponed"].unique())
plt.xlabel("Is Couponed")
plt.ylabel("Count")
plt.savefig("images/hist_is_couponed.png", dpi=300, bbox_inches="tight")
plt.close()


'''5. Статистический анализ'''
numeric_cols = [
    "reputation",
    "total_price",
    "listed_price"
]

# Первичная матрица pairplot
sns.pairplot(data[numeric_cols])
plt.savefig("images/pairplot_all.png", dpi=300, bbox_inches="tight")
plt.close()

# Функция для поиска выбросов по 3 сигмам
def outliers_indices(df, feature):
    mean = df[feature].mean()
    std = df[feature].std()
    return df[(df[feature] < mean - 3*std) | (df[feature] > mean + 3*std)].index
outliers_sets = [outliers_indices(data, col) for col in numeric_cols]
outliers_all = set().union(*outliers_sets)
# print(len(outliers_all))
data_clean = data.drop(outliers_all)

# Матрица pairplot без выбросов
sns.pairplot(data_clean[numeric_cols])
plt.savefig("images/pairplot_clean.png", dpi=300, bbox_inches="tight")
plt.close()

# Корреляционная матрица
plt.figure(figsize=(10,7))
sns.heatmap(data_clean[numeric_cols].corr(method='spearman'), annot=True, fmt=".2f", cmap="coolwarm")
plt.savefig("images/heatmap_corr.png", dpi=300, bbox_inches="tight")
plt.close()
