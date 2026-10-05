# Домашнее задание: Spark SQL 
import csv

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("covid-homework")
    .master("local[*]")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# --- Загрузка данных ---------------------------------------------------------
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("covid-data.csv")
)

countries = df.filter(~F.col("iso_code").startswith("OWID_"))


def save(result, name):
    rows = result.collect()
    with open(f"result_{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(result.columns)
        writer.writerows([list(r) for r in rows])


# --- Задание 1 ---------------------------------------------------------------
# Выберите 15 стран с наибольшим процентом переболевших на 31 марта (в выходящем датасете необходимы колонки: iso_code, страна, процент переболевших)
task1 = (
    countries
    .filter(F.col("date") == "2021-03-31")
    .filter(F.col("total_cases").isNotNull() & F.col("population").isNotNull())
    .select(
        "iso_code",
        F.col("location").alias("страна"),
        F.round(F.col("total_cases") / F.col("population") * 100, 2).alias("процент_переболевших"),
    )
    .orderBy(F.col("процент_переболевших").desc())
    .limit(15)
)
print("\n=== Задание 1: топ-15 стран по проценту переболевших на 31.03.2021 ===")
task1.show(15, truncate=False)
save(task1, "task1")

# --- Задание 2 ---------------------------------------------------------------
# Top 10 стран с максимальным зафиксированным кол-вом новых случаев за последнюю неделю марта 
# 2021 в отсортированном порядке по убыванию
# (в выходящем датасете необходимы колонки: число, страна, кол-во новых случаев)
week = countries.filter(
    (F.col("date") >= "2021-03-25") & (F.col("date") <= "2021-03-31")
    & F.col("new_cases").isNotNull()
)
w = Window.partitionBy("location").orderBy(F.col("new_cases").desc())
task2 = (
    week
    .withColumn("rn", F.row_number().over(w))
    .filter(F.col("rn") == 1)
    .select(
        F.col("date").alias("число"),
        F.col("location").alias("страна"),
        F.col("new_cases").cast("long").alias("кол_во_новых_случаев"),
    )
    .orderBy(F.col("кол_во_новых_случаев").desc())
    .limit(10)
)
print("\n=== Задание 2: топ-10 стран по максимуму новых случаев, 25-31.03.2021 ===")
task2.show(truncate=False)
save(task2, "task2")

# --- Задание 3 ---------------------------------------------------------------
# Россия, последняя неделя марта 2021: изменение числа новых случаев
# относительно предыдущего дня. Для 25 марта "вчера" берём из 24 марта,
# поэтому окно считаем по всем данным, а фильтруем по датам уже после lag().
w_rus = Window.orderBy("date")
task3 = (
    countries
    .filter(F.col("location") == "Russia")
    .withColumn("кол_во_новых_случаев_вчера", F.lag("new_cases").over(w_rus))
    .withColumnRenamed("new_cases", "кол_во_новых_случаев_сегодня")
    .filter((F.col("date") >= "2021-03-25") & (F.col("date") <= "2021-03-31"))
    .select(
        F.col("date").alias("число"),
        F.col("кол_во_новых_случаев_вчера").cast("long"),
        F.col("кол_во_новых_случаев_сегодня").cast("long"),
        (F.col("кол_во_новых_случаев_сегодня") - F.col("кол_во_новых_случаев_вчера"))
        .cast("long").alias("дельта"),
    )
    .orderBy("число")
)
print("\n=== Задание 3: Россия, изменение новых случаев к предыдущему дню, 25-31.03.2021 ===")
task3.show(truncate=False)
save(task3, "task3")

spark.stop()
