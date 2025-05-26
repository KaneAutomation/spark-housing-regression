from pyspark.sql import SparkSession

# Start a Spark session
spark = SparkSession.builder \
    .appName("Housing Price Prediction") \
    .getOrCreate()

# Load the CSV file locally
df = spark.read.csv("housing.csv", header=True, inferSchema=True)

# Show the first few rows of data
df.show(5)

# Show column names and data types
df.printSchema()

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Step 1: Assemble the feature column
assembler = VectorAssembler(
    inputCols=["2018 Population estimate"],
    outputCol="features"
)
data = assembler.transform(df)

# Step 2: Select features and label
final_data = data.select("features", "`2019 median sales price`")

# Step 3: Split into train and test sets
train_data, test_data = final_data.randomSplit([0.8, 0.2])

# Step 4: Train the model
lr = LinearRegression(labelCol="2019 median sales price")
lr_model = lr.fit(train_data)

# Step 5: Evaluate the model
test_results = lr_model.evaluate(test_data)

print("R²:", test_results.r2)
print("RMSE:", test_results.rootMeanSquaredError)
print("Coefficients:", lr_model.coefficients)
print("Intercept:", lr_model.intercept)
