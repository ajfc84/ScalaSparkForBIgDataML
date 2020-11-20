import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import spark.implicits._
import org.apache.spark.ml.feature.VectorAssembler

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("USA_Housing.csv")

// ("Label", "features")
val df = data.select(data("Price").as("label"), $"Avg Area Income", $"Avg Area House Age", $"Avg Area Number of Rooms", $"Avg Area Number of Bedrooms", $"Area Population")
val assembler = (new VectorAssembler()
  .setInputCols(Array(
    "Avg Area Income",
    "Avg Area House Age",
    "Avg Area Number of Rooms",
    "Avg Area Number of Bedrooms",
    "Area Population"))
  .setOutputCol("features"))
val output = assembler.transform(df).select($"label", $"features")

val lr = new LinearRegression()

val lrModel = lr.fit(output)

val trainingSummary = lrModel.summary

trainingSummary.residuals.show()