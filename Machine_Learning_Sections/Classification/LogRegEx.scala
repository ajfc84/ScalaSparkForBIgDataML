import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import spark.implicits._

val spark = SparkSession.builder().getOrCreate()

var data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("titanic.csv")

data.printSchema()

val df = (data.select(data("Survived").as("label"),
  $"Pclass", $"Name", $"Sex", $"Age", $"SibSp", $"Parch", $"Fare", $"Embarked"))

val dfna = df.na.drop()

val sexIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex")

val sexEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVector")
val embarkedEncoder = new OneHotEncoder().setInputCol("EmbarkedIndex").setOutputCol("EmbarkedVector")

val assembler = new VectorAssembler().setInputCols(Array("Pclass", "SexVector", "Age", "SibSp", "Parch", "Fare", "EmbarkedVector")).setOutputCol("features")

val Array(training, test) = dfna.randomSplit(Array(0.7, 0.3), seed=12345)

val lr = new LogisticRegression()

val pipeline = new Pipeline().setStages(Array(sexIndexer, sexEncoder, embarkedIndexer, embarkedEncoder, assembler, lr))

val model = pipeline.fit(training)

val results = model.transform(test)


val predictionAndLabels = results.select($"prediction", $"label")