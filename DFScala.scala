import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()
val df = spark.read().csv("/home/b01t/Projects/scala/Scala-and-Spark-Bootcamp-master/Spark DataFrames/CitiGroup2006_2008")
