////////////////////////////////////////////
//// LINEAR REGRESSION EXERCISE ///////////
/// Complete the commented tasks below ///
/////////////////////////////////////////

// Import LinearRegression

// Optional: Use the following code below to set the Error reporting
import org.apache.log4j._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import spark.implicits._
Logger.getLogger("org").setLevel(Level.ERROR)


// Start a simple Spark Session
val spark = SparkSession.builder().getOrCreate()
// Use Spark to read in the Ecommerce Customers csv file.
val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("Ecommerce Customers")
// Print the Schema of the DataFrame
data.printSchema()

// Print out an example Row
// Various ways to do this, just
// choose whichever way you prefer
data.head(1)

////////////////////////////////////////////////////
//// Setting Up DataFrame for Machine Learning ////
//////////////////////////////////////////////////

// A few things we need to do before Spark can accept the data!
// It needs to be in the form of two columns
// ("label","features")

// Import VectorAssembler and Vectors

// Rename the Yearly Amount Spent Column as "label"
// Also grab only the numerical columns from the data
// Set all of this as a new dataframe called df
data.columns
val df = data.select(data("Yearly Amount Spent").as("label"), $"Avg Session Length", $"Time on App", $"Time on Website", $"Length of Membership")
// An assembler converts the input values to a vector
// A vector is what the ML algorithm reads to train a model

// Use VectorAssembler to convert the input columns of df
// to a single output column of an array called "features"
// Set the input columns from which we are supposed to read the values.
// Call this new object assembler
val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length", "Time on App", "Time on Website", "Length of Membership")).setOutputCol("features")

// Use the assembler to transform our DataFrame to the two columns: label and features
val output = assembler.transform(df).select($"label", $"features")
output.printSchema()
// Create a Linear Regression Model object
val lr = new LinearRegression()

// Fit the model to the data and call this model lrModel
val lrModel = lr.fit(output)

// Print the coefficients and intercept for linear regression
lrModel.coefficients
lrModel.intercept
// Summarize the model over the training set and print out some metrics!
// Use the .summary method off your model to create an object
// called trainingSummary
val trainingSummary = lrModel.summary
trainingSummary
// Show the residuals, the RMSE, the MSE, and the R^2 Values.
trainingSummary.residuals.show()
trainingSummary.rootMeanSquaredError
trainingSummary.meanSquaredError
trainingSummary.r2

// Great Job!
