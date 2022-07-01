package Linear

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, not}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.catalyst.dsl.expressions.StringToAttributeConversionHelper



object Aversiahora {

  def main(args: Array[String]): Unit = {
      //Reducir el número de LOG
      Logger.getLogger("org").setLevel(Level.OFF)
      //Creando el contexto del Servidor
      val sc = new SparkContext("local","winequality", System.getenv("SPARK_HOME"))
      val spark = SparkSession
        .builder()
        .master("local")
        .appName("CargaJSON")
        .config("log4j.rootCategory", "ERROR, console")
        .getOrCreate()

      var miDF=spark.read
        .option("inferSchema","true")
        .option("header", "true")
        .option("delimiter", ",")
        .csv("resources/winequality-red.csv")
        .cache()

      miDF = miDF.na.drop("any")

      val featureCols = Array("fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol")

      //set the input and output column names
      val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
      //return a dataframe with all of the  feature columns in  a vector column
      val df2 = assembler.transform(miDF)
      // the transform method produced a new column: features.
      df2.show
      //  Create a label column with the StringIndexer
      val labelIndexer = new StringIndexer().setInputCol("quality").setOutputCol("label")
      val df3 = labelIndexer.fit(df2).transform(df2)
      // the  transform method produced a new column: label.
      df3.show
      //  split the dataframe into training and test data
      val splitSeed = 2210
      val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), splitSeed)

      val lr = new LinearRegression()
        .setMaxIter(10)
        .setRegParam(0.3)
        .setElasticNetParam(0.8)

      // Fit the model
      val lrModel = lr.fit(trainingData)
      // Print the coefficients and intercept for linear regression
      println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

      val predictions = lrModel.transform(testData)
      predictions.show(false)

      // Summarize the model over the training set and print out some metrics
      val trainingSummary = lrModel.summary
      println(s"numIterations: ${trainingSummary.totalIterations}")
      println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
      trainingSummary.residuals.show()
      println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
      println(s"r2: ${trainingSummary.r2}")
    }

  }




