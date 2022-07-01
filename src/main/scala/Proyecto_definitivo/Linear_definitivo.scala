package Proyecto_definitivo

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

object Linear_definitivo {
  def main(args: Array[String]): Unit = {
    //Reducir el número de LOG
    Logger.getLogger("org").setLevel(Level.OFF)
    //Creando el contexto del Servidor
    val sc = new SparkContext("local", "winequality", System.getenv("SPARK_HOME"))
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("CargaJSON")
      .config("log4j.rootCategory", "ERROR, console")
      .getOrCreate()

    var miDF = spark.read
      .option("inferSchema", "true")
      .option("header", "true")
      .option("delimiter", ",")
      .csv("resources/winequality-red.csv")
      .cache()

    miDF = miDF.na.drop("any")
    miDF.show()

    val features = new VectorAssembler()
      //
      .setInputCols(Array("fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"))
      .setOutputCol("features")

    val lr = new LinearRegression() //0.6456, 0.3605
      .setMaxIter(50)
      .setRegParam(0.01) //0.3
      .setElasticNetParam(0.01) //0.8

    val pipeline = new Pipeline().setStages(Array(features, lr))

    val df1 = miDF.withColumn("label", col("quality").cast("int"))

    val model = pipeline.fit(df1)

    val lrModel = model.stages(1).asInstanceOf[LinearRegressionModel]
    println(s"RMSE:   ${lrModel.summary.rootMeanSquaredError}")
    println(s"r2:     ${lrModel.summary.r2}")
  }
}
