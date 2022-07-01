package Linear

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

object Zari {
  def main(args: Array[String]): Unit = {
    //Reducir el número de LOG
    Logger.getLogger("org").setLevel(Level.OFF)
    //Creando el contexto del Servidor
    val sc = new SparkContext("local", "possum", System.getenv("SPARK_HOME"))
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
      .csv("resources/possumlimpito.csv")
      .cache()

    miDF = miDF.na.drop("any")
    miDF.show()

    val features = new VectorAssembler()
      .setInputCols(Array("site","Pop","sex","hdlngth","skullw","totlngth","taill","footlgth","earconch","eye","chest","belly"))
      .setOutputCol("features")

    val lr = new LinearRegression() // 1.6658, 0.2359
      .setMaxIter(5)
      .setRegParam(0.01)
      .setElasticNetParam(0.01)

    val pipeline = new Pipeline().setStages(Array(features, lr))

    //val data1 = miDF.drop("footlgth")
    val data2 = miDF.withColumn("footlgth", col("footlgth").cast("double"))
    val df1 = data2.withColumn("label", col("age").cast("int"))

    val model = pipeline.fit(df1)

    val lrModel = model.stages(1).asInstanceOf[LinearRegressionModel]
    println(s"RMSE:   ${lrModel.summary.rootMeanSquaredError}")
    println(s"r2:     ${lrModel.summary.r2}")
  }
}


