package Proyecto_definitivo

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.IntegerType

object Proyecto_tree {

  def main(args: Array[String]): Unit = {
    //Reducir el número de LOG
    Logger.getLogger("org").setLevel(Level.OFF)
    //Creando el contexto del Servidor
    val sc = new SparkContext("local", "Proyecto_tree", System.getenv("SPARK_HOME"))
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Proyecto_tree")
      .config("log4j.rootCategory", "ERROR, console")
      .getOrCreate()
    var df = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .load("resources/neo.csv")

    var df2 = df.withColumn("label", col("label").cast(IntegerType))

    df2.printSchema()
    //"est_diameter_min","est_diameter_max", -> 0.8813
    //Con ,"absolute_magnitude" -> 0.8808
    //Con todas -> 0.8805
    //Sin ,"miss_distance" -> 0.8577
    //Con id -> 0.8820
    val inputColumns = Array("id", "est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance", "absolute_magnitude")
    val assembler = new VectorAssembler().setInputCols(inputColumns).setOutputCol("features")

    val featureSet = assembler.transform(df2)

    // split data random in trainingset (70%) and testset (30%) // 0.8855
    val seed = 5043
    val trainingAndTestSet = featureSet.randomSplit(Array[Double](0.85, 0.15), seed)
    val trainingSet = trainingAndTestSet(0)
    val testSet = trainingAndTestSet(1)

    // train the algorithm based on a Random Forest Classification Algorithm with default values// train the algorithm based on a Random Forest Classification Algorithm with default values

    val randomForestClassifier = new RandomForestClassifier().setSeed(seed)
    //randomForestClassifier.setMaxDepth(4)
    val model = randomForestClassifier.fit(trainingSet)
    // test the model against the test set       
    val predictions = model.transform(testSet)

    // evaluate the model
    val evaluator = new MulticlassClassificationEvaluator()

    System.out.println("accuracy: " + evaluator.evaluate(predictions))

  }

}
