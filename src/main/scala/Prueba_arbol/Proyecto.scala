package Prueba_arbol

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Level
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.functions._



object Proyecto {
  implicit def bool2int(b:Boolean) = if (b) 1 else 0
  def main(args: Array[String]): Unit = {
    //Creando el contexto del Servidor
    val sc = new SparkContext("local", "Proyecto", System.getenv("SPARK_HOME"))
    sc.setLogLevel("ERROR")
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Proyecto")
      .getOrCreate()
    // Load the data stored in csv format as a DataFrame.
    var data = spark.read.format("csv").option("header", "true").option("inferSchema","true").option("delimiter",",").load("resources/neo.csv").toDF()
    // data.withColumn("label", when (col("label") === "true", "1").otherwise(col("label")))
    data = data.withColumn("label", col("label").cast("String"))
    data.withColumn("label", when (col("label") === "true", "1").otherwise(col("label")))
    data.show(5)

    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // Automatically identify categorical features, and index them.
    val featureIndexer = new VectorIndexer()
      .setInputCol("id")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a DecisionTree model.
    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(3)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${1.0 - accuracy}")

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
    Thread.sleep(200000);
  }
}