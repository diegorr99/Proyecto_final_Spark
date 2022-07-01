package Prueba_regresion

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Level
import org.apache.spark.sql
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}


object Regression {
  /**
   *  Missing values show up as a dot.  The dot function below
   *  returns -1 if there is a dot or blank value. And it converts strings to double.
   * Then later we will
   *  delete all rows that have any -1 values.
   *
  def dot (s: String) : Double = {
    if (s.contains(".") || s.length == 0) {
      -1
    } else {
      s.toDouble
    }
  }
  **
   *  We are going to use a Dataframe.  It requires a schema.
   * So we create that below.  We use the same column names
   *  as are in the .dat file.
   *
  val schema = StructType (
    StructField("STR", DoubleType, true) ::
      StructField("OBS", DoubleType, true) ::
      StructField("AGMT", DoubleType, true) ::
      StructField("FNDX", DoubleType, true) ::
      StructField("HIGD", DoubleType, true) ::
      StructField("DEG",DoubleType, true) ::
      StructField("CHK", DoubleType, true) ::
      StructField("AGP1", DoubleType, true) ::
      StructField("AGMN", DoubleType, true) ::
      StructField("NLV", DoubleType, true) ::
      StructField("LIV", DoubleType, true) ::
      StructField("WT", DoubleType, true) ::
      StructField("AGLP", DoubleType, true) ::
      StructField("MST", DoubleType, true) ::  Nil)
  **
   *  Read in the .dat file and use the regular expression
   *  \s+ to split it by spaces into an RDD.
   */
  def main(args: Array[String]): Unit = {
    //Creando el contexto del Servidor
    val sc = new SparkContext("local","Regression", System.getenv("SPARK_HOME"))
    sc.setLogLevel("ERROR")
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Regression")
      .getOrCreate()

    var df = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .load("resources/winequality-red.csv")
    /**
    * val readingsRDD = spark.sparkContext.textFile("resources/winequality-red.csv")
    val RDD = readingsRDD.map(_.split("\\s+"))

     *   Run the dot function over every element in the RDD to convert them
     *   to doubles, since that if the format requires by the Spark ML LR model.
     *   Note that we skip the first one since that is just a blank space.
     *
    val rowRDD = RDD.map(s => Row(dot(s(1)),dot(s(2)),dot(s(3)),dot(s(4)),dot(s(5)),dot(s(6)),
      dot(s(7)),dot(s(8)),dot(s(9)),dot(s(10)),dot(s(11)),dot(s(12)),
      dot(s(13)),dot(s(14))))
     **
     * Now create a dataframe with the schema we described above,
     *
     *
    val readingsDF = spark.createDataFrame(rowRDD, schema)
    **
     *  Create a new dataframe dropping all of those with missing values.
     *
    var cleanDF = df.filter(df("STR") > -1 && df("OBS") > -1 && df("AGMT")  > -1  && df("FNDX") > -1 && df("HIGD") > -1  && df("DEG") > -1 && df("CHK") > -1 && df("AGP1") > -1  && df("AGMN") > -1  && df("NLV") > -1  && df("LIV") > -1 && df("WT") > -1 && df("AGLP") > -1 && df("MST") > -1)
    **
     *  Now comes something more complicated.  Our dataframe has the column headings
     *  we created with the schema.  But we need a column called “label” and one called
     * “features” to plug into the LR algorithm.  So we use the VectorAssembler() to do that.
     * Features is a Vector of doubles.  These are all the values like patient age, etc. that
     * we extracted above.  The label indicated whether the patient has cancer.
     */
    val featureCols = Array("fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(df)
    /**
     * Then we use the StringIndexer to take the column FNDX and make that the label.
     *  FNDX is the 1 or 0 indicator that shows whether the patient has cancer.
     * Like the VectorAssembler it will add another column to the dataframe.
     */
    val labelIndexer = new StringIndexer().setInputCol("quality").setOutputCol("label")
    val df3 = labelIndexer.fit(df2).transform(df2)
    /**
     *   Now we declare the LR model and run fit and transform to make predictions.
     */
    val model = new LogisticRegression().fit(df3)
    val predictions = model.transform(df3)
    /**
     *  Now we print it out.  Notice that the LR algorithm added a “prediction” column
     *  to our dataframe.   The prediction in almost all cases will be the same as the label.  That is
     * to be expected it there is a strong correlation between these values.  In other words
     * if the chance of getting cancer was not closely related to these variables then LR
     * was the wrong model to use.  The way to check that is to check the accuracy of the model.
     *  You could use the BinaryClassificationEvaluator Spark ML function to do that.
     * Adding that would be a good exercise for you, the reader.
     */
    predictions.select ("features", "label", "prediction").show()
  }

}




