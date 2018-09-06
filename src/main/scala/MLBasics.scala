import java.io.File

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

object MLBasics {

  // Some string constants for the different columns in our data set
  val unparsedFeaturesColumn = "raw_features"
  val featuresColumn = "features"
  val labelColumn = "label"
  val predictionColumn = "predicted_" + labelColumn

  // Define a schema for the csv file of data we'll be importing
  val gunshotDataSchema = StructType(
    StructField("filename", StringType) ::
    StructField(unparsedFeaturesColumn, StringType) ::
    StructField(labelColumn, DoubleType) ::
    Nil
  )

  def main(args: Array[String]): Unit = {
    // Create our spark application session. Swap out the master to run on a real Spark cluster.
    val spark = SparkSession.builder
      .appName("Gunshot Detection")
      .master("local")
      .getOrCreate()

    // Create our source DataFrame from the data/gunshot-data.csv file
    println("Loading data from: " + getPathToData())
    val df = spark.read
      .option("header", true)
      .schema(gunshotDataSchema)
      .csv(getPathToData())
    previewData("Source Data", df)

    // Create custom Transformer for PipelineStage that will convert our string of doubles into a Spark Vector
    val tokenizer = new DoubleTokenizer()
      .setInputCol(unparsedFeaturesColumn)
      .setOutputCol(featuresColumn)
    // This will give a sample of what the transformed data looks like. However it also means that you'll
    // be transforming the data for the preview, and then again in the pipeline
     previewData("Transformed", tokenizer.transform(df))

    // Create our Estimator PipelineStage that will use the trainingData to build an algorithm and then predict
    // a value for the testData in the predicted_label column
    val lr = new LogisticRegression()
      .setLabelCol(labelColumn)
      .setFeaturesCol(featuresColumn)
      .setPredictionCol(predictionColumn)

    // Spark ML lib uses Pipelines to organize scalable data pipelines. Here we're defining
    // the order of the stages in our pipeline.
    val stages = Array[PipelineStage](
      tokenizer,
      lr
    )

    // Randomly split the source data into training and test data (80% training, 20% test)
    val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))

    // Construct the Pipeline from the defined stages. We then produce a PipelineModel by
    // "fit"ing the Pipeline to the training data
    val pipeline = new Pipeline().setStages(stages)
    val model = pipeline.fit(trainingData)

    // Now we use the PipelineModel to transform our test data adding a predicted label
    val predicted = model.transform(testData)
    previewData("Predictions", predicted)

    // We build up a BinaryClassificationEvaluator that can evaluate our predictions and compare
    // them to provided labels. The can compute two metrics of quality: `areaUnderROC` or `areaUnderPR`
    // ROC stands for "Receiver Operating Characteristic": https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    // PR stands for "Precision and Recall": https://en.wikipedia.org/wiki/Precision_and_recall
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol(labelColumn)
      .setRawPredictionCol(predictionColumn)
      .setMetricName("areaUnderPR")

    // Compute the evaluator metric across our predicted data
    val auc = evaluator.evaluate(predicted)
    println("Area Under Curve: " + auc)

    // That's all folks
    spark.stop()
  }

  def previewData(prefix: String, df: DataFrame) = {
    println(prefix + " Preview:")
    df.printSchema()
    df.show(5)
  }

  def getPathToData() = {
    // To run from a different directory, set "resourcePath" to the path to your files. Something like:
    // val resourcePath = "/Users/someone/foo/data"
    val resourcePath = new File(System.getProperty("user.dir"), "data").getPath

    val largeFile = new File(resourcePath, "gunshot-data-large.csv")
    val smallFile = new File(resourcePath, "gunshot-data.csv")
    if (largeFile.exists && largeFile.length() > 500) {
      largeFile.getPath
    } else if (smallFile.exists) {
      smallFile.getPath
    } else {
      throw new Exception("Unable to find the data file. Looking in: " + resourcePath)
    }
  }
}
