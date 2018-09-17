import java.io.File
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}

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
    val source = spark.read
      .option("header", true)
      .schema(gunshotDataSchema)
      .csv(getPathToData())

    // Balance the source data to avoid improper skew destroying the models ability to fit
    val balancedSource = balanceDataset(source)
    previewData("Balanced Source Data", balancedSource)


    // Create custom Transformer for PipelineStage that will convert our string of doubles into a Spark Vector
    val tokenizer = new DoubleTokenizer()
      .setInputCol(unparsedFeaturesColumn)
      .setOutputCol(featuresColumn)
    // This will give a sample of what the transformed data looks like. However it also means that you'll
    // be transforming the data for the preview, and then again in the pipeline
     previewData("Transformed", tokenizer.transform(balancedSource))

    // Create multilayer perception classifier
    val classifier = new MultilayerPerceptronClassifier()
      .setLabelCol(labelColumn)
      .setFeaturesCol(featuresColumn)
      .setPredictionCol(predictionColumn)
      .setSeed(1234L)

    // Build a param grid appropriate for a MultilayerPerceptionClassifier
    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.layers, Array[Array[Int]](
        Array[Int](270, 100, 10, 2),
        Array[Int](270, 1000, 2),
        Array[Int](270, 300, 100, 5, 2)
      ))
      .addGrid(classifier.blockSize, Array[Int](1, 128, 256, 512))
      .addGrid(classifier.maxIter, Array[Int](10, 100, 200, 300))
      .build()


    // Spark ML lib uses Pipelines to organize scalable data pipelines. Here we're defining
    // the order of the stages in our pipeline. Then we construct the Pipeline.
    val stages = Array[PipelineStage](
      tokenizer,
      classifier
    )
    val pipeline = new Pipeline().setStages(stages)

    // Randomly split the source data into training and test data (80% training, 20% test)
    val Array(trainingData, testData) = balancedSource.randomSplit(Array(0.8, 0.2))

    // We build up a BinaryClassificationEvaluator that can evaluate our predictions and compare
    // them to provided labels. The can compute two metrics of quality: `areaUnderROC` or `areaUnderPR`
    // ROC stands for "Receiver Operating Characteristic": https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    // PR stands for "Precision and Recall": https://en.wikipedia.org/wiki/Precision_and_recall
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol(labelColumn)
      .setRawPredictionCol(predictionColumn)
      .setMetricName("areaUnderPR")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3) // The number of partitions in the data to use for searching. Generally use 3 or more.
      .setParallelism(2) // Evaluate up to 2 parameter settings in parallel
    val model = cv.fit(trainingData)
    val predicted = model.transform(testData)
    previewData("Predictions", predicted)
    val auc = evaluator.evaluate(predicted)
    println("Area Under Curve: " + auc)

    // We'll go ahead and perform the simple pipeline fit so we don't need to adjust the ParamGrid for each classifier
    val simpleModel = pipeline.fit(trainingData)
    val simplePredictions = simpleModel.transform(testData)
    println("Simple Area Under Curve: " + evaluator.evaluate(simplePredictions))

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

  def balanceDataset(df:DataFrame) = {
    // Find the gunshot samples and the non-gunshot samples
    val gunshotDf = df.filter("label=1")
    val nonGunshotDf = df.filter("label=0")

    println("Number of Gunshots: " + gunshotDf.count())
    println("Total Samples: " + df.count())

    val sampleRatio = gunshotDf.count().toDouble / df.count().toDouble
    println("Ratio of Gunshots to non-Gunshots: " + sampleRatio)

    // Randomly choose non gunshot entries using our sample ratio so we come out with roughly the same number
    // of non-gunshot samples as gunshots
    val nonGunshotSampleDf = nonGunshotDf.sample(false, sampleRatio)

    // Finally join the gunshot samples with the non-gunshot samples to produce our balanced set
    gunshotDf.union(nonGunshotSampleDf)
  }
}
