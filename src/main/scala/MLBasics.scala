import java.io.File

import org.apache.spark.sql.SparkSession

object MLBasics {

  def main(args: Array[String]): Unit = {
    // Create our spark application session. Swap out the master to run on a real Spark cluster.
    val spark = SparkSession.builder
      .appName("Gunshot Detection")
      .master("local")
      .getOrCreate()

    println("App running, this would use data from: " + getPathToData())

    // That's all folks
    spark.stop()
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
