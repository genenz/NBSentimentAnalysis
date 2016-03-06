/**
  * Created by gene on 3/3/16.
  */

import java.io.File
import com.github.tototoshi.csv.CSVReader
import com.github.tototoshi.csv.CSVWriter
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.mllib.classification.{NaiveBayes,NaiveBayesModel}
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector

object NBSentimentAnalysis {

  // Application Specific Objects
  val hashingTF = new HashingTF()

  // Sentiment Constants
  private val POSITIVE = 1.0
  private val NEGATIVE = -1.0
  private val NEUTRAL = 0.0

  def main(args: Array[String]): Unit = {
    // Create SparkContext and SparkStreamingContext
    val conf = new SparkConf().setMaster("local[*]").setAppName("NBSentimentAnalysis")
    val sc = new SparkContext(conf)
    val sqlc = new SQLContext(sc)

    // Preprocess the dataset
    val dataSource = this.getClass.getResource("/training")
    prepocessTrainingSet(dataSource.getFile)

    // Let's put everything into a Dataframe just for ease
    val sentimentSourceEdit = getClass.getResource("/training_edited")
    val rawDF = sqlc.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(sentimentSourceEdit.getFile)
    val cleansedDF = cleanDF(rawDF).select("text", "sentiment")


    // Pull out the training sets
    val positiveSentimentSet = cleansedDF.filter("sentiment = 1.0").limit(700)
    val negtiveSentimentSet = cleansedDF.filter("sentiment = -1.0").limit(700)
    val neutalSentimentSet = cleansedDF.filter("sentiment = 0.0").limit(700)

    val positiveSplits = positiveSentimentSet.randomSplit(Array(0.8, 0.2), seed = 11L)
    val negativeSplits = negtiveSentimentSet.randomSplit(Array(0.8, 0.2), seed = 11L)
    val neutralSplits = neutalSentimentSet.randomSplit(Array(0.8, 0.2), seed = 11L)

    val finalTrainingSetDF = positiveSplits(0).unionAll(negativeSplits(0)).unionAll(neutralSplits(0))
    val finalTestingSetDF = positiveSplits(1).unionAll(negativeSplits(1)).unionAll(neutralSplits(1))
    val testingRDD = prepareDFToRDD(finalTestingSetDF)
    val trainingRDD = prepareDFToRDD(finalTrainingSetDF)

    // Train the model
    val model = trainModel(trainingRDD)

    // Test Model
    val outputOfPrediction = testingRDD.map({ case (label, text) => (label, model.predict(tokenizeAndHashString(text))) })

    println("Model Accuracy: " + 100 * outputOfPrediction.filter(x => x._1 == x._2).count() / outputOfPrediction.count() + "%")
    val outputText = outputOfPrediction.map(x => {
      if (x._1 != x._2)
        (0, x._1, x._2)
      else
        (1, x._1, x._2)
    })
    /*val combinedTextWithPrediction = outputText.zip(testingRDD.values)
    combinedTextWithPrediction.foreach({ case (success, text) => {
      if (success._1 == 0)
        println("Orig Text: " + text + ", Actual: " + success._2 + ", Prediction: " + success._3)
    }
    })*/
  }

  def prepareDFToRDD(thisDF: DataFrame): RDD[(Double, String)] = {
    thisDF.map(item => (item(1).toString.toDouble, item(0).toString))
  }

  def trainModel(thisRDD: RDD[(Double, String)]): NaiveBayesModel = {
    val labeledPointRDD = thisRDD.map({ case (label, features) =>
      val tfVector = tokenizeAndHashString(features)
      LabeledPoint(label, tfVector)
    })
    // Train the Model
    NaiveBayes.train(labeledPointRDD, 1.0, "multinomial")
  }

  def tokenizeAndHashString(thisString: String): Vector = {
    val features = tokenize(thisString)
    hashingTF.transform(features)
  }

  def tokenize(line: String): Seq[String] = {
    //line.toLowerCase.replaceAll("""[\p{Punct}]""", " ").replaceAll(" +", " ").trim.split(" ")
    line.split(" ")
  }

  def cleanDF(df: DataFrame): DataFrame = {
    df.filter("id IS NOT NULL").dropDuplicates(Array("text")) // Remove all NULL items
  }

  // Method to cleanse the dataset to remove '\n' characters in the text field.
  // The '\n' breaks the spark-csv library
  def prepocessTrainingSet(csvFileLocation: String): Unit = {
    val origCSV = CSVReader.open(new File(csvFileLocation))
    val editedCSV = ListBuffer[ListBuffer[String]]()

    for (line <- origCSV.iterator) {
      val returnLine = ListBuffer[String]()
      for (i <- 0 to (line.length - 1)) {
        if (line(i).contains("\n"))
          returnLine += line(i).replaceAll("\n", " ")
        else if (line(i).contains("\r"))
          returnLine += line(i).replaceAll("\r", " ")
        else
          returnLine += line(i)
      }
      editedCSV += returnLine
    }
    origCSV.close()

    // Write edited file back to disk
    val outCSV = CSVWriter.open(new File(csvFileLocation + "_edited"))
    outCSV.writeAll(editedCSV)
    outCSV.close()

  }
}