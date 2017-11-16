package se.kth.spark.lab1.task3

import org.apache.spark._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import se.kth.spark.lab1.task2
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.PipelineModel

object Main {
  def main(args: Array[String]) {
    //val conf = new SparkConf().setAppName("lab1").setMaster("local")
    //val sc = new SparkContext(conf)
    //val sqlContext = new SQLContext(sc)

    // Let's call task 2 as a function
    // Get the pipeline stages defined there, as well as the spark and SQL contexts
    val (sc, sqlContext, task2PipelineStages) = task2.Main.main(Array())

    import sqlContext.implicits._

    val filePath = "src/main/resources/millionsong-500k-noquotes.txt"

    // first we need to read the data into RDD to be able to split the file 80/20 (training/testing)
    val RDD = sc.textFile(filePath)
    println("Total data set length: " + RDD.count())
    val splitRDDs = RDD.randomSplit(Array(0.8, 0.2), seed = 23)

    val trainingRDD = splitRDDs(0) // 80% of the dataset
    val trainingDF = trainingRDD.toDF()

    val testingRDD = splitRDDs(1)  // 20% of the dataset
    val testingDF = testingRDD.toDF()

    println("Training set length: " + trainingDF.count())
    println("Testing set length: " + testingDF.count())

    val myLR = new LinearRegression()
      .setLabelCol("label(yearShifted)")
      .setFeaturesCol("f3f")
      .setMaxIter(10)
      .setRegParam(0.1)
      .setElasticNetParam(0.1)
    val pipelineStages = task2PipelineStages :+ myLR
    val lrStage = pipelineStages.length - 1
    val pipeline = new Pipeline().setStages(pipelineStages)
    val pipelineModel: PipelineModel = pipeline.fit(trainingDF)
    val lrModel = pipelineModel.stages(lrStage).asInstanceOf[LinearRegressionModel]

    //print rmse of our model
    val lrTrainingSummary = lrModel.summary
    println(s"RMSE: ${lrTrainingSummary.rootMeanSquaredError}")

    //do prediction - print first k
    val modelProcessedDF = pipelineModel.transform(testingDF)
    println("Task 3 predictions - modelProcessedDF:")
    modelProcessedDF.show(10)
  }
}