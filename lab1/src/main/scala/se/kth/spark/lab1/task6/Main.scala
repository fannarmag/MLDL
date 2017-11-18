package se.kth.spark.lab1.task6

import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import se.kth.spark.lab1.task2

object Main {
  def main(args: Array[String]) {

   // Note: Task 3 code duplicated here and changed a bit

    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    // Get prepared data from task 2
    val RDD = task2.Main.prepareData(sqlContext)
    println("Total data set length: " + RDD.count())

    // Split data 80/20 (training/testing)
    val splitRDDs = RDD.randomSplit(Array(0.8, 0.2), seed = 23)
    val trainingRDD = splitRDDs(0) // 80% of the dataset
    val trainingDF = trainingRDD.toDF()
    val testingRDD = splitRDDs(1)  // 20% of the dataset
    val testingDF = testingRDD.toDF()
    println("Training set length: " + trainingDF.count())
    println("Testing set length: " + testingDF.count())

    // Create linear regression estimator pipeline and model
    val myLR = getLinearRegression
    val pipelineStages = Array(myLR)
    val lrStage = pipelineStages.length - 1
    val pipeline = new Pipeline().setStages(pipelineStages)
    val pipelineModel: PipelineModel = pipeline.fit(trainingDF)
    val lrModel = pipelineModel.stages(lrStage).asInstanceOf[MyLinearModelImpl]

    //print rmse of our model
    println("LinearRegressionTrainingSummary:")
    println(s"RMSE: ${lrModel.trainingError(lrModel.trainingError.length - 1)}")

    //do prediction - print first k
    val modelProcessedDF = pipelineModel.transform(testingDF)
    println("Task 6 predictions - modelProcessedDF:")
    modelProcessedDF.show(10)
  }

  def getLinearRegression : MyLinearRegressionImpl = {
    new MyLinearRegressionImpl()
      .setLabelCol("label")
      .setFeaturesCol("features")
  }
}