package se.kth.spark.lab1.task3

import org.apache.spark._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.PipelineModel

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    val filePath = "src/main/resources/millionsong.txt"

    // first we need to read the data into RDD to be able to split the file 80/20 (training/testing)
    val RDD = sc.textFile(filePath)
    println("Total data set length: " + RDD.count())
    val splitRDDs = RDD.randomSplit(Array(0.8, 0.2), seed = 23)

    val trainingRDD = splitRDDs(0) // 80% of the dataset
    val trainingDF = trainingRDD.toDF()

    val testingRDD = splitRDDs(1)  // 20% of the dataset
    val testingDF = testingRDD.toDF()

    println("Training set length: " + trainingDF.count())
    printf("Testing set length: " + testingDF.count())

    val obsDF: DataFrame = ???

    val myLR = ???
    val lrStage = ???
    val pipeline = ???
    val pipelineModel: PipelineModel = ???
    val lrModel = pipelineModel.stages(lrStage).asInstanceOf[LinearRegressionModel]

    //print rmse of our model
    //do prediction - print first k
  }
}