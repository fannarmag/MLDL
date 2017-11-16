package se.kth.spark.lab1.task3

import org.apache.spark._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import se.kth.spark.lab1.task2

object Main {
  def main(args: Array[String]) {
    //val conf = new SparkConf().setAppName("lab1").setMaster("local")
    //val sc = new SparkContext(conf)
    //val sqlContext = new SQLContext(sc)

    // Let's call task 2 as a function
    // Get the pipeline stages defined there, as well as the spark and SQL contexts
    val (sc, sqlContext, task2PipelineStages) = task2.Main.main(Array())

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val obsDF: DataFrame = sqlContext.read.text(filePath)

    val myLR = new LinearRegression()
      .setLabelCol("label(yearShifted)")
      .setFeaturesCol("f3f")
      .setMaxIter(10)
      .setRegParam(0.1)
      .setElasticNetParam(0.1)
    // TODO What is this?
    //val lrStage = ???
    val pipelineStages = task2PipelineStages :+ myLR
    val pipeline = new Pipeline().setStages(pipelineStages)
    val pipelineModel: PipelineModel = pipeline.fit(obsDF)
    // TODO What is this?
    //val lrModel = pipelineModel.stages(lrStage).asInstanceOf[LinearRegressionModel]

    val modelProcessedDF = pipelineModel.transform(obsDF)
    print("Task 3 predictions - modelProcessedDF:")
    modelProcessedDF.show(10)

    //print rmse of our model
    //do prediction - print first k
  }
}