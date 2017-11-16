package se.kth.spark.lab1.task5

import org.apache.spark._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.PolynomialExpansion
import se.kth.spark.lab1.task2

object Main {
  def main(args: Array[String]) {
    //val conf = new SparkConf().setAppName("lab1").setMaster("local")
    //val sc = new SparkContext(conf)
    //val sqlContext = new SQLContext(sc)

    // Let's call task 2 as a function
    // Get the pipeline stages defined there, as well as the spark and SQL contexts
    val (sc, sqlContext, task2PipelineStages) = task2.Main.main(Array())

    //import sqlContext.implicits._
    //import sqlContext._

    val filePath = "src/main/resources/millionsong-500k-noquotes.txt"
    val obsDF: DataFrame = sqlContext.read.text(filePath)

    // Create polynomial expansion transformer
    val polynomialExpansionTransformer = new PolynomialExpansion().setInputCol("features").setOutputCol("polyfeatures").setDegree(2)

    // Create linear regression transformer to pipeline (duplicated from task 3)
    val myLR = new LinearRegression()
      .setLabelCol("label")
      .setFeaturesCol("polyfeatures")
      .setMaxIter(10)
      .setRegParam(0.1)
      .setElasticNetParam(0.1)

    // Add transformers to pipeline stages array
    val pipelineStages = task2PipelineStages ++ Array(polynomialExpansionTransformer, myLR)

    val pipeline = new Pipeline().setStages(pipelineStages)
    val pipelineModel: PipelineModel = pipeline.fit(obsDF)

    val lrStage = pipelineStages.length - 1
    val lrModel = pipelineModel.stages(lrStage).asInstanceOf[LinearRegressionModel]

    val lrTrainingSummary = lrModel.summary
    println("LinearRegressionTrainingSummary:")
    println(s"RMSE: ${lrTrainingSummary.rootMeanSquaredError}")
    println(s"numIterations: ${lrTrainingSummary.totalIterations}")

    //do prediction - print first k
    val modelProcessedDF = pipelineModel.transform(obsDF)
    println("Task 3 predictions - modelProcessedDF:")
    modelProcessedDF.show(10)
  }
}