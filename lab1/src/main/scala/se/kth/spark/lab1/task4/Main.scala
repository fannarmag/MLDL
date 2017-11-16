package se.kth.spark.lab1.task4

import org.apache.spark._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.tuning.{CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{PipelineModel, PipelineStage}
import se.kth.spark.lab1.task3

object Main {
  def main(args: Array[String]) {

    // Let's call task 3 as a function
    // Get the pipeline stages defined there, as well as the spark and SQL contexts
    val (sc: SparkContext, sqlContext: SQLContext, task3PipelineStages: Array[PipelineStage]) = task3.Main.main(Array())

    //build the parameter grid by setting the values for maxIter and regParam
    // this grid will have 2 x 2 = 4 parameter settings for CrossValidator to choose from.
    val lrStage: LinearRegression = task3PipelineStages(task3PipelineStages.length - 1).asInstanceOf[LinearRegression]
    val paramGrid = new ParamGridBuilder()
      .addGrid(lrStage.maxIter, Array(10, 50))
      .addGrid(lrStage.regParam, Array(0.1, 0.9))
      .build()

    val evaluator = new RegressionEvaluator()
  }
}