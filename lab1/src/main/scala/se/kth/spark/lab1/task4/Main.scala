package se.kth.spark.lab1.task4

import org.apache.spark._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import se.kth.spark.lab1.task3

object Main {
  def main(args: Array[String]) {

    // Let's call task 3 as a function
    // Get the pipeline stages defined there, as well as the spark and SQL contexts
    val (sc, sqlContext, task3PipelineStages) = task3.Main.main(Array())

    // Running Cross-Validation for a number of features on a big dataset is very expensive
    // Here we are using the small dataset for time reasons
    // Normally we would use the big millionsong-500k-noquotes dataset
    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sqlContext.read.text(filePath)

    //create pipeline
    val pipeline = new Pipeline().setStages(task3PipelineStages)

    //build the parameter grid by setting the values for maxIter and regParam
    // this grid will have 2 x 2 = 4 parameter settings for CrossValidator to choose from.
    val lrStage: LinearRegression = task3PipelineStages(task3PipelineStages.length - 1).asInstanceOf[LinearRegression]
    val paramGrid = new ParamGridBuilder()
      // Three values above and three values below the base value
      // Let's take the median of the values given in the assignment as the base values, 30 and 0.5
      .addGrid(lrStage.maxIter, Array(10, 17, 23, 30, 37, 43, 50))
      .addGrid(lrStage.regParam, Array(0.1, 0.25, 0.38, 0.5, 0.65, 0.8, 0.9))
      .build()

    val evaluator = new RegressionEvaluator()
    //create the cross validator and set estimator, evaluator, paramGrid
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)

    // Run cross-validation, and choose the best set of parameters.
    println("Running cross-validation")
    val cvModel = cv.fit(rawDF)

    val bestModel = cvModel
      .bestModel
      .asInstanceOf[PipelineModel]
      .stages(task3PipelineStages.length - 1)
      .asInstanceOf[LinearRegressionModel]
    val bestModelSummary = bestModel.summary

    //print best model RMSE to compare to previous
    println("Best LinearRegressionModel:")
    println(s"RMSE: ${bestModelSummary.rootMeanSquaredError}")
    println(s"numIterations: ${bestModelSummary.totalIterations}")
    println(s"maxIterations: ${bestModel.getMaxIter}")
    println(s"regParam: ${bestModel.getRegParam}")
  }
}