package se.kth.spark.lab1.task4

import org.apache.spark._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.{Pipeline, PipelineModel}
import se.kth.spark.lab1.task2
import se.kth.spark.lab1.task3

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val dataFrame = task2.Main.prepareData(sqlContext)
    val myLR = task3.Main.getLinearRegression

    //create pipeline
    val pipeline = new Pipeline().setStages(Array(myLR))

    //build the parameter grid by setting the values for maxIter and regParam
    // this grid will have 7 x 7 = 49 parameter settings for CrossValidator to choose from.
    val lrStage = myLR
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
    val cvModel = cv.fit(dataFrame)

    val bestModel = cvModel
      .bestModel
      .asInstanceOf[PipelineModel]
      .stages(0)
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