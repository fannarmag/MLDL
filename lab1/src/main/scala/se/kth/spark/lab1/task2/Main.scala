package se.kth.spark.lab1.task2

import se.kth.spark.lab1._
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Column, Row, SQLContext}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer, VectorSlicer}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.functions.{min, udf}

object Main {
  def main(args: Array[String]) : (SparkContext, SQLContext, Array[PipelineStage]) = {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val filePath = "src/main/resources/millionsong-500k-noquotes.txt"
    val rawDF = sqlContext.read.text(filePath)
    println("rawDF size:" + rawDF.collect().length)
    rawDF.show(10)

    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("value")
      .setOutputCol("tokenArray")
      .setPattern(",")

    //Step2: transform with tokenizer and show 5 rows
    val processedDF = regexTokenizer.transform(rawDF)
    processedDF.show(10)

    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector()
      .setInputCol("tokenArray")
      .setOutputCol("tokenVector")

    val processedDF2 = arr2Vect.transform(processedDF)
    processedDF2.show(10)

    //Step4: extract the label(year) into a new column
    val lSlicer = new VectorSlicer()
      .setInputCol("tokenVector")
      .setOutputCol("year")
      .setIndices(Array(0))

    val processedDF3 = lSlicer.transform(processedDF2)
    processedDF3.show(10)

    //Step5: convert type of the label from vector to double (use our Vector2Double)
    val v2d = new Vector2DoubleUDF((vect: Vector) => { vect(0) })
      .setInputCol("year")
      .setOutputCol("yearDouble")

    val processedDF4 = v2d.transform(processedDF3)
    processedDF4.show(10)

    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF)
    // TODO Is it OK to find the min year here, outside of the lshifter stage? Yes? Pipeline creates a model, model is run on prediction data.
    val minYearValue = processedDF4.select(min("yearDouble")).collect()(0).getDouble(0)
    val lShifter = new DoubleUDF((y: Double) => { y - minYearValue })
      .setInputCol("yearDouble")
      .setOutputCol("label(yearShifted)")

    val processedDF5 = lShifter.transform(processedDF4)
    processedDF5.show(10)

    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer()
      .setInputCol("tokenVector")
      .setOutputCol("f3f")
      .setIndices(Array(1, 2, 3))

    val processedDF6 = fSlicer.transform(processedDF5)
    print("processedDF6:")
    processedDF6.show(10)

    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer))

    //Step9: generate model by fitting the rawDf into the pipeline
    val pipelineModel = pipeline.fit(rawDF)

    //Step10: transform data with the model
    val modelProcessedDF = pipelineModel.transform(rawDF)
    print("modelProcessedDF:")
    modelProcessedDF.show(10)

    //Step11: drop all columns from the dataframe other than label and features
    val modelProcessedDF2 = modelProcessedDF.drop("value").drop("tokenArray").drop("tokenVector").drop("year").drop("yearDouble")
    modelProcessedDF2.show(10)

    // Let's return the pipeline operators for use in later tasks
    (sc, sqlContext, Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer))
  }
}