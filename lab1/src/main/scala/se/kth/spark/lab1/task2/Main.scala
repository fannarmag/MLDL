package se.kth.spark.lab1.task2

import se.kth.spark.lab1._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Column, Row, SQLContext}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.functions.{min, udf}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sqlContext.read.text(filePath)
    rawDF.show(2)


    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("value")
      .setOutputCol("tokenArray")
      .setPattern(",")

    //Step2: transform with tokenizer and show 5 rows
    val regexTokenizedDF = regexTokenizer.transform(rawDF)
    regexTokenizedDF.show(2)

    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector()
      .setInputCol("tokenArray")
      .setOutputCol("tokenVector")
      .transform(regexTokenizedDF)
    arr2Vect.show(2)

    //Step4: extract the label(year) into a new column
    val getYear = udf { (token: DenseVector) =>
        Vectors.dense(token(0)) // keep the year in vector - convert it to Double in Step5
    }
    val lSlicer = arr2Vect.withColumn("year", getYear(arr2Vect.col("tokenVector")))
    lSlicer.show(10)

    //Step5: convert type of the label from vector to double (use our Vector2Double)
    val v2d = new Vector2DoubleUDF((vect: Vector) => { vect(0) })
      .setInputCol("year")
      .setOutputCol("yearDouble")
      .transform(lSlicer)
    v2d.show(2)

    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF)
    val minYearValue = v2d.select(min("yearDouble")).collect()(0).getDouble(0)
    val lShifter = new DoubleUDF((y: Double) => { y - minYearValue })
      .setInputCol("yearDouble")
      .setOutputCol("label (yearShifted)")
      .transform(v2d)
    lShifter.show(2)
    //val minYearValueTest = lShifter.select(min("yearShifted")).collect()(0).getDouble(0)

    //Step7: extract just the 3 first features in a new vector column
    val getFirst3Features = udf { (tokenVect: DenseVector) =>
      Vectors.dense(tokenVect(1), tokenVect(2), tokenVect(3))
    }
    val fSlicer = lShifter.withColumn("f3f", getFirst3Features(lShifter.col("tokenVector")))
    fSlicer.show(2)

    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(???)

    //Step9: generate model by fitting the rawDf into the pipeline
    val pipelineModel = pipeline.fit(rawDF)

    //Step10: transform data with the model - do predictions
    ???

    //Step11: drop all columns from the dataframe other than label and features
    ???
  }
}