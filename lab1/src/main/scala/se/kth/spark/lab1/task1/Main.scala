package se.kth.spark.lab1.task1

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

object Main {
  case class Song(year: Double, f1: Double, f2: Double, f3: Double)
  
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    //val rawDF = sqlContext.read.text(filePath)
    val rdd = sc.textFile(filePath)

    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?
    // - The below line prints the first five rows of the text file.
    // - Each row is a record, fields are delimited by a comma and end-of-line is the delimiter between records
    // - There are 13 features, the first one is a string (e.g. 2001.0) the rest are double fields (e.g. 0.884123733793)
    // - (The format is CSV)
    rdd.take(5).foreach(println)

    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(line => line.split(","))

    //Step3: map each row into a Song object by using the year label and the first three features  
    // - Should catch Java NumberFormatException.. 
    val songsRdd = recordsRdd.map(record => Song(record(0).toDouble, record(1).toDouble, record(2).toDouble, record(3).toDouble))

    //Step4: convert your rdd into a dataframe
    val songsDf = songsRdd.toDF
    songsRdd.take(5).foreach(println)
    
    // TODO : Answer questions..
  }
 
}