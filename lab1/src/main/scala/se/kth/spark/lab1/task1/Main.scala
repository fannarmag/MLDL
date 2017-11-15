package se.kth.spark.lab1.task1

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._

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
    songsDf.show()
    
    // Register the DataFrame as a SQL temporary view
    songsDf.createOrReplaceTempView("songs")
    
    
    // Answer both by using higher order functions and Spark SQL 
		// 1. How many songs there are in the DataFrame?
		val numberOfSongs1Rdd = songsRdd.count
		println("numberOfSongs1RDD " + numberOfSongs1Rdd)
		val numberOfSongs1Df = sqlContext.sql("SELECT COUNT(*) FROM songs").collect()(0).getLong(0)
		println("numberOfSongs1Df " + numberOfSongs1Df)
		
    
    //2. How many songs were released between the years 1998 and 2000?
		val numberOfSongs2RDD = songsRdd.map(song => song.year).filter(year => year > 1997 && year < 2001).count()
		println("numberOfSongs2RDD " + numberOfSongs2RDD)
		val numberOfSongs2Df = songsDf.select("year").filter($"year" > 1997 && $"year" < 2001).count()
		println("numberOfSongs2Df " + numberOfSongs2Df)
		
		//3. What is the min, max and mean value of the year column?
		val minYearRDD = songsRdd.map(song => song.year).min()
		println("minYearRDD " + minYearRDD)
		val maxYearRDD = songsRdd.map(song => song.year).max()
		println("maxYearRDD " + maxYearRDD)
		val meanYearRDD = songsRdd.map(song => song.year).mean()
		println("meanYearRDD " + meanYearRDD)
		
		val minYearDf = songsDf.select(min("year")).collect()(0).getDouble(0)
		println("minYearDf " + minYearDf)
		val maxYearDf = songsDf.select(max("year")).collect()(0).getDouble(0)
		println("maxYearDf " + maxYearDf)
		val meanYearDf = songsDf.select(mean("year")).collect()(0).getDouble(0)
		println("meanYearDf " + meanYearDf)
		
		//4. Show the number of songs per year between the years 2000 and 2010?
		println("numberOfSongsPerYearRdd:")
		val numberOfSongsPerYearRdd = songsRdd.map(song => song.year)
		                              .filter(year => year > 1999 && year < 2011)
		                              .countByValue()
		                              .foreach({case(year, count) => println(year, count)})
		println("numberOfSongsPerYearDf")
		val numberOfSongsPerYearDf = songsDf
		                             .filter($"year" > 1999 && $"year" < 2011)
		                             .groupBy("year")
		                             .count()
		                             .show()
  }
 
}