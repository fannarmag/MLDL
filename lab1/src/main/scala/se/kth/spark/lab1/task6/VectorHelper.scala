package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.ml.linalg.Vectors

object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
    var sum = 0.0
    if (v1.size == v2.size) {
      sum = v1.toArray.zip(v2.toArray).map({case(a, b) => a*b}).sum
    }
    sum
  }

  def dot(v: Vector, s: Double): Vector = {
    val a = v.toArray
    Vectors.dense(a.map(x => x*s))
  }

  def sum(v1: Vector, v2: Vector): Vector = {
    // Implement addition of two vectors
    // Assuming vectors have the same dimensions (requirement)
    val a1 = v1.toArray
    val a2 = v2.toArray
    val zipped = a1.zip(a2)
    val summed = zipped.map(tup => tup._1 + tup._2)
    Vectors.dense(summed)
  }

  def fill(size: Int, fillVal: Double): Vector = {
    // Create a vector of predefined size and initialize it with the predefined value
    Vectors.dense(Array.fill[Double](size)(fillVal))
  }
}