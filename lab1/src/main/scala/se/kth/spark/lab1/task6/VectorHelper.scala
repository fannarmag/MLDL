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
    ???
  }

  def sum(v1: Vector, v2: Vector): Vector = {
    ???
  }

  def fill(size: Int, fillVal: Double): Vector = {
    // create a vector of predefined size and initialize it with the predefined value
    Vectors.dense(Array.fill[Double](size)(fillVal))
  }
}