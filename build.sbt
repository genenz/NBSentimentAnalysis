name := "NBSentimentAnalysis"

version := "1.0"

scalaVersion := "2.11.7"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.6.0"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.6.0"

libraryDependencies += "com.databricks" % "spark-csv_2.11" % "1.3.0"

libraryDependencies += "com.github.tototoshi" %% "scala-csv" % "1.2.2"