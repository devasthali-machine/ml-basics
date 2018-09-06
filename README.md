# Intro to Functional Programming in Machine Learning

## Prerequisites
If you're on a Mac `brew install scala` or follow the standard [Scala install instructions.](https://www.scala-lang.org/download/)

## Running the example
To run the application simply: `sbt run`

## Repository Layout
The [`master`](https://github.com/boushley/ml-basics/tree/master) or [`1-basic-app`](https://github.com/boushley/ml-basics/tree/1-basic-app) branch is the start. This branch contains the basic skeleton of a Scala CLI application
with the basic data intact.

Next, [`2-linear-regression`](https://github.com/boushley/ml-basics/tree/2-logistic-regression) shows a basic Spark ML pipeline using linear regression to train a gunshot detection model.

[`3-sample-balancing`](https://github.com/boushley/ml-basics/tree/3-sample-balancing) improves on the basic linear regression application by refining the input data so its more
balanced.

The [`4-cross-validator`](https://github.com/boushley/ml-basics/tree/4-cross-validator) branch further improves by adding a step that runs the training pipeline multiple times to find
the best parameters for our linear regression estimator.

Further experimentation on the [`5-model-samples`](https://github.com/boushley/ml-basics/tree/5-model-samples) branch, which adds examples of multiple alternative classifiers.
