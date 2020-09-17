# Metrics of probabilistic accuracy

## Table of contents

1. [Accuracy and error metrics](#accuracy-and-error-metrics)
1. [Probabilistic forecasts](#probabilistic-forecasts)
1. [Probabilistic reference](#probabilistic-reference)
1. [References](#references)

## Accuracy and error metrics

To our knowledge, there is no standard metric for accuracy.
However, there are some standard metrics for what can be considered to be its opposite: error.
By default, we give back the Mean Absolute Error (MAE),
the Mean Absolute Percentage Error (MAPE)
and the Weighted Absolute Percentage Error (WAPE).
Each of these metrics is a representation of how wrong a belief is (believed to be),
with its convenience depending on use case.
For example, for intermittent demand time series (i.e. sparse data with lots of zero values) MAPE is not a useful metric.
For an intuitive representation of accuracy that works in many cases, we suggest to use:

    >>> df["accuracy"] = 1 - df["wape"]

With this definition:

- 100% accuracy denotes that all values are correct
- 50% accuracy denotes that, on average, the values are wrong by half of the reference value
- 0% accuracy denotes that, on average, the values are wrong by exactly the reference value (i.e. zeros or twice the reference value)
- negative accuracy denotes that, on average, the values are off-the-chart wrong (by more than the reference value itself)

## Probabilistic forecasts

The previous metrics (MAE, MAPE and WAPE) are technically not defined for probabilistic beliefs.
However, there is a straightforward generalisation of MAE called the Continuous Ranked Probability Score (CRPS), which is used instead.
The other metrics follow by dividing over the deterministic reference value.
For simplicity in usage of the `timely-beliefs` package,
the metrics names in the BeliefsDataFrame are the same regardless of whether the beliefs are deterministic or probabilistic.

## Probabilistic reference

It is possible that the reference itself is a probabilistic belief rather than a deterministic belief.
Our implementation of CRPS handles this case, too, by calculating the distance between the cumulative distribution functions of each forecast and reference [(Hans Hersbach, 2000)](#references).
As the denominator for calculating MAPE and WAPE, we use the expected value of the probabilistic reference.

## References

- Hans Hersbach. [Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems](https://journals.ametsoc.org/doi/pdf/10.1175/1520-0434%282000%29015%3C0559%3ADOTCRP%3E2.0.CO%3B2) in Weather and Forecasting, Volume 15, No. 5, pages 559-570, 2000.
