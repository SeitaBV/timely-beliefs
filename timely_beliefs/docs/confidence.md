# Keeping track of confidence

To keep track of the confidence of a data point, timely-beliefs works with probability distributions.
More specifically, the BeliefsDataFrame contains points of interest on the cumulative distribution function (CDF),
and leaves it to the user to set an interpolation policy between those points.
This allows you to describe both discrete possible event values (as a probability mass function) and continuous possible event values (as a probability density function).
A point of interest on the CDF is described by the `cumulative_probability` index level (ranging between 0 and 1) and the `event_value` column (a possible value).

The default interpolation policy is to interpret the CDF points as discrete possible event values,
leading to a non-decreasing step function as the CDF.
In case an event value with a cumulative probability of 1 is missing, the last step is extended to 1 (i.e. the chance of an event value that is greater than largest available event value is taken to be 0).

A deterministic belief consists of a single row in the BeliefsDataFrame.
Regardless of the cumulative probability actually listed (we take 0.5 by default),
the default interpolation policy will interpret the single CDF point as an event value stated with 100% certainty. 
The reason why we choose a default cumulative probability of 0.5 instead of 1 is that, in our experience, sources more commonly intend to report their expected value rather than an event value with absolute confidence.

A probabilistic belief consists of multiple rows in the BeliefsDataFrame,
with a shared `event_start`, `belief_time` and `source`, but different `cumulative_probability` values.
_For a future release we are considering adding interpolation policies to interpret the CDF points as describing a normal distribution or a (piecewise) uniform distribution,
to offer out-of-the-box support for resampling continuous probabilistic beliefs._
