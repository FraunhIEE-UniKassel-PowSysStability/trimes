# trimes

trimes (**tr**ansient t**ime** **s**eries) is a python package for transient time series data in pandas format. The application is actually for all time series data where the time vector has a numerical format (e.g numpy's float64) - as opposed to the frequently used *DateTime* format. To the best of our knowledge, there is currently no other python package focusing on transient time series data as described and the *DateTime* format is not very convenient for transient time series.

trimes provides a thin wrapper for pandas *DataFrames* (in the format mentioned above) with helper functions to get data points, for interpolation, resampling, regression etc.

The package is at an early stage. Please have a look at the 'getting_started' tutorial.

## Installation

```shell
pip install trimes
```
