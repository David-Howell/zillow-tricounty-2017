# Zillow Tri-County ML for Tax Assesed Values - 2017
___

![California with FIPS](https://raw.githubusercontent.com/David-Howell/zillow-tricounty-2017/main/california_fips_codes.png "California with FIPS")

|**Index:**|  |  |
|---|---|---|
| |**Links:**|**Description:**|
|---|---|---|
| |[__PLANNING__](#PLANNING)| The Planning stages explained|
| |[__MVP__](#MVP)| The MVP steps and process - DUE 2022_09_08_12:00:00|
| |[__ITERATION 1__](#ITTER_1)
<br><br>

<a id="PLANNING"></a>

<div class="alert alert-block alert-info">
</div>

***


## Planning:<br>

<a id="MVP"></a>

<div class="alert alert-block alert-info">
</div>

***

### MVP by Thursday @ Noon 2022_09_08<br>

1. Gather Data (ETL):<br>
  - Write SQL Query JOIN `properties_2017`, `predictions_2017` <br>ON `parcelid` WHERE `transactiondate` >= `2017-01-01` AND < `2018-01-01`
  - Use only the `propertylandusetype`s `261` and `279`
  - Use only square feet of the home, number of bedrooms, and number of bathrooms <br>to estimate the property's assessed value, `taxvaluedollarcnt`.
  - Use only houses with non zero non null bathrooms, bedrooms, and squarefeet
  - Define MVP functions for ETL and keep in `wrangle_mvp.py` 
<br>
2. Clean Data:<br>
  - Tranform fields
  - Clear outliers
  - Define MVP functions for cleaning the data and keep in `wrangle_mvp.py`
<br>
3. Split Data:<br>
  - Train(60%), Validate(20%), and Test(20%) = Total Clean Data (100%)
  - Use random seed for repeatability
  - Define MVP functions for spliting the data 
<br>
4. Exploratory Data Analysis (EDA):<br>
  - Univariate:<br>
    - Get counts
    - Check for normality
    - Decide what is categorical and what is continuous
    - Define MVP functions for plots, classifying and scaling data and keep in `eda_mvp.py`
  - Bivariate:<br>
    - Look for trends, maybe split categories further
  - Make some Hypotheses about the data:<br>
    - Test using statistical tests (chi^2 and/or t-tests) to reject null or alternative hypotheses
    - Define MVP functions to repeat statistical testing and keep in `ead_mvp.py`
<br>
5. Modeling:<br>
  - Establish Baseline:<br>
    - Test RMSE for Mean, Median, Mode(rounded)
  - Train Regression models
  - Test Regression models
  - Define MVP functions to repeat modeling for testing hyperparameters keep in `model_mvp.py`
<br>
6. MVP Conclusions:<br>
  - Hopefully we're already better than baseline!
  - Explain results and insights so far
  - Theorize on possible features that could be added for further analysis
  - Set prediction Goals for next stage(s)

<a id="ITTER_1"></a>

<div class="alert alert-block alert-info">
</div>

***
### Iteration 1:<br>

1. Repaet the above process, but with MORE!!
  - Already thinking about, zips, lat/long, block codes... Get back to the DataBasics
  