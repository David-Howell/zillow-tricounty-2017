import os
import math
import pandas as pd
import numpy as np
from env import gdb
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import harmonic_mean

# modeling methods

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score

import warnings
warnings.filterwarnings("ignore")


'''
<  GET_ZILLOW_MVP  >
<  CLEAN_ZILLOW  >
<  SPLIT_DATA_CONTINUOUS  >
<  BOXPLOTS  >
<  REMOVE_OUTLIERS  >
<  HR (HUMAN READABLE)  >
<  HISTS  >
<  SLICER  >
<  SCALE_DATA  >
<  BASELINES  >
<  MODEL_SETS  >
<  MAKE_METRICS  >
<  MAKE_MODELS  >
<  GET_MODELS  >
'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  GET_ZILLOW_MVP  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def get_zillow_mvp():
    '''
    This is basically the aqcuire function for the zillow data:
    -----------------------------------------------------------

    A DataFrame is returned with df. shape: (52_279, 5)

    The query used on the zillow schema on the codeup MySQL database:

    SELECT  p.bedroomcnt beds, 
	    p.bathroomcnt baths, 
		p.calculatedfinishedsquarefeet area, 
        p.taxvaluedollarcnt tax_value,
        pred.transactiondate
        FROM properties_2017 p
        JOIN predictions_2017 pred
        ON p.parcelid = pred.parcelid
        WHERE propertylandusetypeid IN (261, 279)
        AND pred.transactiondate BETWEEN "2017-01-01" AND "2018-01-01"
        -- AND pred.transactiondate < 2018-01-01
        AND p.bedroomcnt > 0
        AND p.bathroomcnt > 0 
        AND p.calculatedfinishedsquarefeet > 0
        AND p.taxvaluedollarcnt > 0
        
    If pickled and available locally as filename: 'zillow_data_mvp':

        The DataFrame is pulled from there instead.

    else:

        This will pickle the DataFrame and store it locally for next time
    '''
    # Set the filename for caching
    filename= 'zillow_data_mvp'
    
    # if the file is available locally, read it
    if os.path.isfile(filename):
        df = pd.read_pickle(filename)
        
    else:
        # Read the SQL query into a DataFrame
        df = gdb('zillow','''
        SELECT  p.bedroomcnt beds, 
	    p.bathroomcnt baths, 
		p.calculatedfinishedsquarefeet area, 
        p.taxvaluedollarcnt tax_value,
        pred.transactiondate
        FROM properties_2017 p
        JOIN predictions_2017 pred
        ON p.parcelid = pred.parcelid
        WHERE propertylandusetypeid IN (261, 279)
        AND pred.transactiondate BETWEEN "2017-01-01" AND "2018-01-01"
        -- AND pred.transactiondate < 2018-01-01
        AND p.bedroomcnt > 0
        AND p.bathroomcnt > 0 
        AND p.calculatedfinishedsquarefeet > 0
        AND p.taxvaluedollarcnt > 0
        ''')

        df['date'] = pd.to_datetime(df.transactiondate)
        df = df.drop(columns='transactiondate')
    
    # Pickle the DataFrame for caching (pickling is much faster than using .csv)
    df.to_pickle(filename)
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  CLEAN_ZILLOW  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def clean_zillow():
    '''
    No parameters!
    ----------------

    This will output a clean dataframe of zillow information with outliers removed
    also, why model based on houses less than $100,000.00 or more than $1M 
    and we will reset the index, which will allow us to use the boxplots function for explore
    '''
# Use get_zillow_mvp
    df= get_zillow_mvp()
# Set the cols to everything but 'date'
    cols = [col for col in df.columns if col not in ['date']]
# Use remove_outliers
    df = remove_outliers(df, 1.5, cols)
# Limit the data to houses in the $100K - $1M range
    df = df[(df.tax_value <= 1_000_000) & (df.tax_value >= 100_000) & (df.area > 1000)]
# Set the cols to df
    df = df[cols]
# Reset the index
    df = df.reset_index()
    # drop the resulting 'index' column
    df = df.drop(columns='index')

    return df


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  SPLIT_DATA_CONTINUOUS  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def split_data_continuous(df, rand_st=123, with_baseline=False):
    '''
    Takes in: a pd.DataFrame()
          and a random state           ;if no random state is specifed defaults to [123]
          and a boolean value for with_baseline (=False) 
            if True: the baselines are computed for mean, median, mode, 
            the mean of those, the harmonic mean of those, and the harmonic mean of all predictions
          
      return: train, validate, test    ;subset dataframes
    '''
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=.2, 
                               random_state=rand_st)
    train, validate = train_test_split(train, test_size=.25, 
                 random_state=rand_st)
    print(f'Prepared df: {df.shape}')
    print()
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')
# here we add the if with_baselines to add all the possible baselines
    if with_baseline:
        baselines = ['mean_preds',
        'median_preds',
        'mode_preds',
        'm_mmm_preds',
        'hm_mmm_preds',
        'h_m_total_preds']
        # Use the get_baselines function
        train, validate, test = get_baselines(train, validate, test)
# Set the best basline RMSE to 1M so that it's above any reasonable baseline and baseline to none
        best_rmse = 1_000_000_000
        best_baseline = None
# test the RSME of each baseline and compare; sticking with only the lowest RSME
        for i in baselines:
            rmse_train = mean_squared_error(train.tax_value, train[i]) ** 0.5
            rmse_validate = mean_squared_error(validate.tax_value, validate[i]) ** 0.5

            if rmse_train < best_rmse:
                best_rmse = rmse_train
                best_baseline = i
                in_out = rmse_train/rmse_validate
# round the baseline values for human readability
        our_baseline = round(train[best_baseline].values[0])
# add to our dataframe
        train['baseline'] = our_baseline
# drop all the baselines we tested
        train = train.drop(columns= baselines)
# set the validate set with baseline and drop the others
        validate['baseline'] = our_baseline

        validate = validate.drop(columns= baselines)
# And the same for test
        test['baseline'] = our_baseline

        test = test.drop(columns= baselines)
            
        print(f'The {best_baseline} had the lowest RMSE: {round(best_rmse)} with an in/out of: {round(in_out,3)}')

    return train, validate, test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  BOXPLOTS  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def boxplots(df, excluding=False):
    '''
    make boxplots from all the columns in a dataframe, excluding anything you want to exclude
    '''
    # Set the cols for use in creating the boxplots
    cols = [col for col in df.columns if col not in [excluding]]
# set the figure and for loop to plot each column
    plt.figure(figsize=(16, 20))
    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[col])

        # Hide gridlines.
        plt.grid(False)

    plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  REMOVE_OUTLIERS  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def remove_outliers(df, k, col_list):
    ''' 
    Removes outliers from a list of columns in a dataframe 
    and return that dataframe
    
    PARAMETERS:
    ------------
    
    df    :   DataFrame that you want outliers removed from
    
    k     :   The scaler of IQR you want to use for tromming outliers
                 k = 1.5 gives a 8σ total range
    '''
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df['outlier'] = False
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    df = df[df.outlier == False]
    df = df.drop(columns=['outlier'])
    print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  HR (HUMAN READABLE)  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def hr(n, suffix='', places=2, prefix='$'):
    '''
    Return a human friendly approximation of n, using SI prefixes

    '''
    prefixes = ['','K','M','B','T']
    
    # if n <= 99_999:
    #     base, step, limit = 10, 4, 100
    # else:
    #     base, step, limit = 10, 3, 100

    base, step, limit = 10, 3, 100

    if n == 0:
        magnitude = 0 #cannot take log(0)
    else:
        magnitude = math.log(n, base)

    order = int(round(magnitude)) // step
    return '%s%.1f %s%s' % (prefix, float(n)/base**(order*step), \
    prefixes[order], suffix)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  HISTS  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def hists(df, exclude='', granularity=5): 

    '''
    Make histograms of all the columns in a dataframe, except any you want to exclude, and you can set the number of bins with granularity
    '''   
    # Set figure size. Went with 4x for the width:height to display 4 graphs... future version could have these set be the DataFrame columns used
    plt.figure(figsize=(16, 4))

    # List of columns
    cols = [col for col in df.columns if col not in [exclude]]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=granularity)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        # mitigate overlap
        plt.tight_layout()

    plt.suptitle(f'{hr(len(df),prefix="")} \
Houses in $ Range > {hr(df.tax_value.min())} - {hr(df.tax_value.max())} <',
                 y=1.05,
                 size=20
                )
    plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  heatmaps  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def heatmaps(train):
    # create the correlation matrix using pandas .corr()
    zill_corr = train.corr()

    # pass my correlation matrix to a heatmap
    kwargs = {'alpha':.9,
            'linewidth':3, 
            'linestyle':'-',
            'linecolor':'black'}

    sns.heatmap(zill_corr, cmap='Purples', annot=True,
            mask=np.triu(zill_corr), **kwargs)
    plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  SLICER  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def slicer(df, min=0, max=1_000_000, step=50_000):
    '''
    SLICER:
    ------------------------------
    Takes in a DataFrame and provides Histograms of each slice of tax_value
    the min max and step size of the bins can be set.
    also the standard deviation of each slice is output
    '''
    
    for i in range(min, max, step):
        price_range = 50_000
        houses = df[(i < df.tax_value) & (df.tax_value < i + price_range)]
        
    #     print(f'The standard deviation of houses between:\n\
    #     {i} and {i+price_range} is:\n ${round(houses.tax_value.std())}')
        
    #     print(houses.tax_value.describe())
        
        hists(houses, 'date')
        
        print(f'''
        σ = {round(houses.beds.std())} beds         |     \
    σ = {round(houses.baths.std())} baths      |     \
    σ = {round(houses.area.std())} sqft      |     \
    σ = {hr(houses.tax_value.std())} 
        ''')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  SCALE_DATA  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def scale_data(train, validate, test):
    '''
    scales the data using MinMaxScaler from SKlearn
    '''
#     Remember incoming columns and index numbers to output DataFrames
    cols = train.columns
    train_index = train.index
    validate_index = validate.index
    test_index = test.index
    
#     Make the scaler
    scaler = MinMaxScaler()
    
#     Use the scaler
    train = scaler.fit_transform(train)
    validate = scaler.transform(validate)
    test = scaler.transform(test)
    
#     Reset the transformed datasets into DataFrames
    train = pd.DataFrame(train, columns= cols, index= train_index)

    validate = pd.DataFrame(validate, columns= cols, index= validate_index)

    test = pd.DataFrame(test, columns= cols, index= test_index)
    
    return train, validate, test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  BASELINES  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def get_baselines(train, validate, test, y='tax_value'):
    '''
    Parameters:
    --------------------
    
    train           :       your training set
    validate        :       your validation set
    test            :       your test set
    y=  (tax_value) :       the target variable
    '''
    # Various methods for baseline predictions
    # We'll make new columns for each, and stick them in our training set

    train['mean_preds'] = \
    train[y].mean()

    train['median_preds'] = \
    train[y].median()

    train['mode_preds'] = \
    train[y].round(1).mode()[0]

    train['m_mmm_preds'] = \
    sum([train[y].mean(), train[y].median(), train[y].round(1).mode()[0]])/3

    train['hm_mmm_preds'] = \
    harmonic_mean([train[y].mean(), train[y].median(), train[y].round(1).mode()[0]])

    train['h_m_total_preds'] = \
    harmonic_mean(train[y])

    train_index = train.index.tolist()
    #  broke out the number ... damn, i need to rewrite all of this to use enumerate SMH
    one = train_index[0]

    baselines = ['mean_preds',
    'median_preds',
    'mode_preds',
    'm_mmm_preds',
    'hm_mmm_preds',
    'h_m_total_preds']

    for i in baselines:
        validate[i] = train[i][one]
        test[i] = train[i][one]
    
    return train, validate, test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  MODEL_SETS  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def model_sets(train, validate, test, scale_X_cols=True, target='tax_value'):
    '''
    Takes in the train, validate, and test sets and returns the X_train, y_train, etc. subsets

    Should look like: 
            
            X_train, y_train, x_validate, y_validate, X_test, y_test = model_sets(train, validate, test)

    Parameters:
    ------------------
    train     :  your split        train data
    
    validate  :  your split   validation data
    
    test      :  your split         test data

    scale_X_cols  :  (=True) this will invoke the scale_data function to scale the data using MinMaxScaler
                        False will skip the scaling and return the unscaled version

    target        :  (='tax_value') is your target variable, set to tax_value, cause that's what I was working on ;)


    Returns:
    ------------------
    X_train, y_train,
    X_validate, y_validate,
    X_test, y_test

    These can be used to train and evaluate model performance!

    '''

    # use forloop to get columns for X_cols exckuding the target and the baseline
    X_cols = []
    for i in train.columns:
        if i not in [target, 'baseline']:
            X_cols.append(i)
    y_cols = [target, 'baseline']

    # print what they are for the users reference
    print(f'\nX_cols = {X_cols}\n\ny_cols = {y_cols}\n\n')

    # set the X_ and y_ for train, validate and test
    X_train, y_train = train[X_cols], train[y_cols]
    X_validate, y_validate = validate[X_cols], validate[y_cols]
    X_test, y_test = test[X_cols], test[y_cols]

    # if scale_X_cols is true then we send all of our X_ columns trhough the scale_data function
    if scale_X_cols:
        X_train, X_validate, X_test = scale_data(X_train, X_validate, X_test)

    # 
    return X_train, y_train, X_validate, y_validate, X_test, y_test



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  MAKE_METRICS  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def make_metrics(zillow, target='tax_value'):
    '''
    takes in a list of DataFrames:
    -------------------
            zillow = [df, X_train, y_train, X_validate, y_validate, X_test, y_test]

    and a target variable
    '''

    # Make metrics
    rmse = mean_squared_error(zillow[2][target], zillow[2].baseline) ** (1/2)
    r2 = explained_variance_score(zillow[2][target], zillow[2].baseline)

    rmse_v = mean_squared_error(zillow[4][target], zillow[4].baseline) ** (1/2)
    r2_v = explained_variance_score(zillow[4].tax_value, zillow[4].baseline)
# Setup the metric dataframe
    metric_df = pd.DataFrame(data=[{
        'model': 'baseline',
        'rmse_train': hr(rmse),
        'r^2': r2,
        'rmse_validate': hr(rmse_v),
        'r^2_validate': r2_v}])

    return metric_df


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  MAKE_MODELS  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def make_models(zillow, target='tax_value'):
    '''
    Makes trained models from the X_train and y_train sets and evaluates them on validation sets

    models include:
                    Linear Regression
                    Lasso Lars
                    GLM (Tweedie Reg)

    Returns:
    -----------------
    a list of the X_ y_ dataframes, the models in a dataframe, and the metrics for the models in a dataframe
    zillow, models, metric_df

    '''
    
    # Name and make models
    models = pd.DataFrame(\
    {'model_name':['Linear Regression',
                'Lasso Lars',
                'GLM (Tweedie Reg)',
                ],
    'made_model': [LinearRegression(normalize=True),
                LassoLars(alpha=1, random_state=123),
                TweedieRegressor(power=(1), alpha=0)
                ],}
    )

    # Fit the models
    models['fit_model'] = models.model_name
    for i, j in enumerate(models.made_model):
        models['fit_model'][i] = j.fit(zillow[1], zillow[2].tax_value)

    # Make Model Predictors
    models['predict_model'] = models.model_name
    for i, j in enumerate(models.fit_model):
        models.predict_model[i] = j.predict

    # Make metrics_df
    metric_df = make_metrics(zillow)

    # Fill metrics_df with predictions
    for i, j in enumerate(models.predict_model):
        
    #     Make prediction: zillow[2] is y_train, [4] is y_validate, j is the .predict
        zillow[2][models.model_name[i]] = j(zillow[1])
        zillow[4][models.model_name[i]] = j(zillow[3])
        
    # Make metrics
            
        rmse = mean_squared_error(zillow[2][target], j(zillow[1])) ** (1/2)
        r2 = explained_variance_score(zillow[2][target], j(zillow[1])) 

        rmse_v = mean_squared_error(zillow[4][target], j(zillow[3])) ** (1/2)
        r2_v = explained_variance_score(zillow[4][target], j(zillow[3]))

        metric_df = metric_df.append([{
            'model': models.model_name[i],
                'rmse_train': hr(rmse),
                'r^2': r2,
                'rmse_validate': hr(rmse_v),
                'r^2_validate': r2_v}])

    return zillow, models, metric_df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  GET_MODELS  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def get_models():
    '''
    No Parameters!
    -----------------
    just run the function saving the three outputs

    outputs a list of the X_ y_ dataframes, the models in a dataframe, 
    and the metrics for the models in a dataframe

    zillow, models, metric_df 
    '''
    # grab the clean data
    df = clean_zillow()
    # split it
    train, validate, test = split_data_continuous(df, with_baseline=True)
    # get the model sets
    X_train, y_train, X_validate, y_validate, X_test, y_test = model_sets(train, validate, test)
    # make the list of sets to put into the models
    zillow = [df, X_train, y_train, X_validate, y_validate, X_test, y_test]
    # Use the make_models function
    zillow, models, metric_df = make_models(zillow)
    # output the results
    return zillow, models, metric_df


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  HAVERSINE  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


from math import radians, cos, sin, asin, sqrt

def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance between two points on the 
    earth (specified in decimal degrees), returns the distance in
    meters.
    All arguments must be of equal length.
    :param lon1: longitude of first place
    :param lat1: latitude of first place
    :param lon2: longitude of second place
    :param lat2: latitude of second place
    :return: distance in meters between the two sets of coordinates
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return c * r

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  evaluate  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def evaluate(y_validate):
    '''
    evaluate the models on the y_validate set 
    '''
    # y_validate.head()
    plt.figure(figsize=(16,8))
    plt.plot(y_validate.tax_value, y_validate.baseline, alpha=1, color="gray", label='_nolegend_')
    plt.annotate("Baseline: Predict Using Mean", (16, 9.5))
    plt.plot(y_validate.tax_value, y_validate.tax_value, alpha=.5, color="blue", label='_nolegend_')
    plt.annotate("The Ideal Line: Predicted = Actual", (.5, 3.5), rotation=15.5)

    plt.scatter(y_validate.tax_value, y_validate['Linear Regression'], 
                alpha=.2, color="red", s=100, label="Model: LinearRegression")
    plt.scatter(y_validate.tax_value, y_validate['GLM (Tweedie Reg)'], 
                alpha=.2, color="yellow", s=100, label="Model: TweedieRegressor")
    plt.scatter(y_validate.tax_value, y_validate['Lasso Lars'], 
                alpha=.2, color="green", s=100, label="Model: Lasso Lars")
    plt.legend()
    plt.xlabel("Actual Tax Value")
    plt.ylabel("Predicted Tax Value")
    plt.title("Where are predictions more extreme? More modest?")
    plt.annotate("The polynomial model appears to overreact to noise", (2.0, -10))
    plt.annotate("The OLS model (LinearRegression)\n appears to be most consistent", (15.5, 3))
    plt.show()