from env import gdb
import os
import pandas as pd


# def wrangle_zillow():
#     '''
#     The wrangle_zillow() function returns a DataFrame:
#     --------------------------------------------------

#     1. First the file is either unpickled locally,

#         - or -

#         a SQL query is run to pull the information from
#         the codeup MySQL database. An env.py file is necessary.
#         and the result is pickled for caching

#     The DataFrame has df.shape: (2_985_217, 7)
    
#     2. From there the approximately: 
#         135_K rows with null or 0 data is dropped 

#     The returned DataFrame has df.shape: (2_855_303, 7)

#     '''
#     df = get_zillow()

#     # For now... we're just hacking and slashing with one command
#     df = df.dropna()

#     # NULL drop
    
#     # There's still zeros in theh beds and baths columns, that's not right.
#     no_beds_baths_index = df[(df.beds == 0) | (df.baths == 0)].index
    
#     df = df.drop(index= no_beds_baths_index)

#     return df




    # '''Wrangles data from Zillow Database'''

##################################################Wrangle.py###################################################

# import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# from env import user, password, host

#**************************************************Acquire*******************************************************

def get_zillow():
    '''
    This is basically the aqcuire function for the zillow data:
    -----------------------------------------------------------

    A DataFrame is returned with df. shape: (2_985_217, 7)

    The query used on the zillow schema on the codeup MySQL database:

    SELECT 	bedroomcnt AS beds, 
		bathroomcnt AS baths, 
		calculatedfinishedsquarefeet AS sqft, 
        taxvaluedollarcnt AS tax_appraisal, 
        yearbuilt AS yr_built,
        taxamount AS taxes,
        fips
        FROM properties_2017

        WHERE propertylandusetypeid IN (261, 279);
        
    If pickled and available locally as filename: 'zillow_data':

        The DataFrame is pulled from there instead.

    else:

        This will pickle the DataFrame and store it locally for next time
    '''
    # Set the filename for caching
    filename= 'zillow_data'
    
    # if the file is available locally, read it
    if os.path.isfile(filename):
        df = pd.read_pickle(filename)
        
    else:
        # Read the SQL query into a DataFrame
        df = gdb('zillow','''
        SELECT 	bedroomcnt beds, 
		bathroomcnt baths, 
		calculatedfinishedsquarefeet sqft, 
        taxvaluedollarcnt tax_appraisal, 
        yearbuilt yr_built,
        taxamount taxes,
        fips
        FROM properties_2017

        WHERE propertylandusetypeid IN (261, 279);
        ''')
    
    # Pickle the DataFrame for caching (pickling is much faster than using .csv)
    df.to_pickle(filename)
    return df

#**************************************************Remove Outliers*******************************************************

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#**************************************************Distributions*******************************************************

def get_hist(df):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df.columns if col not in ['fips', 'yr_built']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()
        
        
def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = ['beds', 'baths', 'sqft', 'tax_appraisal', 'taxes']

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()
        
#**************************************************Prepare*******************************************************

def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''

    # removing outliers
    df = remove_outliers(df, 1.5, ['beds', 'baths', 'sqft', 'tax_appraisal', 'taxes'])
    
    # get distributions of numeric data
    get_hist(df)
    get_box(df)

    # converting column datatypes
    df.fips = df.fips.astype(object)
    df.fips = df.fips.map({6037.0: 'Los Angeles County',
                6059.0: 'Orange County',
                6111.0: 'Ventura County'
               })
    df.yr_built = df.yr_built.astype(object)    

    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # impute year built using mode
    imputer = SimpleImputer(strategy='median')

    imputer.fit(train[['yr_built']])

    train[['yr_built']] = imputer.transform(train[['yr_built']])
    validate[['yr_built']] = imputer.transform(validate[['yr_built']])
    test[['yr_built']] = imputer.transform(test[['yr_built']])       

    train.yr_built = train.yr_built.astype(int).astype(object)
    validate.yr_built = validate.yr_built.astype(int).astype(object)
    test.yr_built = test.yr_built.astype(int).astype(object)

    return train, validate, test    


#**************************************************Wrangle*******************************************************


def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(get_zillow())
    
    return train, validate, test

#**************************************************SCALE**********************************************************


def scale_zillow(train, validate, test, return_scaler=False):
    '''
    This is a Docstring: If you\'re reading this, I still need to write more );
    '''
    
    # set the coloumns for scaling
    # yr_built and fips are objects and don't need to be scaled
    # tax_appraisal is the target and doesn't need to be scaled
    cols = ['beds', 'baths', 'sqft', 'taxes']

    # sort the columns for pairity
    cols = sorted(cols)
    
    scaler_minmax = MinMaxScaler()
    
    minmax_cols = []

    for col in train[cols]:
        mimax_cols.append(f'{col}_minmax')
    
    
    train[minmax_cols] = scaler_minmax.fit_transform(train[cols])
    
    validate[minmax_cols] = scaler_minmax.transform(validate[cols])
    
    test[minmax_cols] = scaler_minmax.transform(test[cols])
    
    train, validate, test = train[sorted(train)], validate[sorted(validate)], test[sorted(test)]
    
    if return_scaler:
        return train, validate, test, scaler_minmax
    else:
        return train, validate, test
    

def imports(phase='start'):
    '''
    <!--- NOTE: DocString best read in md --->
    **Prints `imports` for different phases of the Data Science Pipeline!**

    # PARAMETERS
    ___
    __phase__ : set to start if you don't provide a phase
    |phase|description|
    |---|---|
    |`'start'`|This imports everything you need to get started from env.gdb to numpy, pandas, matplotlib, seaborn, stats...|
    |`''`|
    '''

