# capstone
My Capstone project for the Nanodegree program
For this capstone project I stood on the shoulders of giants and used a few libraries and programs that were written before me.
In order to get the current date and in order to be able to pull data with a specific time frame, the datetime library was used, more specifically, the datetime, date and timedelta functions.

In addition, the urllib library was imported, this was used in order to get the most updated list of the S&P500 shares (instead of writing them manually), I used the library 
in conjugation with a function that I found online (link: http://www.thealgoengineer.com/2014/download_sp500_data/ ) that scapres the entire list from Wikipedia.
For the same reason BeautifulSoup was imported.

Next I also used the Python Data Analysis Library, in order to create the DataFrames for the features and the labels.

I also used Yahoo Finance's API in order to get the stock data for the required shares.

For performing all the machine learning and deeplearning algorithms, I used scikit-learn, Python's Machine Learning library. 

In order to run this fily, simply run capstone.py

The reason why I mention 41 days of data, however in the file itself I'm pulling 60 is because the 60 days are calendary days and not every day the stock exchange is open.
In order to pull more data it is suffice to change the delta variable.
if more columns are required (i.e. build the model for each trading day based on more than the 5 trading days currently used) the share_info variable would have to be changed,
specifically, the 1.2*delta would have to be larger, this is in order to pull a buffer of trading days as to complete the current 5 days of trading)
The train_features and test_features would also have to be changed, specifically the columns range. Raw 79, where I check if all the raws are full would also needs to be changed

I decided not to fix the data since I believe that the results should be reproducibe every trading day within a +-0.05 range  
