Folder containing the Jester joke rating datasets.

Average ratings for 100 jokes, from 73421 users. Each rating is in the range
[-10,10] (continuous), with 99 meaning "no rating".

The ratings are spread across three spreadsheets:
- jester-data-1.zip : Data from 24,983 users who have rated 36 or more jokes, a matrix with dimensions 24983 X 101.
- jester-data-2.zip : Data from 23,500 users who have rated 36 or more jokes, a matrix with dimensions 23500 X 101.
- jester-data-3.zip : Data from 24,938 users who have rated between 15 and 35 jokes, a matrix with dimensions 24,938 X 101.

These give .xls files, which we then converted to .csv. First column gives the
number of jokes rated by this user.

For the experiments, we shift all values by 10 to be positive, and then cast
them as integers.

Source:
http://www.ieor.berkeley.edu/~goldberg/jester-data/
https://grouplens.org/datasets/jester/