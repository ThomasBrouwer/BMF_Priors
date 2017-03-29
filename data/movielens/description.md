Folder containing the MovieLens datasets.

Each dataset provides ratings of 0 (no rating) or 1-5, with increments of 1.

There are several versions of varying size:
- MovieLens 100K: 100,000 ratings from 1000 users on 1700 movies, in u.data (each line of format "user_id\tmovie_id\trating\ttimestamp\n").
- MovieLens 1M: 1 million ratings from 6000 users on 4000 movies, in movies.dat (each line of format "user_id::movie_id::rating::timestamp\n").

There are also the MovieLens 10M and 20M datasets, but we stick to the smaller
ones for our experiments.

The raw data files are stores in /100K/ and /1M/.
These files can be processed and loaded in using the methods in load_data.py.

Source:
- MovieLens 100K: https://grouplens.org/datasets/movielens/100k/
- MovieLens 1M: https://grouplens.org/datasets/movielens/1m/
- MovieLens 10M: https://grouplens.org/datasets/movielens/10m/
- MovieLens 20M: https://grouplens.org/datasets/movielens/20m/