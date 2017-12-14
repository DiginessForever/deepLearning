This dataset came from the people identified in the other readme.

I have hacked the hour.csv, copied all the rows from day.csv into it.

Day.csv does not have an hour column, while hour.csv did, so for all the rows I copied in,
I gave them an hour column and set them all to 12.  It wasn't important for what the
predictor was trying to predict - the total customers per day (at least I don't think it is
terribly important, or if it is, the predictor should be able to overcome that difficulty).
The important thing is that I needed the holiday data from the day.csv dataset.