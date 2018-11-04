Due to a bug in naming (checkpointer uses a reset epoch when you compile your model several times), most models have epochs < 15. (First compile is 15 epochs, next ones 10/5 epochs).
By checking the creation date (modified date in unix though) you can determine what epoch offset is probably likely.

Creation times can also help estimating training time duration.
