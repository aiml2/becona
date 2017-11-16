#Steps for evaluating:

#1.0) for each of the CV split:
    # 1.1) Read in evaluation file list and ground truths (cached GT file)
    # 1.2.0)  Read in the filtered models relevant for this split. 
    # 1.3.0) For each of the models:
    # 1.3.1)  Batch predict the validation set of the CV split
    # 1.3.2)  Write file with the results: conf matrix, stat measures and which images are wrong/right 


