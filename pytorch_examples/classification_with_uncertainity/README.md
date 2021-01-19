# Classification With Uncertain Labels

Let us consider a scenario where we have to classify something into 2 categories but the 
experiment is such that each time we sample n items together and the we only know how many of 
them were of the first kind and how many of the second. 

Thus we do not know the individual labels but only that of a batch (not to be confused with the 
mini-batch for training). The question is can we learn something.

In this repository I am conducing this study by taking images of cats and dogs and for each 
image I give randomly choose n-1 extra labels. The label of the image for training is then the 
average of all these. 