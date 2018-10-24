# Titanic Survivors KNN

Analysis of the Mathematical Model for KNN Algorithm and predict the survival of passengers on Titanic using the same with Python.

**Introduction**

*1. Abstract:*
The KNN algorithm is a robust and versatile classifier that is often used as a benchmark
for more complex classifiers such as Artificial Neural Networks (ANN) and Support Vector
Machines (SVM). Despite its simplicity, KNN can outperform more powerful classifiers and
is used in a variety of applications such as economic forecasting, data compression and
genetics. For example, KNN was leveraged in a 2006 study of functional genomics for the
assignment of genes based on their expression profiles.

*What is KNN?*

Let’s first start by establishing some definitions and notations. We will use x to denote a
feature (aka. predictor, attribute) and y to denote the target (aka. label, class) we are
trying to predict.
KNN falls in the supervised learning family of algorithms. Informally, this means that we
are given a labelled dataset consisting of training observations (x,y) and would like to
capture the relationship between x and y. More formally, our goal is to learn a function
h:X→Y so that given an unseen observation x, h(x) can confidently predict the
corresponding output y.

*Non-parametric* means it makes no explicit assumptions about the functional form of h,
avoiding the dangers of mismodeling the underlying distribution of the data. For example,
suppose our data is highly non-Gaussian but the learning model we choose assumes a
Gaussian form. In that case, our algorithm would make extremely poor predictions.

*Instance-based learning* means that our algorithm doesn’t explicitly learn a model.
Instead, it chooses to memorize the training instances which are subsequently used as
“knowledge” for the prediction phase. Concretely, this means that only when a query to
our database is made (i.e. when we ask it to predict a label given an input), will the
algorithm use the training instances to spit out an answer.
KNN is non-parametric, instance-based and used in a supervised learning setting.
It is worth noting that the minimal training phase of KNN comes both at a memory cost,
since we must store a potentially huge data set, as well as a computational cost during
test time since classifying a given observation requires a rundown of the whole data set.
Practically speaking, this is undesirable since we usually want fast responses.


**2 Algorithm and Pseudocode:**
*Generic for KNN:*
1. Calculate “d(x, xi)” i =1, 2, ….., n; where d denotes the Euclidean distance between
the points.
2. Arrange the calculated n Euclidean distances in non-decreasing order.
3. Let k be a +ve integer, take the first k distances from this sorted list.
4. Find those k-points corresponding to these k-distances.
5. Let ki denotes the number of points belonging to the ith class among k points i.e. k ≥ 0
6. If ki >kj ∀ i ≠ j then put x in class i.

*For Titanic Problem:*
1. begin
2. initialize the n×n distance matrix D , initialize the Ω×Ω confusion matrix C , set t←0
, TotAcc←0 , and set NumIterations equal to the desired number of iterations (repartitions).
3. calculate distances between all the input samples and store in n×n matrix D . (For a
large number of samples, use only the lower or upper triangular of D for storage
since it is a square symmetric matrix.)
for t← 1 to Num Iterations do
set C←0, and ntotal←0 .
partition the input samples into κ equally-sized groups.
for fold← 1 to κ do
assign samples in the fourth partition to testing, and use the remaining
samples for training. Set the number of samples used for testing as test .
set ntotal←ntotal+ntest.
for i ← 1 to test do
for test sample xi determine the k closest training samples based on the
calculated distances. determine ω^ , the most frequent class label among the k
closest training samples.
increment confusion matrix C by 1 in element cω,ω^ , where ω is the true
and ω^ the predicted class label for test sample xi . If ω=ω^ then the increment of +1
will occur on the diagonal of the confusion matrix, otherwise, the increment will
occur in an off-diagonal.
determine the classification accuracy using Acc=∑Ωjcjjntotal where cjj is a
diagonal element of the confusion matrix C .
calculate TotAcc=TotAcc+Acc .
calculate AvgAcc=TotAcc/NumIterations
4. end


 **3. Conclusions**
 
*Advantages of K-nearest neighbors algorithm:*

• KNN is simple to implement.

• KNN executes quickly for small training data sets.

• performance asymptotically approaches the performance of the Bayes Classifier.

• Don’t need any prior knowledge about the structure of data in the training set.

• No retraining is required if the new training pattern is added to the existing
training set.

*Limitation to K-nearest neighbors algorithm:*

• When the training set is large, it may take a lot of space.

• For every test data, the distance should be computed between test data and all
the training data. Thus a lot of time may be needed for the testing.

**4. References:**

• http://dataaspirant.com/2016/12/23/k-nearest-neighborclassifier-intro/

• http://www.scholarpedia.org/article/K-nearest_neighbor

• https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2243774/?page=1

• https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wiscon
sin+(Original)

• http://me.seekingqed.com/files/intro_KNN.pdf

• https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/

• https://machinelearningmastery.com/tutorial-to-implement-k-nearestneighbors-in-python-from-scratch/
