# 0.2.8
- Fixed bug on gradient calculation of Logistic Regression.
- Fixed to change accessor of params of estimators to read only.
- Added parameter validation.

# 0.2.7
- Fixed to support multiclass classifiction into LinearSVC, LogisticRegression, KernelSVC, and FactorizationMachineClassifier

# 0.2.6
- Added class for Decision Tree classifier.
- Added class for Random Forest classifier.
- Fixed to use frozen string literal.
- Refactored marshal dump method on some classes.
- Introduced Coveralls to confirm test coverage.

# 0.2.5
- Added classes for Naive Bayes classifier.
- Fixed decision function method on Logistic Regression class.
- Fixed method visibility on RBF kernel approximation class.

# 0.2.4
- Added class for Factorization Machine classifier.
- Added classes for evaluation measures.
- Fixed the method for prediction of class probability in Logistic Regression.

# 0.2.3
- Added class for cross validation.
- Added specs for base modules.
- Fixed validation of the number of splits when a negative label is given.

# 0.2.2
- Added data splitter classes for K-fold cross validation.

# 0.2.1
- Added class for K-nearest neighbors classifier.

# 0.2.0
- Migrated the linear algebra library to Numo::NArray.
- Added module for loading and saving libsvm format file.

# 0.1.3
- Added class for Kernel Support Vector Machine with Pegasos algorithm.
- Added module for calculating pairwise kernel fuctions and euclidean distances.

# 0.1.2
- Added the function learning a model with bias term to the PegasosSVC and LogisticRegression classes.
- Rewrited the document with yard notation.

# 0.1.1
- Added class for Logistic Regression with SGD optimization.
- Fixed some mistakes on the document.

# 0.1.0
- Added basic classes.
- Added an utility module.
- Added class for RBF kernel approximation.
- Added class for Support Vector Machine with Pegasos alogrithm.
- Added class that performs mutlclass classification with one-vs.-rest strategy.
- Added classes for preprocessing such as min-max scaling, standardization, and L2 normalization.
