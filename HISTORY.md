# 0.3.0
- Add class for Support Vector Regression.
- Add class for K-Nearest Neighbor Regression.
- Add class for evaluating coefficient of determination.
- Add class for evaluating mean squared error.
- Add class for evaluating mean absolute error.
- Fix to use min method instead of sort and first methods.
- Fix cross validation class to be able to use for regression problem.
- Fix some typos on document.
- Rename spec filename for Factorization Machine classifier.

# 0.2.9
- Add predict_proba method to SVC and KernelSVC.
- Add class for evaluating logarithmic loss.
- Add classes for Label- and One-Hot- encoding.
- Add some validator.
- Fix bug on training data score calculation of cross validation.
- Fix fit method of SVC for performance.
- Fix criterion calculation on Decision Tree for performance.
- Fix data structure of Decision Tree for performance.

# 0.2.8
- Fix bug on gradient calculation of Logistic Regression.
- Fix to change accessor of params of estimators to read only.
- Add parameter validation.

# 0.2.7
- Fix to support multiclass classifiction into LinearSVC, LogisticRegression, KernelSVC, and FactorizationMachineClassifier

# 0.2.6
- Add class for Decision Tree classifier.
- Add class for Random Forest classifier.
- Fix to use frozen string literal.
- Refactor marshal dump method on some classes.
- Introduce Coveralls to confirm test coverage.

# 0.2.5
- Add classes for Naive Bayes classifier.
- Fix decision function method on Logistic Regression class.
- Fix method visibility on RBF kernel approximation class.

# 0.2.4
- Add class for Factorization Machine classifier.
- Add classes for evaluation measures.
- Fix the method for prediction of class probability in Logistic Regression.

# 0.2.3
- Add class for cross validation.
- Add specs for base modules.
- Fix validation of the number of splits when a negative label is given.

# 0.2.2
- Add data splitter classes for K-fold cross validation.

# 0.2.1
- Add class for K-nearest neighbors classifier.

# 0.2.0
- Migrated the linear algebra library to Numo::NArray.
- Add module for loading and saving libsvm format file.

# 0.1.3
- Add class for Kernel Support Vector Machine with Pegasos algorithm.
- Add module for calculating pairwise kernel fuctions and euclidean distances.

# 0.1.2
- Add the function learning a model with bias term to the PegasosSVC and LogisticRegression classes.
- Rewrite the document with yard notation.

# 0.1.1
- Add class for Logistic Regression with SGD optimization.
- Fix some mistakes on the document.

# 0.1.0
- Add basic classes.
- Add an utility module.
- Add class for RBF kernel approximation.
- Add class for Support Vector Machine with Pegasos alogrithm.
- Add class that performs mutlclass classification with one-vs.-rest strategy.
- Add classes for preprocessing such as min-max scaling, standardization, and L2 normalization.
