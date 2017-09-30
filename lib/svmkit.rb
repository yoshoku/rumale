begin
  require 'nmatrix/nmatrix'
rescue LoadError
end

require 'svmkit/version'
require 'svmkit/utils'
require 'svmkit/base/base_estimator'
require 'svmkit/base/classifier'
require 'svmkit/base/transformer'
require 'svmkit/kernel_approximation/rbf'
require 'svmkit/linear_model/pegasos_svc'
require 'svmkit/multiclass/one_vs_rest_classifier'
require 'svmkit/preprocessing/l2_normalizer'
require 'svmkit/preprocessing/min_max_scaler'
require 'svmkit/preprocessing/standard_scaler'
