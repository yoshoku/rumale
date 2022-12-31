# frozen_string_literal: true

require 'rumale/dataset'
require 'rumale/pairwise_metric'
require 'rumale/evaluation_measure/f_score'
require 'rumale/evaluation_measure/log_loss'
require 'rumale/evaluation_measure/mean_absolute_error'
require 'rumale/kernel_approximation/rbf'
require 'rumale/kernel_machine/kernel_svc'
require 'rumale/linear_model/svc'
require 'rumale/linear_model/svr'
require 'rumale/linear_model/logistic_regression'
require 'rumale/pipeline/pipeline'
require 'rumale/preprocessing/min_max_scaler'
require 'rumale/tree/decision_tree_regressor'
require 'rumale/naive_bayes/gaussian_nb'

require 'simplecov'

SimpleCov.start do
  add_filter %w[spec vendor]
end

require 'rumale/model_selection'

def two_clusters_dataset
  centers = Numo::DFloat[[-10, 0], [10, 0]]
  Rumale::Dataset.make_blobs(200, centers: centers, cluster_std: 0.5, random_seed: 1)
end

def three_clusters_dataset
  centers = Numo::DFloat[[0, 5], [-5, -5], [5, -5]]
  Rumale::Dataset.make_blobs(300, centers: centers, cluster_std: 0.5, random_seed: 1)
end

def xor_dataset
  x_a, y_a = Rumale::Dataset.make_blobs(200, centers: Numo::DFloat[[-5, 5], [5,  5]], cluster_std: 0.6, random_seed: 1)
  x_b, y_b = Rumale::Dataset.make_blobs(200, centers: Numo::DFloat[[5, -5], [-5, -5]], cluster_std: 0.6, random_seed: 1)
  x = Numo::DFloat.vstack([x_a, x_b])
  y = 2 * y_a.concatenate(y_b) - 1
  [x, y]
end

RSpec.configure do |config|
  # Enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = '.rspec_status'

  # Disable RSpec exposing methods globally on `Module` and `main`
  config.disable_monkey_patching!

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end
end
