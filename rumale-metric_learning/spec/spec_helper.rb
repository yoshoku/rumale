# frozen_string_literal: true

require 'numo/linalg/autoloader'
require 'rumale/dataset'
require 'rumale/utils'
require 'rumale/nearest_neighbors/k_neighbors_classifier'
require 'rumale/nearest_neighbors/k_neighbors_regressor'

require 'simplecov'

SimpleCov.start do
  add_filter %w[spec vendor]
end

require 'rumale/metric_learning'

def three_clusters_dataset
  centers = Numo::DFloat[[0, 5], [-5, -5], [5, -5]]
  Rumale::Dataset.make_blobs(300, centers: centers, cluster_std: 0.5, random_seed: 1)
end

def regression_dataset(n_samples: 200, n_features: 8, n_informative: 4)
  rng = Random.new(1)
  x = Rumale::Utils.rand_normal([n_samples, n_features], rng)

  ground_truth = Numo::DFloat.zeros(n_features, 1)
  ground_truth[0...n_informative, true] = 100 * Rumale::Utils.rand_uniform([n_informative, 1], rng)
  y = x.dot(ground_truth)
  y = y.flatten

  rand_ids = Array(0...n_samples).shuffle(random: rng)
  x = x[rand_ids, true].dup
  y = y[rand_ids].dup

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
