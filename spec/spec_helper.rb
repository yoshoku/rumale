# frozen_string_literal: true

require 'simplecov'

SimpleCov.start

require 'bundler/setup'
require 'numo/linalg/autoloader'
require 'rumale'
require 'parallel'
require 'mmh3'

if defined?(GC.verify_compaction_references) == 'method'
  # Calling this method will help to find object movement bugs.
  GC.verify_compaction_references(double_heap: true, toward: :empty)
end

def two_clusters_dataset
  rng = Random.new(8)
  x_a = (2 * Rumale::Utils.rand_uniform([100, 2], rng) - 1) + Numo::DFloat[-2, 0]
  y_a = Numo::Int32.zeros(100)
  x_b = (2 * Rumale::Utils.rand_uniform([100, 2], rng) - 1) + Numo::DFloat[2, 0]
  y_b = Numo::Int32.zeros(100) + 1
  x = Numo::DFloat.vstack([x_a, x_b])
  y = y_a.concatenate(y_b)
  [x, y]
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
