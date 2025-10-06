# frozen_string_literal: true

require 'numo/linalg'
require 'rumale/dataset'
require 'rumale/kernel_approximation/rbf'
require 'simplecov'

SimpleCov.start do
  add_filter %w[spec vendor]
end

require 'rumale/manifold'

def two_clusters_dataset
  centers = Numo::DFloat[[-10, 0], [10, 0]]
  Rumale::Dataset.make_blobs(200, centers: centers, cluster_std: 0.5, random_seed: 1)
end

def swiss_roll
  n_samples = 1000
  rng = Random.new(42)
  theta = 1.5 * Math::PI * (1 + 2 * Rumale::Utils.rand_uniform(n_samples, rng))
  y = 90.0 * Rumale::Utils.rand_uniform(n_samples, rng)
  x = theta * Numo::NMath.cos(theta)
  z = theta * Numo::NMath.sin(theta)
  data = Numo::NArray.vstack([x, y, z]).transpose.dup
  data_ids = Array(0...n_samples)
  test_ids = data_ids.sample(100, random: rng)
  train_ids = data_ids - test_ids
  [data[train_ids, true].dup, data[test_ids, true].dup]
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
