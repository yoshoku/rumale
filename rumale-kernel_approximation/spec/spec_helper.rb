# frozen_string_literal: true

require 'numo/linalg'
require 'rumale/dataset'
require 'simplecov'

SimpleCov.start do
  add_filter %w[spec vendor]
end

require 'rumale/kernel_approximation'

def three_clusters_dataset
  centers = Numo::DFloat[[0, 5], [-5, -5], [5, -5]]
  Rumale::Dataset.make_blobs(300, centers: centers, cluster_std: 0.5, random_seed: 1)
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
