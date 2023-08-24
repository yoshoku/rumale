# frozen_string_literal: true

require 'numo/tiny_linalg'
Numo::Linalg = Numo::TinyLinalg

require 'rumale/utils'
require 'rumale/kernel_approximation/rbf'

require 'simplecov'

SimpleCov.start do
  add_filter %w[spec vendor]
end

require 'rumale/decomposition'

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

RSpec.configure do |config|
  # Enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = '.rspec_status'

  # Disable RSpec exposing methods globally on `Module` and `main`
  config.disable_monkey_patching!

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end
end
