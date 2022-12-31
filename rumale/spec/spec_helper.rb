# frozen_string_literal: true

require 'rumale'
require 'datasets-numo-narray'

def iris
  dataset = Datasets::LIBSVM.new('iris').to_narray
  x = dataset[true, 1..-1]
  y = Numo::Int32.cast(dataset[true, 0])

  ss = Rumale::ModelSelection::StratifiedShuffleSplit.new(n_splits: 1, test_size: 0.2, random_seed: 38)
  train_ids, test_ids = ss.split(x, y)[0]

  { x_train: x[train_ids, true].dup,
    y_train: y[train_ids].dup,
    x_test: x[test_ids, true].dup,
    y_test: y[test_ids].dup }
end

def housing
  dataset = Datasets::LIBSVM.new('housing', note: 'scaled to [-1,1]').to_narray
  x = dataset[true, 1..-1]
  y = dataset[true, 0]

  ss = Rumale::ModelSelection::ShuffleSplit.new(n_splits: 1, test_size: 0.1, random_seed: 38)
  train_ids, test_ids = ss.split(x, y)[0]

  { x_train: x[train_ids, true].dup,
    y_train: y[train_ids].dup,
    x_test: x[test_ids, true].dup,
    y_test: y[test_ids].dup }
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
