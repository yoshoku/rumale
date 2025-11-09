# frozen_string_literal: true

require 'rumale'

def iris
  Marshal.load(File.binread('spec/iris.dat')) # rubocop:disable Security/MarshalLoad
end

def housing
  Marshal.load(File.binread('spec/housing.dat')) # rubocop:disable Security/MarshalLoad
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
