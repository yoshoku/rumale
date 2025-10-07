# frozen_string_literal: true

require_relative 'lib/rumale/neural_network/version'

Gem::Specification.new do |spec|
  spec.name = 'rumale-neural_network'
  spec.version = Rumale::NeuralNetwork::VERSION
  spec.authors = ['yoshoku']
  spec.email = ['yoshoku@outlook.com']

  spec.summary = <<~MSG
    Rumale::NeuralNetwork provides classifiers and regression algorithms based on multi-layer perceptron,
    radial basis function network, and random vector functional link network in the Rumale interface.
  MSG
  spec.description = <<~MSG
    Rumale::NeuralNetwork provides classifiers and regression algorithms based on multi-layer perceptron,
    radial basis function network, and random vector functional link network in the Rumale interface.
  MSG
  spec.homepage = 'https://github.com/yoshoku/rumale'
  spec.license = 'BSD-3-Clause'

  spec.metadata['homepage_uri'] = spec.homepage
  spec.metadata['source_code_uri'] = "#{spec.homepage}/tree/main/rumale-neural_network"
  spec.metadata['changelog_uri'] = "#{spec.homepage}/blob/main/CHANGELOG.md"
  spec.metadata['documentation_uri'] = 'https://yoshoku.github.io/rumale/doc/'
  spec.metadata['rubygems_mfa_required'] = 'true'

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  spec.files = Dir.chdir(File.expand_path(__dir__)) do
    `git ls-files -z`.split("\x0").reject { |f| f.match(%r{\A(?:(?:test|spec|features)/)}) }
                     .select { |f| f.match(/\.(?:rb|rbs|h|hpp|c|cpp|md|txt)$/) }
  end
  spec.bindir = 'exe'
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ['lib']

  spec.add_dependency 'numo-narray-alt', '~> 0.9.4'
  spec.add_dependency 'rumale-core', '~> 2.0.0'
end
