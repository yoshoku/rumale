# frozen_string_literal: true

require_relative 'lib/rumale/version'

Gem::Specification.new do |spec|
  spec.name = 'rumale'
  spec.version = Rumale::VERSION
  spec.authors = ['yoshoku']
  spec.email = ['yoshoku@outlook.com']

  spec.summary = <<~MSG
    Rumale is a machine learning library in Ruby.
    Rumale supports Support Vector Machine,
    Logistic Regression, Multi-layer Perceptron,
    Naive Bayes, Decision Tree, Random Forest,
    K-Means, Gaussian Mixture Model, DBSCAN,
    Principal Component Analysis,
    and many other algorithms.
  MSG
  spec.description = <<~MSG
    Rumale is a machine learning library in Ruby.
    Rumale provides machine learning algorithms with interfaces similar to Scikit-Learn in Python.
    Rumale supports Support Vector Machine,
    Logistic Regression, Ridge, Lasso,
    Multi-layer Perceptron,
    Naive Bayes, Decision Tree, Gradient Tree Boosting, Random Forest,
    K-Means, Gaussian Mixture Model, DBSCAN, Spectral Clustering,
    Mutidimensional Scaling, t-SNE,
    Fisher Discriminant Analysis, Neighbourhood Component Analysis,
    Principal Component Analysis, Non-negative Matrix Factorization,
    and many other algorithms.
  MSG
  spec.homepage = 'https://github.com/yoshoku/rumale'
  spec.license = 'BSD-3-Clause'

  spec.metadata['homepage_uri'] = spec.homepage
  spec.metadata['source_code_uri'] = spec.homepage
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

  spec.add_dependency 'numo-narray-alt', '~> 0.9.10'
  spec.add_dependency 'rumale-clustering', '~> 2.0.2'
  spec.add_dependency 'rumale-core', '~> 2.0.2'
  spec.add_dependency 'rumale-decomposition', '~> 2.0.2'
  spec.add_dependency 'rumale-ensemble', '~> 2.0.2'
  spec.add_dependency 'rumale-evaluation_measure', '~> 2.0.2'
  spec.add_dependency 'rumale-feature_extraction', '~> 2.0.2'
  spec.add_dependency 'rumale-kernel_approximation', '~> 2.0.2'
  spec.add_dependency 'rumale-kernel_machine', '~> 2.0.2'
  spec.add_dependency 'rumale-linear_model', '~> 2.0.2'
  spec.add_dependency 'rumale-manifold', '~> 2.0.2'
  spec.add_dependency 'rumale-metric_learning', '~> 2.0.2'
  spec.add_dependency 'rumale-model_selection', '~> 2.0.2'
  spec.add_dependency 'rumale-naive_bayes', '~> 2.0.2'
  spec.add_dependency 'rumale-nearest_neighbors', '~> 2.0.2'
  spec.add_dependency 'rumale-neural_network', '~> 2.0.2'
  spec.add_dependency 'rumale-pipeline', '~> 2.0.2'
  spec.add_dependency 'rumale-preprocessing', '~> 2.0.2'
  spec.add_dependency 'rumale-tree', '~> 2.0.2'
end
