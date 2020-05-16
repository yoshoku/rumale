lib = File.expand_path('lib', __dir__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'rumale/version'

Gem::Specification.new do |spec|
  spec.name          = 'rumale'
  spec.version       = Rumale::VERSION
  spec.authors       = ['yoshoku']
  spec.email         = ['yoshoku@outlook.com']

  spec.summary       = <<~MSG
    Rumale is a machine learning library in Ruby.
    Rumale provides machine learning algorithms with interfaces similar to Scikit-Learn in Python.
  MSG
  spec.description = <<~MSG
    Rumale is a machine learning library in Ruby.
    Rumale provides machine learning algorithms with interfaces similar to Scikit-Learn in Python.
    Rumale supports Support Vector Machine,
    Logistic Regression, Ridge, Lasso, Factorization Machine,
    Multi-layer Perceptron,
    Naive Bayes, Decision Tree, Gradient Tree Boosting, Random Forest,
    K-Means, Gaussian Mixture Model, DBSCAN, Spectral Clustering,
    Mutidimensional Scaling, t-SNE,
    Fisher Discriminant Analysis, Neighbourhood Component Analysis,
    Principal Component Analysis, Non-negative Matrix Factorization,
    and many other algorithms.
  MSG
  spec.homepage      = 'https://github.com/yoshoku/rumale'
  spec.license       = 'BSD-2-Clause'

  spec.files         = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end
  spec.bindir        = 'exe'
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ['lib']
  spec.extensions    = ['ext/rumale/extconf.rb']

  spec.metadata      = {
    'homepage_uri' => 'https://github.com/yoshoku/rumale',
    'changelog_uri' => 'https://github.com/yoshoku/rumale/blob/master/CHANGELOG.md',
    'source_code_uri' => 'https://github.com/yoshoku/rumale',
    'documentation_uri' => 'https://yoshoku.github.io/rumale/doc/',
    'bug_tracker_uri' => 'https://github.com/yoshoku/rumale/issues'
  }

  spec.add_runtime_dependency 'numo-narray', '>= 0.9.1'
  spec.add_runtime_dependency 'mopti', '>= 0.1.0'
end
