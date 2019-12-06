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
    Mutidimensional Scaling, t-SNE, Principal Component Analysis, Non-negative Matrix Factorization,
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

  spec.required_ruby_version = '>= 2.3'

  spec.add_runtime_dependency 'numo-narray', '>= 0.9.1'

  spec.add_development_dependency 'bundler', '~> 2.0'
  spec.add_development_dependency 'coveralls', '~> 0.8'
  spec.add_development_dependency 'numo-linalg', '>= 0.1.4'
  spec.add_development_dependency 'parallel', '>= 1.17.0'
  spec.add_development_dependency 'rake', '~> 10.0'
  spec.add_development_dependency 'rake-compiler', '~> 1.0'
  spec.add_development_dependency 'rspec', '~> 3.0'
end
