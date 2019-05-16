lib = File.expand_path('lib', __dir__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'rumale/version'

Gem::Specification.new do |spec|
  spec.name          = 'rumale'
  spec.version       = Rumale::VERSION
  spec.authors       = ['yoshoku']
  spec.email         = ['yoshoku@outlook.com']

  spec.summary       = <<MSG
Rumale is a machine learning library in Ruby.
Rumale provides machine learning algorithms with interfaces similar to Scikit-Learn in Python.
MSG
  spec.description   = <<MSG
Rumale is a machine learning library in Ruby.
Rumale provides machine learning algorithms with interfaces similar to Scikit-Learn in Python.
Rumale currently supports Linear / Kernel Support Vector Machine,
Logistic Regression, Linear Regression, Ridge, Lasso, Factorization Machine,
Naive Bayes, Decision Tree, AdaBoost, Gradient Tree Boosting, Random Forest, Extra-Trees, K-nearest neighbor algorithm,
K-Means, DBSCAN, t-SNE, Principal Component Analysis, and Non-negative Matrix Factorization.
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

  spec.required_ruby_version = '>= 2.3'

  spec.add_runtime_dependency 'numo-narray', '>= 0.9.1'

  spec.add_development_dependency 'bundler', '>= 1.16'
  spec.add_development_dependency 'coveralls', '~> 0.8'
  spec.add_development_dependency 'rake', '~> 12.0'
  spec.add_development_dependency 'rake-compiler'
  spec.add_development_dependency 'rspec', '~> 3.0'
end
