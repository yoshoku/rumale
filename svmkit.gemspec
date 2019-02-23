lib = File.expand_path('lib', __dir__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'svmkit/version'

Gem::Specification.new do |spec|
  spec.name          = 'svmkit'
  spec.version       = SVMKit::VERSION
  spec.authors       = ['yoshoku']
  spec.email         = ['yoshoku@outlook.com']

  spec.summary       = <<MSG
SVMKit is a machine learninig library in Ruby.
SVMKit provides machine learning algorithms with interfaces similar to Scikit-Learn in Python.
MSG
  spec.description   = <<MSG
SVMKit is a machine learninig library in Ruby.
SVMKit provides machine learning algorithms with interfaces similar to Scikit-Learn in Python.
SVMKit supports Linear / Kernel Support Vector Machine,
Logistic Regression, Linear Regression, Ridge, Lasso, Factorization Machine,
Naive Bayes, Decision Tree, AdaBoost, Random Forest, K-nearest neighbor algorithm,
K-Means, DBSCAN, Principal Component Analysis, and Non-negative Matrix Factorization.
Note that the SVMKit has been deprecated and has been renamed to Rumale.
SVMKit will continue releasing for bugfix but will not add new features.
MSG
  spec.post_install_message = <<MSG
*************************************************************************
Note that the SVMKit has been deprecated and has been renamed to Rumale.
Please see https://rubygems.org/gems/rumale
*************************************************************************
MSG
  spec.homepage      = 'https://github.com/yoshoku/svmkit'
  spec.license       = 'BSD-2-Clause'

  spec.files         = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end
  spec.bindir        = 'exe'
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ['lib']

  spec.required_ruby_version = '>= 2.1'

  spec.add_runtime_dependency 'numo-narray', '>= 0.9.1'

  spec.add_development_dependency 'bundler', '>= 1.16'
  spec.add_development_dependency 'coveralls', '~> 0.8'
  spec.add_development_dependency 'rake', '~> 12.0'
  spec.add_development_dependency 'rspec', '~> 3.0'
end
