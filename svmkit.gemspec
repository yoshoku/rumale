# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
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
SVMKit currently supports Linear / Kernel Support Vector Machine,
Logistic Regression, Factorization Machine, Naive Bayes,
K-nearest neighbor classifier, and cross-validation.
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

  spec.add_runtime_dependency 'numo-narray', '~> 0.9.0'

  spec.add_development_dependency 'bundler', '~> 1.16'
  spec.add_development_dependency 'rake', '~> 12.0'
  spec.add_development_dependency 'rspec', '~> 3.0'
  spec.add_development_dependency 'simplecov', '~> 0.15'

  spec.post_install_message = <<MSG
*************************************************************************
Thank you for installing SVMKit!!

Note that the SVMKit has been changed to use Numo::NArray for
linear algebra library from version 0.2.0.
*************************************************************************
MSG

end
