# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)

require 'svmkit/version'

SVMKit::DESCRIPTION = <<MSG
SVMKit is a library for machine learninig in Ruby.
SVMKit implements machine learning algorithms with an interface similar to Scikit-Learn in Python.
However, since SVMKit is an experimental library, there are few machine learning algorithms implemented.
MSG

Gem::Specification.new do |spec|
  spec.name          = 'svmkit'
  spec.version       = SVMKit::VERSION
  spec.authors       = ['yoshoku']
  spec.email         = ['yoshoku@outlook.com']

  spec.summary       = %q{SVMKit is an experimental library of machine learning in Ruby.}
  spec.description   = SVMKit::DESCRIPTION
  spec.homepage      = 'https://github.com/yoshoku/svmkit'
  spec.license       = 'BSD-2-Clause'

  spec.files         = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end
  spec.bindir        = 'exe'
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ['lib']

  #spec.add_runtime_dependency 'nmatrix', '~> 0.2.3'

  spec.add_development_dependency 'bundler', '~> 1.15'
  spec.add_development_dependency 'rake', '~> 10.0'
  spec.add_development_dependency 'rspec', '~> 3.0'
  spec.add_development_dependency 'nmatrix', '~> 0.2.3'
end
