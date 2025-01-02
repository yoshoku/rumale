# frozen_string_literal: true

require_relative 'lib/rumale/manifold/version'

Gem::Specification.new do |spec|
  spec.name = 'rumale-manifold'
  spec.version = Rumale::Manifold::VERSION
  spec.authors = ['yoshoku']
  spec.email = ['yoshoku@outlook.com']

  spec.summary = 'Rumale::Manifold provides data embedding algorithms with Rumale interface.'
  spec.description = <<~MSG
    Rumale::Manifold provides data embedding algorithms,
    such as Multi-dimensional Scaling, Locally Linear Embedding, Laplacian Eigenmaps, Hessian Eigenmaps,
    and t-distributed Stochastic Neighbor Embedding,
    with Rumale interface.
  MSG
  spec.homepage = 'https://github.com/yoshoku/rumale'
  spec.license = 'BSD-3-Clause'

  spec.metadata['homepage_uri'] = spec.homepage
  spec.metadata['source_code_uri'] = "#{spec.homepage}/-/tree/main/rumale-manifold"
  spec.metadata['changelog_uri'] = "#{spec.homepage}/-/blob/main/CHANGELOG.md"
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

  spec.add_dependency 'numo-narray', '>= 0.9.1'
  spec.add_dependency 'rumale-core', '~> 1.0.0'
  spec.add_dependency 'rumale-decomposition', '~> 1.0.0'
end
