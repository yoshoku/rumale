# frozen_string_literal: true

require_relative 'lib/rumale/tree/version'

Gem::Specification.new do |spec|
  spec.name = 'rumale-tree'
  spec.version = Rumale::Tree::VERSION
  spec.authors = ['yoshoku']
  spec.email = ['yoshoku@outlook.com']

  spec.summary = 'Rumale::Tree provides classifier and regression based on decision tree algorithms with Rumale interface.'
  spec.description = 'Rumale::Tree provides classifier and regression based on decision tree algorithms with Rumale interface.'
  spec.homepage = 'https://github.com/yoshoku/rumale'
  spec.license = 'BSD-3-Clause'

  spec.metadata['homepage_uri'] = spec.homepage
  spec.metadata['source_code_uri'] = "#{spec.homepage}/tree/main/rumale-tree"
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
  spec.extensions = ['ext/rumale/tree/extconf.rb']

  spec.add_dependency 'numo-narray-alt', '~> 0.9.4'
  spec.add_dependency 'rumale-core', '~> 2.0.1'
end
