# frozen_string_literal: true

require_relative 'lib/rumale/preprocessing/version'

Gem::Specification.new do |spec|
  spec.name = 'rumale-preprocessing'
  spec.version = Rumale::Preprocessing::VERSION
  spec.authors = ['yoshoku']
  spec.email = ['yoshoku@outlook.com']

  spec.summary = 'Rumale::Preprocessing provides preprocessing techniques with Rumale interface.'
  spec.description = <<~MSG
    Rumale::Preprocessing provides preprocessing techniques,
    such as L2 normalization, standard scaling, and one-hot encoding,
    with Rumale interface.
  MSG
  spec.homepage = 'https://github.com/yoshoku/rumale'
  spec.license = 'BSD-3-Clause'

  spec.metadata['homepage_uri'] = spec.homepage
  spec.metadata['source_code_uri'] = "#{spec.homepage}/tree/main/rumale-preprocessing"
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

  spec.add_dependency 'numo-narray', '>= 0.9.1'
  spec.add_dependency 'rumale-core', '~> 1.0.0'
end
