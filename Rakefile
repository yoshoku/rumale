require 'bundler/gem_tasks'
require 'rspec/core/rake_task'

require 'rubocop/rake_task'

RuboCop::RakeTask.new

RSpec::Core::RakeTask.new(:spec)

require 'rake/extensiontask'

task :build => :compile

Rake::ExtensionTask.new('rumaleext') do |ext|
  ext.ext_dir = 'ext/rumale'
  ext.lib_dir = 'lib/rumale'
end

task :default => [:clobber, :compile, :rubocop, :spec]
