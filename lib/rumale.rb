# frozen_string_literal: true

require 'numo/narray'

require 'rumale/rumaleext'

require 'rumale/multiclass/one_vs_rest_classifier'
Dir[File.join(__dir__, 'rumale/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/base/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/pipeline/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/kernel_approximation/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/linear_model/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/kernel_machine/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/nearest_neighbors/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/naive_bayes/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/tree/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/ensemble/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/clustering/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/decomposition/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/manifold/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/metric_learning/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/neural_network/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/feature_extraction/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/preprocessing/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/model_selection/*.rb')].sort.each { |file| require file }
Dir[File.join(__dir__, 'rumale/evaluation_measure/*.rb')].sort.each { |file| require file }
