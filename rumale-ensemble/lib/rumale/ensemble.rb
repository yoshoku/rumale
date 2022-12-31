# frozen_string_literal: true

require 'numo/narray'

require_relative 'ensemble/version'

require_relative 'ensemble/value'

require_relative 'ensemble/ada_boost_classifier'
require_relative 'ensemble/ada_boost_regressor'
require_relative 'ensemble/extra_trees_classifier'
require_relative 'ensemble/extra_trees_regressor'
require_relative 'ensemble/gradient_boosting_classifier'
require_relative 'ensemble/gradient_boosting_regressor'
require_relative 'ensemble/random_forest_classifier'
require_relative 'ensemble/random_forest_regressor'
require_relative 'ensemble/stacking_classifier'
require_relative 'ensemble/stacking_regressor'
require_relative 'ensemble/voting_classifier'
require_relative 'ensemble/voting_regressor'
