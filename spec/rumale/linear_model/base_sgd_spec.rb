# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::LinearModel::BaseSGD do
  let(:x) { Numo::DFloat[[40, 50], [60, 70], [80, 90]] }
  let(:y) { Numo::Int32[-1, 1, 1] }
  let(:w) { Numo::DFloat[10, 20, 30] }
  let(:base_sgd) { described_class.new }

  context 'when loss function is given' do
    before do
      base_sgd.instance_variable_set(:@loss_func, Rumale::LinearModel::Loss::MeanSquaredError.new)
      base_sgd.instance_variable_set(:@rng, Random.new(1))
      base_sgd.params[:learning_rate] = 0.0001
      base_sgd.params[:decay] = 0.00001
      base_sgd.params[:fit_bias] = true
    end

    it 'performs learning process' do
      uw, ub = base_sgd.send(:partial_fit, x, y)
      expect(uw.class).to eq(Numo::DFloat)
      expect(uw.ndim).to eq(1)
      expect(uw.shape[0]).to eq(2)
      expect(ub.class).to eq(Float)
    end
  end

  context 'when fit_bias is "false"' do
    before do
      base_sgd.params[:fit_bias] = false
    end

    it 'does not consider bias term' do
      expect(base_sgd.send(:fit_bias?)).to be_falsey
    end

    it 'adds constant value for bias term' do
      expect(base_sgd.send(:expand_feature, x)).to eq(Numo::DFloat[[40, 50, 1], [60, 70, 1], [80, 90, 1]])
    end

    it 'does not split vector into weight vector and bias term' do
      expect(base_sgd.send(:split_weight, w)).to eq([w, 0.0])
    end
  end

  context 'when fit_bias is "true"' do
    before do
      base_sgd.params[:fit_bias] = true
      base_sgd.params[:bias_scale] = 5
    end

    it 'considers bias term' do
      expect(base_sgd.send(:fit_bias?)).to be_truthy
    end

    it 'adds constant value for bias term' do
      expect(base_sgd.send(:expand_feature, x)).to eq(Numo::DFloat[[40, 50, 5], [60, 70, 5], [80, 90, 5]])
    end

    it 'splits vector into weight vector and bias term' do
      expect(base_sgd.send(:split_weight, w)).to eq([Numo::DFloat[10, 20], 30])
    end
  end

  context 'when penalty is L2' do
    before do
      base_sgd.instance_variable_set(:@penalty_type, 'l2')
      base_sgd.params[:reg_param] = 0.1
      base_sgd.params[:l1_ratio] = 0.2
    end

    it 'applies L2 regularization', :aggregate_failures do
      expect(base_sgd.send(:apply_l2_penalty?)).to be_truthy
      expect(base_sgd.send(:apply_l1_penalty?)).to be_falsey
      expect(base_sgd.send(:l2_reg_param)).to eq(0.1)
      expect(base_sgd.send(:l1_reg_param)).to eq(0.0)
    end
  end

  context 'when penalty is L1' do
    before do
      base_sgd.instance_variable_set(:@penalty_type, 'l1')
      base_sgd.params[:reg_param] = 0.1
      base_sgd.params[:l1_ratio] = 0.2
    end

    it 'applies L1 regularization', :aggregate_failures do
      expect(base_sgd.send(:apply_l2_penalty?)).to be_falsey
      expect(base_sgd.send(:apply_l1_penalty?)).to be_truthy
      expect(base_sgd.send(:l2_reg_param)).to eq(0.0)
      expect(base_sgd.send(:l1_reg_param)).to eq(0.1)
    end
  end

  context 'when penalty is Elastic-net' do
    before do
      base_sgd.instance_variable_set(:@penalty_type, 'elasticnet')
      base_sgd.params[:reg_param] = 1
      base_sgd.params[:l1_ratio] = 0.2
    end

    it 'applies L2 and L1 regularization', :aggregate_failures do
      expect(base_sgd.send(:apply_l2_penalty?)).to be_truthy
      expect(base_sgd.send(:apply_l1_penalty?)).to be_truthy
      expect(base_sgd.send(:l2_reg_param)).to eq(0.8)
      expect(base_sgd.send(:l1_reg_param)).to eq(0.2)
    end
  end
end
