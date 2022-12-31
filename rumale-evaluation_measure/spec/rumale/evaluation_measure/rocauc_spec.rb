# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::ROCAUC do
  let(:roc_auc) { described_class.new }
  let(:y_bin) { Numo::Int32[1, 1, 2, 2] }
  let(:y_bin_score) { Numo::DFloat[0.1, 0.4, 0.35, 0.8] }
  let(:y_mult) do
    Numo::Int32[[1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [1, 0, 0]]
  end
  let(:y_mult_score) do
    Numo::DFloat[[0.5, 0.4, 0.1],
                 [0.1, 0.8, 0.1],
                 [0.3, 0.5, 0.2],
                 [0.5, 0.2, 0.3],
                 [0.8, 0.1, 0.1],
                 [0.1, 0.7, 0.2],
                 [0.2, 0.7, 0.1],
                 [0.3, 0.2, 0.5],
                 [0.3, 0.5, 0.2],
                 [0.7, 0.1, 0.2]]
  end

  it 'caclulates ROC-AUC', :aggregate_failures do
    expect(roc_auc.score(y_bin, y_bin_score)).to be_within(1e-4).of(0.75)
    expect(roc_auc.score(y_mult, y_mult_score)).to be_within(1e-4).of(0.7812)
  end

  it 'calculates receiver operating characteristic curve', :aggregate_failures do
    fpr, tpr, thd = roc_auc.roc_curve(y_bin, y_bin_score, 2)
    expect(fpr).to eq(Numo::DFloat[0, 0, 0.5, 0.5, 1])
    expect(tpr).to eq(Numo::DFloat[0, 0.5, 0.5, 1, 1])
    expect(thd).to eq(Numo::DFloat[1.8, 0.8, 0.4, 0.35, 0.1])
  end

  it 'calculates area under the ROC' do
    fpr, tpr, = roc_auc.roc_curve(y_bin, y_bin_score, 2)
    expect(roc_auc.auc(fpr, tpr)).to be_within(1e-4).of(0.75)
  end

  it 'raises ArgumentError given non-existent label to the roc_curve method' do
    expect { roc_auc.roc_curve(Numo::Int32[0, 1, 2], Numo::DFloat[0.1, 0.2, 0.8], 3) }.to raise_error(ArgumentError)
  end

  it 'raises ArgumentError given ground truth multi-label to the roc_curve method' do
    expect { roc_auc.roc_curve(Numo::Int32[0, 1, 2], Numo::DFloat[0.1, 0.2, 0.8]) }.to raise_error(ArgumentError)
  end

  it 'raises ArgumentError given arrays with only one element to the auc method', :aggregate_failures do
    expect { roc_auc.auc(Numo::Int32[0, 1, 2], Numo::Int32[1]) }.to raise_error(ArgumentError)
    expect { roc_auc.auc(Numo::Int32[0], Numo::Int32[1, 2, 3]) }.to raise_error(ArgumentError)
    expect { roc_auc.auc(Numo::Int32[0], Numo::Int32[1]) }.to raise_error(ArgumentError)
  end
end
