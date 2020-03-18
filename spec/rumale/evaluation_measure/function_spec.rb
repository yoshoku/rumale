# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure do
  describe '#confusion_matrix' do
    let(:y_true) { Numo::Int32[0, 1, 2, 2, 2, 0, 1, 2, 0] }
    let(:y_pred) { Numo::Int32[0, 0, 2, 0, 2, 0, 0, 2, 1] }
    let(:res) { described_class.confusion_matrix(y_true, y_pred) }

    it 'calculates confusion matrix' do
      expect(res).to eq(Numo::Int32[[2, 1, 0], [2, 0, 0], [1, 0, 3]])
    end
  end

  describe '#classification_report' do
    let(:y_true) { Numo::Int32[0, 1, 2, 2, 2, 0, 1, 2] }
    let(:y_pred) { Numo::Int32[0, 0, 2, 0, 2, 0, 0, 2] }
    let(:target_name) { nil }
    let(:output_hash) { false }
    let(:res) { described_class.classification_report(y_true, y_pred, target_name: target_name, output_hash: output_hash) }

    context 'when output_hash sets to false' do
      let(:output_hash) { false }

      it 'outputs table' do
        expect(res).to eq(
          <<~RESULT
                          precision    recall  f1-score   support

                       0       0.40      1.00      0.57         2
                       1       0.00      0.00      0.00         2
                       2       1.00      0.75      0.86         4

                accuracy                           0.62         8
               macro avg       0.47      0.58      0.48         8
            weighted avg       0.60      0.62      0.57         8
          RESULT
        )
      end
    end

    context 'when output_hash sets to true' do
      let(:output_hash) { true }

      it 'outputs hash' do
        expect(res.keys.map(&:to_s)).to eq(%w[0 1 2 accuracy macro_avg weighted_avg])
        expect(res['0'].keys).to eq(%i[precision recall fscore support])
        expect(res[:macro_avg].keys).to eq(%i[precision recall fscore support])
        expect(res[:weighted_avg].keys).to eq(%i[precision recall fscore support])
      end
    end

    context 'when target_name is given' do
      let(:target_name) { %w[a b c] }

      context 'when output_hash sets to true' do
        let(:output_hash) { true }

        it 'outputs hash' do
          expect(res.keys[0...3]).to eq(target_name)
        end
      end

      context 'when output is string' do
        let(:output_hash) { false }

        it 'outputs table' do
          expect(res).to eq(
            <<~RESULT
                            precision    recall  f1-score   support

                         a       0.40      1.00      0.57         2
                         b       0.00      0.00      0.00         2
                         c       1.00      0.75      0.86         4

                  accuracy                           0.62         8
                 macro avg       0.47      0.58      0.48         8
              weighted avg       0.60      0.62      0.57         8
            RESULT
          )
        end
      end

      context 'when target_name contains long name' do
        let(:target_name) { %w[a b cdefghijklmno] }
        let(:output_hash) { false }

        it 'outputs table' do
          expect(res).to eq(
            <<~RESULT
                             precision    recall  f1-score   support

                          a       0.40      1.00      0.57         2
                          b       0.00      0.00      0.00         2
              cdefghijklmno       1.00      0.75      0.86         4

                   accuracy                           0.62         8
                  macro avg       0.47      0.58      0.48         8
               weighted avg       0.60      0.62      0.57         8
            RESULT
          )
        end
      end
    end
  end
end
