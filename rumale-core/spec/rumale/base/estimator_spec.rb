# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Base::Estimator do
  let(:dummy) do
    Class.new(described_class) do
      def initialize
        super
        @params = {}
        @params[:n_jobs] = 2
      end

      def linalg?(warning: false)
        enable_linalg?(warning: warning)
      end

      def parallel?(warning: false)
        enable_parallel?(warning: warning)
      end

      def n_procs
        n_processes
      end
    end.new
  end

  describe '#enable_linalg?' do
    context 'when Numo::Linalg is loaded' do
      before { stub_const('Numo::Linalg::VERSION', '0.1.4') }

      it 'returns true' do
        expect(dummy).to be_linalg
      end
    end

    context 'when Numo::Linalg is not loaded' do
      it 'returns false', :aggregate_failures do
        expect { dummy.linalg?(warning: true) }.to output(/you should install and load Numo::Linalg in advance./).to_stderr
        expect(dummy).not_to be_linalg
      end

      it 'returns false without warning messages', :aggregate_failures do
        expect { dummy.linalg?(warning: false) }.not_to output(/you should install and load Numo::Linalg/).to_stderr
        expect(dummy).not_to be_linalg
      end
    end

    context 'when the version of Numo::Linalg is 0.1.3 or lower' do
      before { stub_const('Numo::Linalg::VERSION', '0.1.3') }

      it 'returns false', :aggregate_failures do
        expect { dummy.linalg?(warning: true) }.to output(/Please load Numo::Linalg version 0.1.4 or later./).to_stderr
        expect(dummy).not_to be_linalg
      end
    end
  end

  describe '#enable_parallel?' do
    context 'when Parallel is loaded' do
      before { stub_const('Parallel', Module.new) }

      it 'returns true' do
        expect(dummy).to be_parallel
      end
    end

    context 'when Parallel is not loaded' do
      it 'returns false', :aggregate_failures do
        expect { dummy.parallel?(warning: true) }.to output(/you should install and load Parallel in advance./).to_stderr
        expect(dummy).not_to be_parallel
      end
    end
  end

  describe '#n_processes' do
    context 'when Parallel is loaded' do
      before do
        stub_const(
          'Parallel',
          Module.new do
            def self.processor_count
              3
            end
          end
        )
      end

      context 'with positive value of n_jobs' do
        it 'returns value of n_jobs' do
          expect(dummy.n_procs).to eq(dummy.params[:n_jobs])
        end
      end

      context 'with negative value of n_jobs' do
        before { dummy.params[:n_jobs] = -1 }

        it 'returns value of Parallel.processor_count' do
          expect(dummy.n_procs).to eq(3)
        end
      end
    end

    context 'when Parallel is not loaded' do
      it 'returns 1' do
        expect(dummy.n_procs).to eq(1)
      end
    end
  end
end
