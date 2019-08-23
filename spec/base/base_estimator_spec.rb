# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Base::BaseEstimator do
  let(:dummy) do
    class Dummy
      include Rumale::Base::BaseEstimator

      def initialize
        @params = {}
        @params[:n_jobs] = 1
      end

      def linalg?
        enable_linalg?
      end

      def parallel?
        enable_parallel?
      end
    end
    Dummy.new
  end

  describe '#enable_linalg?' do
    context 'If Numo::Linalg is loaded' do
      it 'returns true' do
        expect(dummy.linalg?).to be_truthy
      end
    end

    context 'If Numo::Linalg is not loaded' do
      before do
        @backup = Numo::Linalg
        Numo.class_eval { remove_const(:Linalg) }
      end

      after { Numo::Linalg = @backup }

      it 'returns false' do
        expect { dummy.linalg? }.to output(/you should install and load Numo::Linalg in advance./).to_stderr
        expect(dummy.linalg?).to be_falsy
      end
    end

    context 'When the version of Numo::Linalg is 0.1.3 or lower' do
      before do
        @backup = Numo::Linalg::VERSION
        Numo::Linalg.class_eval { remove_const(:VERSION) }
        Numo::Linalg::VERSION = '0.1.3'
      end

      after do
        Numo::Linalg.class_eval { remove_const(:VERSION) }
        Numo::Linalg::VERSION = @backup
      end

      it 'returns false' do
        expect { dummy.linalg? }.to output(/Please load Numo::Linalg version 0.1.4 or later./).to_stderr
        expect(dummy.linalg?).to be_falsy
      end
    end
  end

  describe '#enable_parallel?' do
    context 'If Parallel is loaded' do
      it 'returns true' do
        expect(dummy.parallel?).to be_truthy
      end
    end

    context 'If Numo::Linalg is not loaded' do
      before do
        @backup = Parallel
        Object.class_eval { remove_const(:Parallel) }
      end

      after { Parallel = @backup }

      it 'returns false' do
        expect { dummy.parallel? }.to output(/you should install and load Parallel in advance./).to_stderr
        expect(dummy.parallel?).to be_falsy
      end
    end
  end
end
