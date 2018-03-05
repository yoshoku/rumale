# frozen_string_literal: true

module SVMKit
  # Module for loading and saving a dataset file.
  module Dataset
    class << self
      # Load a dataset with the libsvm file format into Numo::NArray.
      #
      # @param filename [String] A path to a dataset file.
      # @param zero_based [Boolean] Whether the column index starts from 0 (true) or 1 (false).
      #
      # @return [Array<Numo::NArray>]
      #   Returns array containing the (n_samples x n_features) matrix for feature vectors
      #   and (n_samples) vector for labels or target values.
      def load_libsvm_file(filename, zero_based: false)
        ftvecs = []
        labels = []
        n_features = 0
        File.read(filename).split("\n").each do |line|
          label, ftvec, max_idx = parse_libsvm_line(line, zero_based)
          labels.push(label)
          ftvecs.push(ftvec)
          n_features = [n_features, max_idx].max
        end
        [convert_to_matrix(ftvecs, n_features), Numo::NArray.asarray(labels)]
      end

      # Dump the dataset with the libsvm file format.
      #
      # @param data [Numo::NArray] (shape: [n_samples, n_features]) matrix consisting of feature vectors.
      # @param labels [Numo::NArray] (shape: [n_samples]) matrix consisting of labels or target values.
      # @param filename [String] A path to the output libsvm file.
      # @param zero_based [Boolean] Whether the column index starts from 0 (true) or 1 (false).
      def dump_libsvm_file(data, labels, filename, zero_based: false)
        n_samples = [data.shape[0], labels.shape[0]].min
        label_type = detect_dtype(labels)
        value_type = detect_dtype(data)
        File.open(filename, 'w') do |file|
          n_samples.times do |n|
            file.puts(dump_libsvm_line(labels[n], data[n, true],
                                       label_type, value_type, zero_based))
          end
        end
      end

      private

      def parse_libsvm_line(line, zero_based)
        tokens = line.split
        label = tokens.shift
        label = label.to_i.to_s == label ? label.to_i : label.to_f
        ftvec = tokens.map do |el|
          idx, val = el.split(':')
          idx = idx.to_i - (zero_based == false ? 1 : 0)
          val = val.to_i.to_s == val ? val.to_i : val.to_f
          [idx, val]
        end
        max_idx = ftvec.map { |el| el[0] }.max
        max_idx ||= 0
        [label, ftvec, max_idx]
      end

      def convert_to_matrix(data, n_features)
        mat = []
        data.each do |ft|
          vec = Array.new(n_features) { 0 }
          ft.each { |el| vec[el[0]] = el[1] }
          mat.push(vec)
        end
        Numo::NArray.asarray(mat)
      end

      def detect_dtype(data)
        arr_type_str = Numo::NArray.array_type(data).to_s
        type = '%s'
        type = '%d' if ['Numo::Int8', 'Numo::Int16', 'Numo::Int32', 'Numo::Int64'].include?(arr_type_str)
        type = '%d' if ['Numo::UInt8', 'Numo::UInt16', 'Numo::UInt32', 'Numo::UInt64'].include?(arr_type_str)
        type = '%.10g' if ['Numo::SFloat', 'Numo::DFloat'].include?(arr_type_str)
        type
      end

      def dump_libsvm_line(label, ftvec, label_type, value_type, zero_based)
        line = format(label_type.to_s, label)
        ftvec.to_a.each_with_index do |val, n|
          idx = n + (zero_based == false ? 1 : 0)
          line += format(" %d:#{value_type}", idx, val) if val != 0.0
        end
        line
      end
    end
  end
end
