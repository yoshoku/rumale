# frozen_string_literal: true

require 'mkmf'
require 'numo/narray'

$LOAD_PATH.each do |lp|
  if File.exist?(File.join(lp, 'numo/numo/narray.h'))
    $INCFLAGS = "-I#{lp}/numo #{$INCFLAGS}"
    break
  end
end

if RUBY_PLATFORM =~ /mswin|cygwin|mingw/
  $LOAD_PATH.each do |lp|
    if File.exist? File.join(lp, 'numo/libnarray.a')
      $LDFLAGS = "-L#{lp}/numo #{$LDFLAGS}"
      break
    end
  end
end

create_makefile('rumale/rumale')
