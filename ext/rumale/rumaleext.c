#include "rumaleext.h"

VALUE mRumale;

void Init_rumaleext(void) {
  mRumale = rb_define_module("Rumale");

  init_tree_module();
}
