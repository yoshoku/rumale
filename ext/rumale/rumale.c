#include "rumale.h"

VALUE mRumale;

void Init_rumale(void)
{
  mRumale = rb_define_module("Rumale");

  init_tree_module();
}
