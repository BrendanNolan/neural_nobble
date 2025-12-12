- In the ffi module, decide where we want void* and float* and decide whether `count`s are counting
  floats or bytes. Probably all counts should represent floats and there should be no void pointers.
- Think of making a simple struct for a cost function (one that contains a fn) and think of giving
  it a gradient() method which returns perhaps an Option.
