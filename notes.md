- Think of making a simple struct for a cost function (one that contains a fn) and think of giving
  it a gradient() method which returns perhaps an Option.
- Implement (on the CUDA side) matrix multiplication that optionally transposes one of the operand
  matrices.
- Finish implementing the `calculate_training_loop_buffer_size` function and use it to find the
  required size a memory arena on the CUDA side, which can be used and reused during the training
  loop. Each matrix product grabs the arena pointer, puts the produce matrix in there, and bumps the
  arena pointer. ALTERNATIVELY, cudaMalloc the required memory on the first training loop iteration
  only, storing the pointers in an array, then just reuse them on all subsequent iterations; the
  training loop has no branches and is single threaded, so you just grab the next pointer from the
  array every time you need to do a matrix multiplication. Then just cudaFree them all in the end.
