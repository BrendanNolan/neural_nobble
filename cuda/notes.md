# Pseudocode For GPU Tiled Matrix Multiplication

M = A.dim.i N = b.dim.j K = a.dim.j T = tile_size # = block_size

launch_config = (M/T, N/T)

```python
kernel() {
    i = blockDim.x * blockIdx
    j = blockDim.y * blockIdy
    shared=int[T][T] # Need shared_A, shared_B, shared_C
    # top_left = (i,j)
    for k=0; k < A.dim.j; k+= block_size
        for ii=i; ii<min(i+T,M); ++ii
            for jj=j; jj<min(j+T,N); ++jj
                shared[ii,jj] =
}
```
