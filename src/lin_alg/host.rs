use crate::lin_alg::Dim;
use std::ops::{Index, IndexMut, Mul};

pub struct HostMatrix {
    raw: Vec<f64>,
    dim: Dim,
}

impl HostMatrix {
    fn zeroes(dim: Dim) -> Self {
        let raw = vec![0_f64; dim.rows * dim.cols];
        Self { raw, dim }
    }
}

impl Index<(usize, usize)> for HostMatrix {
    type Output = f64;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        &self.raw[i * self.dim.cols + j]
    }
}

impl IndexMut<(usize, usize)> for HostMatrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = index;
        &mut self.raw[i * self.dim.cols + j]
    }
}

fn compatible_for_multiplication(a: &HostMatrix, b: &HostMatrix) -> bool {
    a.dim.cols == b.dim.rows
}

fn tiled_multiply(a: &HostMatrix, b: &HostMatrix, tile_size: usize) -> HostMatrix {
    debug_assert!(compatible_for_multiplication(a, b));
    let mut c = HostMatrix::zeroes(Dim {
        rows: a.dim.rows,
        cols: b.dim.cols,
    });
    let m = a.dim.rows;
    let n = b.dim.cols;
    let t = tile_size;
    let common = a.dim.cols;
    for i in (0..m).step_by(t) {
        for j in (0..n).step_by(t) {
            for k in (0..common).step_by(t) {
                for ii in (i..(i + t).min(m)) {
                    for kk in (k..(k + t).min(common)) {
                        for jj in (j..(j + t).min(n)) {
                            c[(ii, jj)] += a[(ii, kk)] * b[(kk, jj)];
                        }
                    }
                }
            }
        }
    }
    c
}

impl Mul for &HostMatrix {
    type Output = HostMatrix;
    fn mul(self, rhs: Self) -> Self::Output {
        tiled_multiply(self, rhs, 8)
    }
}
