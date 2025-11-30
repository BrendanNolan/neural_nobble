#![allow(unused)] // TODO: Remove this
#![allow(dead_code)] // TODO: Remove this
#![allow(unused_imports)] // TODO: Remove this

pub mod activation_functions;
mod array_utils;
mod back_propagation;
mod common;
pub mod convolution;
pub mod cost_functions;
mod distribution;
pub mod feed_forward;
mod ffi;
mod gradient_descent;
mod lin_alg;
mod logging;
pub mod mini_batch;
pub mod neural_network;
pub mod one_hot;
pub mod train;
