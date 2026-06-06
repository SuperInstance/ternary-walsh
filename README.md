# ternary-walsh

Walsh functions and transforms for ternary-valued signal analysis.

[![crates.io](https://img.shields.io/crates/v/ternary-walsh.svg)](https://crates.io/crates/ternary-walsh)

## Overview

Walsh functions are the square-wave analogs of sinusoids — orthogonal functions that take
only values in {-1, +1} (or {-1, 0, +1} in the ternary extension). They form complete
orthogonal bases ideal for representing piecewise-constant and digital signals, especially
those defined over the ternary alphabet {-1, 0, +1}.

This crate provides:

- **Hadamard matrix generation** — Sylvester-type recursive construction for any power-of-2 order
- **Walsh (Hadamard) transform** — Naive O(N²) matrix multiplication on ternary signals
- **Fast Walsh-Hadamard Transform (FWHT)** — O(N log N) butterfly algorithm
- **Sequency ordering** — Reorder Hadamard rows by zero-crossing count (analogous to frequency)
- **Walsh power spectrum** — Squared magnitude of transform coefficients with Parseval's theorem
- **Round-trip transforms** — Perfect reconstruction via inverse transforms
- **Ternary type** — First-class `Ternary` enum for {-1, 0, +1} values

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ternary-walsh = "0.1.0"
```

## Quick Start

```rust
use ternary_walsh::*;

// Generate a Hadamard matrix of order 8
let h = hadamard_matrix(8);

// Create a ternary signal
let signal = vec![1, -1, 0, 1, -1, 0, 1, -1];

// Naive Walsh transform
let transformed = walsh_transform(&signal, &h);

// Fast Walsh-Hadamard Transform (equivalent, O(N log N))
let fast = fast_walsh_hadamard_transform(&signal);

// Power spectrum (all values non-negative)
let spectrum = walsh_power_spectrum(&signal, &h);

// Inverse transform (perfect reconstruction)
let recovered = inverse_walsh_transform(&transformed, &h);

// Sequency-ordered matrix (rows sorted by "frequency")
let w = sequency_order(&h);
```

## Mathematical Background

### Hadamard Matrices

A Hadamard matrix H of order n has entries ±1 and satisfies:

```
H × Hᵀ = n × I
```

This orthogonality property makes Hadamard matrices ideal for signal transforms.
The Sylvester construction builds H(2k) from H(k):

```
H(2k) = | H(k)   H(k)  |
        | H(k)  -H(k)  |
```

### Walsh Functions

Walsh functions are the rows of a Hadamard matrix, ordered by sequency (the number
of sign changes). Sequency is the square-wave analog of frequency:

- Sequency 0: constant (like DC)
- Sequency 1: one sign change (like half a period)
- Higher sequency: more oscillations

### Fast Walsh-Hadamard Transform

The FWHT decomposes the full N×N matrix multiplication into log₂(N) stages of
butterfly operations, reducing complexity from O(N²) to O(N log N). This is
analogous to the FFT for Fourier analysis.

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Ternary` | Enum with variants `Neg`, `Zero`, `Pos` representing {-1, 0, +1} |

### Functions

| Function | Description |
|----------|-------------|
| `hadamard_matrix(n)` | Generate Sylvester Hadamard matrix of order n (power of 2) |
| `walsh_transform(signal, matrix)` | Naive O(N²) Walsh-Hadamard transform |
| `inverse_walsh_transform(transformed, matrix)` | Inverse Walsh-Hadamard transform |
| `walsh_power_spectrum(signal, matrix)` | Compute squared magnitude spectrum |
| `fast_walsh_hadamard_transform(signal)` | O(N log N) butterfly FWHT |
| `inverse_fast_walsh_hadamard_transform(transformed)` | Inverse FWHT |
| `sequency_order(matrix)` | Reorder Hadamard rows by zero-crossing count |
| `walsh_ordered_matrix(n)` | Direct sequency-ordered Hadamard matrix |

## Properties Verified by Tests

- **Orthogonality**: H × Hᵀ = N × I for all generated matrices
- **Entries ±1**: All Hadamard matrix entries are exactly ±1
- **Determinant**: det(H) = ±N^(N/2)
- **Round-trip**: Forward then inverse transform recovers the original signal
- **Fast = Naive**: FWHT produces identical results to matrix multiplication
- **Power spectrum non-negative**: All power values ≥ 0
- **Parseval's theorem**: Sum of power spectrum equals signal energy
- **Linearity**: Transform(a + b) = Transform(a) + Transform(b)
- **Sequency ordering**: Rows are sorted by non-decreasing zero-crossing count

## License

MIT
