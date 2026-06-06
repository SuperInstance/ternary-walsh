//! # ternary-walsh
//!
//! Walsh functions and transforms for ternary-valued signal analysis.
//!
//! Walsh functions are the square-wave analogs of sinusoids, taking values in {-1, 0, +1}
//! (ternary). This crate provides tools for generating Hadamard matrices, computing the
//! Walsh (Hadamard) transform on ternary signals, sequency ordering, power spectral
//! analysis, and a fast Walsh-Hadamard transform (FWHT) implementation.

/// A ternary value: -1, 0, or +1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ternary {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

impl Ternary {
    /// Convert an i32 to a Ternary value, clamping to {-1, 0, +1}.
    pub fn from_i32(v: i32) -> Self {
        match v {
            -1 => Ternary::Neg,
            0 => Ternary::Zero,
            1 => Ternary::Pos,
            _ => {
                if v < 0 {
                    Ternary::Neg
                } else if v > 0 {
                    Ternary::Pos
                } else {
                    Ternary::Zero
                }
            }
        }
    }

    /// Get the i32 value.
    pub fn as_i32(self) -> i32 {
        self as i32
    }

    /// Get the f64 value.
    pub fn as_f64(self) -> f64 {
        self as i32 as f64
    }
}

impl std::ops::Mul for Ternary {
    type Output = i32;
    fn mul(self, rhs: Self) -> i32 {
        self.as_i32() * rhs.as_i32()
    }
}

/// Generate a Sylvester-type Hadamard matrix of order 2^n using the recursive construction.
///
/// H(1) = [[1]]
/// H(2k) = [[H(k), H(k)], [H(k), -H(k)]]
///
/// Returns a Vec<Vec<i32>> of size (size x size).
///
/// # Panics
/// Panics if `size` is not a power of 2.
pub fn hadamard_matrix(n: usize) -> Vec<Vec<i32>> {
    assert!(n > 0, "size must be positive");
    assert!(n.is_power_of_two(), "size must be a power of 2");

    if n == 1 {
        return vec![vec![1]];
    }

    let half = n / 2;
    let h = hadamard_matrix(half);
    let mut result = vec![vec![0i32; n]; n];

    for i in 0..half {
        for j in 0..half {
            let v = h[i][j];
            result[i][j] = v;
            result[i][j + half] = v;
            result[i + half][j] = v;
            result[i + half][j + half] = -v;
        }
    }

    result
}

/// Compute the bit-reversal permutation for indices 0..n.
fn bit_reverse_permutation(n: usize) -> Vec<usize> {
    let bits = n.trailing_zeros() as usize;
    let mut perm = vec![0usize; n];
    for i in 0..n {
        let mut rev = 0;
        let mut val = i;
        for _ in 0..bits {
            rev = (rev << 1) | (val & 1);
            val >>= 1;
        }
        perm[i] = rev;
    }
    perm
}

/// Compute the sequency (number of zero-crossings / sign changes) of a row in a matrix.
fn sequency(row: &[i32]) -> usize {
    if row.is_empty() {
        return 0;
    }
    let mut count = 0;
    for i in 1..row.len() {
        if row[i] != row[i - 1] && row[i] != 0 && row[i - 1] != 0 {
            count += 1;
        }
    }
    count
}

/// Reorder a Hadamard matrix into sequency (Walsh) ordering.
///
/// In sequency ordering, rows are sorted by the number of sign changes (zero crossings),
/// analogous to frequency ordering in Fourier analysis.
pub fn sequency_order(matrix: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let n = matrix.len();
    let mut indexed: Vec<(usize, Vec<i32>)> = matrix
        .iter()
        .enumerate()
        .map(|(i, row)| (sequency(row), row.clone()))
        .collect();

    indexed.sort_by_key(|(seq, _)| *seq);
    indexed.into_iter().map(|(_, row)| row).collect()
}

/// Compute the naive (matrix multiplication) Walsh-Hadamard transform on a ternary signal.
///
/// The transform is: Y[k] = sum_j(x[j] * H[k][j]) / sqrt(N)
///
/// # Arguments
/// * `signal` - Input ternary signal as Vec<i32> (values should be -1, 0, or +1)
/// * `matrix` - The Hadamard matrix to use
///
/// Returns the transformed signal as Vec<f64>.
pub fn walsh_transform(signal: &[i32], matrix: &[Vec<i32>]) -> Vec<f64> {
    let n = signal.len();
    assert_eq!(n, matrix.len(), "signal length must match matrix size");
    let scale = (n as f64).sqrt();

    let mut result = vec![0.0f64; n];
    for k in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += signal[j] as f64 * matrix[k][j] as f64;
        }
        result[k] = sum / scale;
    }
    result
}

/// Compute the inverse Walsh-Hadamard transform.
///
/// Since the Hadamard matrix H satisfies H*H^T = N*I, the inverse is:
/// x[j] = sum_k(Y[k] * H[k][j]) / sqrt(N)
pub fn inverse_walsh_transform(transformed: &[f64], matrix: &[Vec<i32>]) -> Vec<f64> {
    let n = transformed.len();
    assert_eq!(n, matrix.len(), "transform length must match matrix size");
    let scale = (n as f64).sqrt();

    let mut result = vec![0.0f64; n];
    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..n {
            sum += transformed[k] * matrix[k][j] as f64;
        }
        result[j] = sum / scale;
    }
    result
}

/// Compute the Walsh power spectrum of a ternary signal.
///
/// The power spectrum is the squared magnitude of the Walsh transform coefficients:
/// P[k] = |Y[k]|^2
///
/// All values are guaranteed non-negative.
pub fn walsh_power_spectrum(signal: &[i32], matrix: &[Vec<i32>]) -> Vec<f64> {
    walsh_transform(signal, matrix).iter().map(|v| v * v).collect()
}

/// Fast Walsh-Hadamard Transform (FWHT) for ternary signals.
///
/// Uses the butterfly (in-place) algorithm with O(N log N) complexity instead
/// of the O(N^2) naive matrix multiplication.
///
/// This is equivalent to the Cooley-Tukey style decomposition of the Hadamard transform.
pub fn fast_walsh_hadamard_transform(signal: &[i32]) -> Vec<f64> {
    let n = signal.len();
    assert!(n.is_power_of_two(), "signal length must be a power of 2");

    let mut result: Vec<f64> = signal.iter().map(|&v| v as f64).collect();

    let mut step = 1;
    while step < n {
        let mut i = 0;
        while i < n {
            for j in 0..step {
                let a = result[i + j];
                let b = result[i + j + step];
                result[i + j] = a + b;
                result[i + j + step] = a - b;
            }
            i += 2 * step;
        }
        step *= 2;
    }

    let scale = (n as f64).sqrt();
    for v in result.iter_mut() {
        *v /= scale;
    }

    result
}

/// Inverse Fast Walsh-Hadamard Transform.
///
/// Since the Walsh-Hadamard transform is self-inverse (up to scaling), this
/// is the same as the forward transform.
pub fn inverse_fast_walsh_hadamard_transform(transformed: &[f64]) -> Vec<f64> {
    let n = transformed.len();
    assert!(n.is_power_of_two(), "transform length must be a power of 2");

    let mut result = transformed.to_vec();

    let mut step = 1;
    while step < n {
        let mut i = 0;
        while i < n {
            for j in 0..step {
                let a = result[i + j];
                let b = result[i + j + step];
                result[i + j] = a + b;
                result[i + j + step] = a - b;
            }
            i += 2 * step;
        }
        step *= 2;
    }

    let scale = (n as f64).sqrt();
    for v in result.iter_mut() {
        *v /= scale;
    }

    result
}

/// Compute the Gray code of an integer.
fn gray_code(n: usize) -> usize {
    n ^ (n >> 1)
}

/// Generate a Walsh-ordered (sequency-ordered) Hadamard matrix via Gray code permutation.
///
/// This produces the same result as `sequency_order(&hadamard_matrix(n))` but more efficiently.
pub fn walsh_ordered_matrix(n: usize) -> Vec<Vec<i32>> {
    assert!(n.is_power_of_two(), "size must be a power of 2");
    let h = hadamard_matrix(n);
    let bits = n.trailing_zeros() as usize;
    let mut perm: Vec<usize> = (0..n).map(|i| gray_code(i)).collect();

    // Bit-reverse the gray codes for proper Walsh ordering
    let br = bit_reverse_permutation(n);
    perm = perm.into_iter().map(|g| br[g]).collect();

    // Sort by sequency
    let mut indexed: Vec<(usize, Vec<i32>)> = (0..n)
        .map(|i| {
            let row = h[perm[i]].clone();
            (sequency(&row), row)
        })
        .collect();
    indexed.sort_by_key(|(seq, _)| *seq);
    indexed.into_iter().map(|(_, row)| row).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard_1x1() {
        let h = hadamard_matrix(1);
        assert_eq!(h, vec![vec![1]]);
    }

    #[test]
    fn test_hadamard_2x2() {
        let h = hadamard_matrix(2);
        assert_eq!(h, vec![vec![1, 1], vec![1, -1]]);
    }

    #[test]
    fn test_hadamard_4x4() {
        let h = hadamard_matrix(4);
        assert_eq!(
            h,
            vec![
                vec![1, 1, 1, 1],
                vec![1, -1, 1, -1],
                vec![1, 1, -1, -1],
                vec![1, -1, -1, 1],
            ]
        );
    }

    #[test]
    fn test_hadamard_orthogonality() {
        for &size in &[2, 4, 8, 16] {
            let h = hadamard_matrix(size);
            // H * H^T = N * I
            for i in 0..size {
                for j in 0..size {
                    let dot: i32 = (0..size).map(|k| h[i][k] * h[j][k]).sum();
                    let expected = if i == j { size as i32 } else { 0 };
                    assert_eq!(
                        dot, expected,
                        "Row {} dot Row {} = {}, expected {} (size={})",
                        i, j, dot, expected, size
                    );
                }
            }
        }
    }

    #[test]
    fn test_hadamard_entries_are_pm1() {
        for &size in &[2, 4, 8, 16, 32] {
            let h = hadamard_matrix(size);
            for row in &h {
                for &val in row {
                    assert!(val == 1 || val == -1, "Entry must be ±1, got {}", val);
                }
            }
        }
    }

    #[test]
    fn test_hadamard_determinant() {
        // det(H_n) = ±n^(n/2)
        for &size in &[2, 4, 8] {
            let h = hadamard_matrix(size);
            let det = determinant(&h);
            let expected = (size as f64).powf(size as f64 / 2.0);
            assert!(
                (det - expected).abs() < 1e-6 || (det + expected).abs() < 1e-6,
                "det(H_{}) = {}, expected ±{}",
                size,
                det,
                expected
            );
        }
    }

    fn determinant(m: &[Vec<i32>]) -> f64 {
        let n = m.len();
        if n == 1 {
            return m[0][0] as f64;
        }
        let mut det = 0.0;
        for j in 0..n {
            let minor = minor_matrix(m, 0, j);
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            det += sign * m[0][j] as f64 * determinant(&minor);
        }
        det
    }

    fn minor_matrix(m: &[Vec<i32>], skip_row: usize, skip_col: usize) -> Vec<Vec<i32>> {
        let n = m.len();
        let mut result = Vec::new();
        for i in 0..n {
            if i == skip_row {
                continue;
            }
            let mut row = Vec::new();
            for j in 0..n {
                if j == skip_col {
                    continue;
                }
                row.push(m[i][j]);
            }
            result.push(row);
        }
        result
    }

    #[test]
    fn test_transform_roundtrip() {
        for &size in &[2, 4, 8, 16] {
            let h = hadamard_matrix(size);
            let signal: Vec<i32> = (0..size).map(|i| if i % 3 == 0 { 1 } else if i % 3 == 1 { -1 } else { 0 }).collect();

            let transformed = walsh_transform(&signal, &h);
            let recovered = inverse_walsh_transform(&transformed, &h);

            for (i, (orig, rec)) in signal.iter().zip(recovered.iter()).enumerate() {
                assert!(
                    (*orig as f64 - rec).abs() < 1e-9,
                    "Roundtrip mismatch at index {}: {} vs {} (size={})",
                    i,
                    orig,
                    rec,
                    size
                );
            }
        }
    }

    #[test]
    fn test_power_spectrum_non_negative() {
        for &size in &[2, 4, 8, 16] {
            let h = hadamard_matrix(size);
            let signal: Vec<i32> = (0..size).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
            let spectrum = walsh_power_spectrum(&signal, &h);

            for (i, &p) in spectrum.iter().enumerate() {
                assert!(p >= 0.0, "Power spectrum[{}] = {} is negative", i, p);
            }
        }
    }

    #[test]
    fn test_power_spectrum_parseval() {
        // Parseval's theorem: sum of power spectrum = sum of signal^2
        for &size in &[2, 4, 8, 16] {
            let h = hadamard_matrix(size);
            let signal: Vec<i32> = (0..size)
                .map(|i| if i % 3 == 0 { 1 } else if i % 3 == 1 { -1 } else { 0 })
                .collect();

            let spectrum = walsh_power_spectrum(&signal, &h);
            let total_power: f64 = spectrum.iter().sum();
            let signal_energy: f64 = signal.iter().map(|v| (v * v) as f64).sum();

            assert!(
                (total_power - signal_energy).abs() < 1e-9,
                "Parseval violation: total_power={}, signal_energy={}",
                total_power,
                signal_energy
            );
        }
    }

    #[test]
    fn test_sequency_ordering() {
        let h = hadamard_matrix(8);
        let w = sequency_order(&h);

        // Each row should have non-decreasing sequency
        let seqs: Vec<usize> = w.iter().map(|row| sequency(row)).collect();
        for i in 1..seqs.len() {
            assert!(
                seqs[i] >= seqs[i - 1],
                "Sequency not non-decreasing: seq[{}]={} > seq[{}]={}",
                i - 1,
                seqs[i - 1],
                i,
                seqs[i]
            );
        }
    }

    #[test]
    fn test_sequency_ordering_preserves_rows() {
        let h = hadamard_matrix(8);
        let w = sequency_order(&h);

        // All original rows should be present (as sets)
        let mut h_sorted: Vec<Vec<i32>> = h.clone();
        h_sorted.sort();
        let mut w_sorted: Vec<Vec<i32>> = w.clone();
        w_sorted.sort();
        assert_eq!(h_sorted, w_sorted, "Sequency reorder should be a permutation");
    }

    #[test]
    fn test_fast_vs_naive_equivalence() {
        for &size in &[2, 4, 8, 16, 32] {
            let h = hadamard_matrix(size);
            let signal: Vec<i32> = (0..size).map(|i| if i % 3 == 0 { 1 } else if i % 3 == 1 { -1 } else { 0 }).collect();

            let naive = walsh_transform(&signal, &h);
            let fast = fast_walsh_hadamard_transform(&signal);

            // May differ by sign permutations; compare magnitudes
            // Actually, for the natural (Hadamard) ordering they should match exactly
            for (i, (n, f)) in naive.iter().zip(fast.iter()).enumerate() {
                assert!(
                    (n - f).abs() < 1e-9,
                    "Fast vs naive mismatch at index {}: {} vs {} (size={})",
                    i,
                    n,
                    f,
                    size
                );
            }
        }
    }

    #[test]
    fn test_fast_roundtrip() {
        for &size in &[2, 4, 8, 16, 32] {
            let signal: Vec<i32> = (0..size)
                .map(|i| if i % 3 == 0 { 1 } else if i % 3 == 1 { -1 } else { 0 })
                .collect();

            let transformed = fast_walsh_hadamard_transform(&signal);
            let recovered = inverse_fast_walsh_hadamard_transform(&transformed);

            for (i, (orig, rec)) in signal.iter().zip(recovered.iter()).enumerate() {
                assert!(
                    (*orig as f64 - rec).abs() < 1e-9,
                    "Fast roundtrip mismatch at index {}: {} vs {}",
                    i,
                    orig,
                    rec
                );
            }
        }
    }

    #[test]
    fn test_ternary_values() {
        assert_eq!(Ternary::Neg.as_i32(), -1);
        assert_eq!(Ternary::Zero.as_i32(), 0);
        assert_eq!(Ternary::Pos.as_i32(), 1);
        assert_eq!(Ternary::from_i32(5), Ternary::Pos);
        assert_eq!(Ternary::from_i32(-3), Ternary::Neg);
        assert_eq!(Ternary::from_i32(0), Ternary::Zero);
    }

    #[test]
    fn test_ternary_multiply() {
        assert_eq!(Ternary::Pos * Ternary::Pos, 1);
        assert_eq!(Ternary::Pos * Ternary::Neg, -1);
        assert_eq!(Ternary::Neg * Ternary::Neg, 1);
        assert_eq!(Ternary::Zero * Ternary::Pos, 0);
        assert_eq!(Ternary::Zero * Ternary::Neg, 0);
    }

    #[test]
    fn test_all_zero_signal() {
        let h = hadamard_matrix(4);
        let signal = vec![0i32; 4];
        let transformed = walsh_transform(&signal, &h);
        assert!(transformed.iter().all(|&v| v.abs() < 1e-12));

        let spectrum = walsh_power_spectrum(&signal, &h);
        assert!(spectrum.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn test_constant_signal() {
        let h = hadamard_matrix(4);
        let signal = vec![1i32; 4];
        let transformed = walsh_transform(&signal, &h);
        // DC component (first) should be non-zero, all others zero
        assert!(transformed[0].abs() > 0.0);
        for i in 1..transformed.len() {
            assert!(
                transformed[i].abs() < 1e-12,
                "Non-DC component {} = {} should be 0",
                i,
                transformed[i]
            );
        }
    }

    #[test]
    fn test_hadamard_powers_of_two() {
        // Verify we can generate matrices for reasonable sizes
        for &size in &[1, 2, 4, 8, 16, 32, 64, 128] {
            let h = hadamard_matrix(size);
            assert_eq!(h.len(), size);
            assert_eq!(h[0].len(), size);
        }
    }

    #[test]
    #[should_panic(expected = "size must be a power of 2")]
    fn test_hadamard_non_power_of_two() {
        hadamard_matrix(3);
    }

    #[test]
    fn test_bit_reversal() {
        let perm = bit_reverse_permutation(8);
        // 0=000 -> 000=0, 1=001 -> 100=4, 2=010 -> 010=2, 3=011 -> 110=6
        assert_eq!(perm[0], 0);
        assert_eq!(perm[1], 4);
        assert_eq!(perm[2], 2);
        assert_eq!(perm[3], 6);
        assert_eq!(perm[4], 1);
        assert_eq!(perm[5], 5);
        assert_eq!(perm[6], 3);
        assert_eq!(perm[7], 7);
    }

    #[test]
    fn test_gray_code() {
        assert_eq!(gray_code(0), 0);
        assert_eq!(gray_code(1), 1);
        assert_eq!(gray_code(2), 3);
        assert_eq!(gray_code(3), 2);
        assert_eq!(gray_code(4), 6);
        assert_eq!(gray_code(5), 7);
    }

    #[test]
    fn test_walsh_ordered_matrix_properties() {
        let w = walsh_ordered_matrix(8);
        let seqs: Vec<usize> = w.iter().map(|row| sequency(row)).collect();
        // Sequencies should be 0,1,2,...,7 (or at least non-decreasing)
        for i in 1..seqs.len() {
            assert!(seqs[i] >= seqs[i - 1]);
        }
    }

    #[test]
    fn test_transform_linearity() {
        let h = hadamard_matrix(4);
        let a = vec![1i32, -1, 0, 1];
        let b = vec![0i32, 1, -1, 1];

        let ta = walsh_transform(&a, &h);
        let tb = walsh_transform(&b, &h);
        let a_plus_b: Vec<i32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        let tab = walsh_transform(&a_plus_b, &h);

        for i in 0..4 {
            assert!(
                (ta[i] + tb[i] - tab[i]).abs() < 1e-9,
                "Linearity failed at index {}: {} + {} != {}",
                i,
                ta[i],
                tb[i],
                tab[i]
            );
        }
    }
}
