//! Bivariate polynomial arithmetic in F_p[X][Y] for the amortized
//! evaluation-interpolation resultant (eprint 2026/150, Section 3.2.1).
//!
//! A `BiPoly` represents f(X, Y) = sum_i f_i(X) * Y^i. It is stored as a
//! `Vec<pl::Poly>` indexed by the Y-degree i; entry i is the univariate
//! coefficient polynomial f_i(X) in normal-domain F_p[X] (low-degree-first,
//! exactly the `poseidon_core::poly::Poly` convention). The zero bivariate
//! polynomial is the empty vec.
//!
//! All X-coefficient arithmetic delegates to the validated `poseidon_core::poly`
//! module so the underlying field math is byte-for-byte identical to the per-X
//! oracle: the amortization (build-once + multipoint-eval) is the only thing
//! under test here, never the field/poly kernel.
//!
//! This matches the paper's layout f(X,Y) = sum_i f_i(X) Y^i directly, so the
//! multipoint-evaluation step is just "evaluate each f_i(X) at the grid points".

use poseidon_core::poly as pl;

/// f(X, Y) = sum_i coeff[i](X) * Y^i. coeff[i] is a poly in X (F_p[X]).
pub type BiPoly = Vec<pl::Poly>;

/// Drop trailing zero (empty) Y-coefficients.
pub fn trim(f: &[pl::Poly]) -> BiPoly {
    let mut v = f.to_vec();
    while matches!(v.last(), Some(c) if c.is_empty()) {
        v.pop();
    }
    v
}

/// Y-degree (degree in Y); the zero polynomial has degree -1.
pub fn deg_y(f: &[pl::Poly]) -> i64 {
    let mut n = f.len();
    while n > 0 && f[n - 1].is_empty() {
        n -= 1;
    }
    n as i64 - 1
}

/// Maximum X-degree across all Y-coefficients (the zero polynomial yields -1).
pub fn deg_x(f: &[pl::Poly]) -> i64 {
    f.iter().map(|c| pl::deg(c)).max().unwrap_or(-1)
}

/// The constant-in-Y bivariate polynomial whose Y^0 coefficient is the given
/// X-polynomial. `from_x_poly([0,1])` is the monomial X; `from_x_poly([c])` is c.
pub fn from_x_poly(c: &[u32]) -> BiPoly {
    let c = pl::trim(c);
    if c.is_empty() {
        vec![]
    } else {
        vec![c]
    }
}

/// The bivariate polynomial that is exactly the field constant c (degree 0 in
/// both X and Y).
pub fn constant(c: u32) -> BiPoly {
    from_x_poly(&[c % poseidon_core::P])
}

/// The monomial X (degree 1 in X, degree 0 in Y).
pub fn var_x() -> BiPoly {
    vec![pl::trim(&[0, 1])]
}

/// The monomial Y (degree 0 in X, degree 1 in Y).
pub fn var_y() -> BiPoly {
    vec![vec![], pl::trim(&[1])]
}

/// a + b (coefficient-wise in Y; each Y-coefficient is an F_p[X] add).
pub fn add(a: &[pl::Poly], b: &[pl::Poly]) -> BiPoly {
    let n = a.len().max(b.len());
    let empty: pl::Poly = vec![];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ai = a.get(i).unwrap_or(&empty);
        let bi = b.get(i).unwrap_or(&empty);
        out.push(pl::add(ai, bi));
    }
    trim(&out)
}

/// a - b.
pub fn sub(a: &[pl::Poly], b: &[pl::Poly]) -> BiPoly {
    let n = a.len().max(b.len());
    let empty: pl::Poly = vec![];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ai = a.get(i).unwrap_or(&empty);
        let bi = b.get(i).unwrap_or(&empty);
        out.push(pl::sub(ai, bi));
    }
    trim(&out)
}

/// c * a for a field scalar c.
pub fn scalar(a: &[pl::Poly], c: u32) -> BiPoly {
    let c = c % poseidon_core::P;
    if c == 0 {
        return vec![];
    }
    trim(&a.iter().map(|coef| pl::scalar(coef, c)).collect::<Vec<_>>())
}

/// a * b. Convolution in Y; each coefficient product is an F_p[X] multiply
/// (delegated to `pl::mul`, which is NTT-backed for large X-degrees).
pub fn mul(a: &[pl::Poly], b: &[pl::Poly]) -> BiPoly {
    let a = trim(a);
    let b = trim(b);
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let out_len = a.len() + b.len() - 1;
    let mut out: BiPoly = vec![vec![]; out_len];
    for (i, ai) in a.iter().enumerate() {
        if ai.is_empty() {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            if bj.is_empty() {
                continue;
            }
            let prod = pl::mul(ai, bj);
            out[i + j] = pl::add(&out[i + j], &prod);
        }
    }
    trim(&out)
}

/// The S-box cube f^3 in F_p[X][Y].
pub fn cube(f: &[pl::Poly]) -> BiPoly {
    let sq = mul(f, f);
    mul(&sq, f)
}

/// Evaluate f(X, Y) at X = x, returning the univariate poly f(x, Y) in F_p[Y]
/// (a `pl::Poly` indexed by Y-degree). This is the per-grid-point reduction:
/// each Y-coefficient f_i(X) is evaluated at the scalar x.
pub fn eval_x(f: &[pl::Poly], x: u32) -> pl::Poly {
    let ys: Vec<u32> = f.iter().map(|coef| pl::eval(coef, x)).collect();
    pl::trim(&ys)
}

/// Evaluate f(X, Y) at Y = y, returning the univariate poly f(X, y) in F_p[X].
/// (Used to cross-check / back-substitute Y in the secondary witness search.)
pub fn eval_y(f: &[pl::Poly], y: u32) -> pl::Poly {
    // Horner in Y over F_p[X] coefficients.
    let mut acc: pl::Poly = vec![];
    for coef in f.iter().rev() {
        acc = pl::add(&pl::scalar(&acc, y), coef);
    }
    pl::trim(&acc)
}
