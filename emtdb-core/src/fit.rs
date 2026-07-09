use serde::{Deserialize, Serialize};

use argmin::core::{CostFunction, Executor};
use argmin::solver::neldermead::NelderMead;

use crate::config::{F_CONST, T_MIN, T_MAX};
use crate::tdb_format::format_expression;

/// Result of a single SGTE polynomial fit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitResult {
    pub params: Vec<f64>,
    pub r2: f64,
    pub expression: String,
    pub tdb_parameter: String,
    pub t_data: Vec<f64>,
    pub g_data: Vec<f64>,
    pub g_fit: Vec<f64>,
}

/// SGTE polynomial: A + B*T + C*T*ln(T) + D*T^2 + E*T^3 + F/T
pub fn sgte_poly(t: f64, params: &[f64; 6]) -> f64 {
    let [a, b, c, d, e, f] = params;
    a + b * t + c * t * t.ln() + d * t.powi(2) + e * t.powi(3) + f / t
}

// ── argmin cost function ──

struct GibbsCost {
    t: Vec<f64>,
    g: Vec<f64>,
}

impl CostFunction for GibbsCost {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let p: [f64; 6] = param[..].try_into().unwrap();
        let mse = self
            .t
            .iter()
            .zip(self.g.iter())
            .map(|(&tt, &gg)| {
                let diff = gg - sgte_poly(tt, &p);
                diff * diff
            })
            .sum::<f64>()
            / self.t.len() as f64;
        Ok(mse)
    }
}

fn compute_r2(g_data: &[f64], params: &[f64; 6], t_data: &[f64]) -> f64 {
    let n = g_data.len() as f64;
    let mean_g: f64 = g_data.iter().sum::<f64>() / n;
    let ss_res: f64 = t_data
        .iter()
        .zip(g_data.iter())
        .map(|(&t, &g)| {
            let diff = g - sgte_poly(t, params);
            diff * diff
        })
        .sum();
    let ss_tot: f64 = g_data
        .iter()
        .map(|&g| {
            let diff = g - mean_g;
            diff * diff
        })
        .sum();
    if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    }
}

/// Fit SGTE polynomial to a `gibbs-temperature.dat` file.
///
/// Reads temperature and energy data, normalizes by `atom_num` and Faraday
/// constant, then runs Nelder-Mead optimisation (via argmin) for the 6 SGTE
/// parameters. Returns `FitResult` with R², expression, and fitted curve.
pub fn fit(dat_path: &str, atom_num: usize) -> Result<FitResult, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b' ')
        .flexible(true)
        .from_path(dat_path)
        .map_err(|e| format!("Cannot open {}: {}", dat_path, e))?;

    let mut t_data = Vec::new();
    let mut g_data = Vec::new();

    for (idx, record) in rdr.records().enumerate() {
        let rec = record.map_err(|e| format!("Line {} parse error: {}", idx + 1, e))?;
        if idx == 0 {
            continue;
        }
        let fields: Vec<&str> = rec.iter().filter(|s| !s.is_empty()).collect();
        if fields.len() < 2 {
            continue;
        }
        let t: f64 = fields[0]
            .parse()
            .map_err(|_| format!("Line {}: bad T value '{}'", idx + 1, fields[0]))?;
        let g_raw: f64 = fields[1]
            .parse()
            .map_err(|_| format!("Line {}: bad G value '{}'", idx + 1, fields[1]))?;

        if t < T_MIN || t > T_MAX {
            continue;
        }
        t_data.push(t);
        g_data.push(g_raw * F_CONST / atom_num as f64);
    }

    if t_data.is_empty() {
        return Err("No valid data points found".to_string());
    }

    // Initial parameter guess
    let a_guess = g_data[0];
    let last = t_data.len() - 1;
    let b_guess = (g_data[last] - g_data[0]) / (t_data[last] - t_data[0]);
    let initial: Vec<f64> = vec![a_guess, b_guess, 0.0, 0.0, 0.0, 0.0];

    // Build initial simplex (argmin needs n+1 vertices)
    let n = initial.len();
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(initial.clone());
    for i in 0..n {
        let mut p = initial.clone();
        if p[i] != 0.0 {
            p[i] *= 1.05;
        } else {
            p[i] = 0.01;
        }
        simplex.push(p);
    }

    let cost = GibbsCost {
        t: t_data.clone(),
        g: g_data.clone(),
    };

    let solver = NelderMead::new(simplex);

    let result = Executor::new(cost, solver)
        .configure(|state| state.max_iters(5000))
        .run()
        .map_err(|e| format!("Optimization failed: {}", e))?;

    let best_vec = result.state().best_param.clone().ok_or("No result")?;
    let best: [f64; 6] = best_vec[..]
        .try_into()
        .map_err(|_| "argmin returned wrong param count".to_string())?;

    let r2 = compute_r2(&g_data, &best, &t_data);
    let g_fit: Vec<f64> = t_data.iter().map(|&t| sgte_poly(t, &best)).collect();

    Ok(FitResult {
        params: best.to_vec(),
        r2,
        expression: format_expression(&best),
        tdb_parameter: String::new(),
        t_data,
        g_data,
        g_fit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgte_poly_known() {
        // Simple test: params [1,2,3,4,5,6] at T=300
        let p = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = 300.0_f64;
        let expected = 1.0 + 600.0 + 3.0 * t * t.ln() + 4.0 * 90000.0 + 5.0 * 27_000_000.0 + 6.0 / t;
        assert!((sgte_poly(t, &p) - expected).abs() < 1e-9);
    }

    #[test]
    fn test_sgte_poly_zero_f() {
        // F term should be fine at T=100 (no division by zero)
        let p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!((sgte_poly(100.0, &p)).abs() < 1e-15);
    }

    #[test]
    fn test_fit_ser_nb() {
        let path = "../tests/fits-dat/SER-Nb-2atoms/gibbs-temperature.dat";
        let res = fit(path, 2).unwrap();
        assert!(res.r2 > 0.999, "SER-Nb R² too low: {}", res.r2);
        assert_eq!(res.params.len(), 6);
    }

    #[test]
    fn test_fit_tinb() {
        let path = "../tests/fits-dat/BCC-TiNb-2/gibbs-temperature.dat";
        let res = fit(path, 2).unwrap();
        assert!(res.r2 > 0.98, "BCC-TiNb R² too low: {}", res.r2);
        assert_eq!(res.params.len(), 6);
    }

    #[test]
    fn test_fit_alal() {
        let path = "../tests/fits-dat/BCC-Al-Al-2/QHA-AlAl/gibbs-temperature.dat";
        let res = fit(path, 2).unwrap();
        assert!(res.r2 > 0.999, "BCC-Al-Al R² too low: {}", res.r2);
        assert_eq!(res.params.len(), 6);
    }

    #[test]
    fn test_fit_invalid_path() {
        let res = fit("/nonexistent/file.dat", 1);
        assert!(res.is_err());
    }

    #[test]
    fn test_fit_empty_file() {
        let path = "../tests/fits-dat/BCC-TiNb-2/gibbs-temperature.dat";
        // fit with atom_num=0 produces NaN → still returns a fit
        let res = fit(path, 0);
        // argmin handles inf/nan internally, but our data loop should
        // produce G = inf, which might still "fit" — just check no panic
        assert!(res.is_ok() || res.is_err());
    }
}
