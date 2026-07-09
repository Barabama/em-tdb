use std::fmt::Write;

/// SGTE expression template: `+A+B*T+C*T*LN(T)+D*T**2+E*T**3+F*T**(-1)`
const FORMULA: &str = "+A+B*T+C*T*LN(T)+D*T**2+E*T**3+F*T**(-1)";

/// Format a float in Python-compatible `+d.ddddddE+dd` notation.
///
/// Python's `:+E` produces output like `+1.234500E+05` or `-5.678900E+01`.
/// Rust's built-in `{:+.6E}` omits the `+` sign on the exponent and
/// doesn't zero-pad: `+1.234500E5`.  This function matches Python's format
/// so cross-validation snapshots agree.
fn format_coeff(c: f64) -> String {
    // Decompose into mantissa and exponent
    let abs_c = c.abs();
    if abs_c == 0.0 || c.is_nan() || c.is_infinite() {
        return format!("{:+.6E}", c);
    }
    // floor(log10) gives the exponent for normalized scientific notation
    let exp = abs_c.log10().floor() as i32;
    let mantissa = abs_c / 10.0_f64.powi(exp);
    // Correct for floating-point rounding near powers of 10
    let (mantissa, exp) = if mantissa >= 10.0 {
        (mantissa / 10.0, exp + 1)
    } else if mantissa < 1.0 {
        (mantissa * 10.0, exp - 1)
    } else {
        (mantissa, exp)
    };
    let sign = if c < 0.0 { "-" } else { "+" };
    // Format mantissa with 6 decimal places
    let mut mant_str = String::new();
    write!(&mut mant_str, "{:.6}", mantissa).unwrap();
    format!("{}{}E{:+03}", sign, mant_str, exp)
}

/// Format [A,B,C,D,E,F] coefficients into an SGTE expression string.
///
/// Example output:
/// `+1.234567E+05-5.678901E+01*T+1.234567E+01*T*LN(T)-9.876543E-03*T**2+1.234567E-06*T**3-4.321098E+04*T**(-1)`
pub fn format_expression(params: &[f64; 6]) -> String {
    FORMULA
        .replace("+A", &format_coeff(params[0]))
        .replace("+B", &format_coeff(params[1]))
        .replace("+C", &format_coeff(params[2]))
        .replace("+D", &format_coeff(params[3]))
        .replace("+E", &format_coeff(params[4]))
        .replace("+F", &format_coeff(params[5]))
}

/// Format a normalised metric for TDB output, stripping trailing zeros.
///
/// `0.5000` → `"0.5"`, `0.2500` → `"0.25"`, `1.0000` → `"1"`
fn fmt_metric(m: f64) -> String {
    let s = format!("{:.4}", m);
    if let Some(dot) = s.find('.') {
        let trimmed = s[..dot + 1].to_string()
            + s[dot + 1..].trim_end_matches('0');
        trimmed.trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// Build a full TDB PARAMETER line.
///
/// # Format
/// `PARAMETER G({phase},{elem1}:{elem2}:...;0) 1.00 {expr}{ser_refs}; 6000.00 N !`
///
/// The `norm_metrics` should sum to 1.0 (see `crate::folder_parse::normalize_metrics`).
pub fn format_tdb_parameter(
    phase: &str,
    elements: &[String],
    norm_metrics: &[f64],
    expression: &str,
) -> String {
    let elems_str: Vec<String> = elements.iter().map(|e| e.to_uppercase()).collect();
    let elems = elems_str.join(":");
    let ser_terms: String = elements
        .iter()
        .zip(norm_metrics.iter())
        .map(|(e, &m)| format!("-{}*GHSER{}#", fmt_metric(m), e.to_uppercase()))
        .collect();
    format!(
        "PARAMETER G({},{};0) 1.00 {}{}; 6000.00 N !",
        phase.to_uppercase(),
        elems,
        expression,
        ser_terms,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_coeff_positive() {
        assert_eq!(format_coeff(1.2345e5), "+1.234500E+05");
    }

    #[test]
    fn test_format_coeff_negative() {
        assert_eq!(format_coeff(-5.6789e1), "-5.678900E+01");
    }

    #[test]
    fn test_format_coeff_small() {
        assert_eq!(format_coeff(-9.8765e-3), "-9.876500E-03");
    }

    #[test]
    fn test_format_coeff_zero() {
        let s = format_coeff(0.0);
        // Rust's built-in format for 0.0 — just check it doesn't panic and contains 'E'
        assert!(s.contains('E'));
    }

    #[test]
    fn test_format_expression() {
        let params = [1.2345e5, -5.6789e1, 1.2345e1, -9.8765e-3, 1.2345e-6, -4.3210e4];
        let s = format_expression(&params);
        assert!(s.starts_with("+"));               // starts with coefficient sign
        assert!(s.contains("E+"));                  // Python-style exponent sign
        assert!(s.contains("*T*LN(T)"));            // formula structure
        assert!(s.contains("*T**(-1)"));            // trailing term
        // Full string verification
        assert_eq!(
            s,
            "+1.234500E+05-5.678900E+01*T+1.234500E+01*T*LN(T)-9.876500E-03*T**2+1.234500E-06*T**3-4.321000E+04*T**(-1)"
        );
    }

    #[test]
    fn test_fmt_metric() {
        assert_eq!(fmt_metric(0.5), "0.5");
        assert_eq!(fmt_metric(0.25), "0.25");
        assert_eq!(fmt_metric(0.75), "0.75");
        assert_eq!(fmt_metric(1.0), "1");
        assert_eq!(fmt_metric(0.3333), "0.3333");
        assert_eq!(fmt_metric(0.0), "0");
    }

    #[test]
    fn test_format_tdb_parameter_fcc_al_ni() {
        let elems = vec!["AL".into(), "NI".into()];
        let metrics = [0.25, 0.75];
        let expr = "+1.23E+05+2.34E+01*T-3.45E+00*T*LN(T)-4.56E-03*T**2+5.67E-07*T**3-6.78E+04*T**(-1)";
        let line = format_tdb_parameter("FCC", &elems, &metrics, expr);
        assert_eq!(
            line,
            "PARAMETER G(FCC,AL:NI;0) 1.00 +1.23E+05+2.34E+01*T-3.45E+00*T*LN(T)-4.56E-03*T**2+5.67E-07*T**3-6.78E+04*T**(-1)-0.25*GHSERAL#-0.75*GHSERNI#; 6000.00 N !"
        );
    }

    #[test]
    fn test_format_tdb_parameter_ser_nb() {
        let elems = vec!["NB".into()];
        let metrics = [1.0];
        let expr = "+1.23E+05-4.56E+01*T+...";
        let line = format_tdb_parameter("SER", &elems, &metrics, expr);
        assert_eq!(
            line,
            "PARAMETER G(SER,NB;0) 1.00 +1.23E+05-4.56E+01*T+...-1*GHSERNB#; 6000.00 N !"
        );
    }

    #[test]
    fn test_format_tdb_parameter_matches_user_sample() {
        let elems = vec!["ZR".into(), "ZR".into()];
        let metrics = [0.25, 0.75];
        let expr = "-8.138924E+05+1.022378E+02*T-2.118418E+01*T*LN(T)+1.978669E-04*T**2-1.098457E-08*T**3+2.159698E+04*T**(-1)";
        let line = format_tdb_parameter("FCC", &elems, &metrics, expr);
        assert!(line.starts_with("PARAMETER G(FCC,ZR:ZR;0)"));
        assert!(line.contains("-0.25*GHSERZR#"));
        assert!(line.contains("-0.75*GHSERZR#"));
        assert!(line.ends_with("N !"));
        assert!(line.contains("6000.00"));
    }

    #[test]
    fn test_format_tdb_parameter_bcc_swap() {
        let elems = vec!["CR".into(), "FE".into()];
        let metrics = [0.5, 0.5];
        let expr = "-1.00E+05+...";
        let line = format_tdb_parameter("BCC", &elems, &metrics, expr);
        assert!(line.starts_with("PARAMETER G(BCC,CR:FE;0)"));
        assert!(line.contains("-0.5*GHSERCR#"));
        assert!(line.contains("-0.5*GHSERFE#"));
    }

    #[test]
    fn test_format_expression_zero() {
        let params = [0.0; 6];
        let s = format_expression(&params);
        assert!(s.contains('E'));
    }
}
