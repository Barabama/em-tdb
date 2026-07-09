/// Faraday constant (C/mol)
pub const F_CONST: f64 = 96485.0;

/// Temperature range for fitting
pub const T_MIN: f64 = 100.0;
pub const T_MAX: f64 = 2900.0;

/// Default sublattice stoichiometries per phase.
/// Batch mode uses these to fill `--metrics` when `--phase` matches a key.
pub fn phase_metrics(phase: &str) -> Option<&'static [f64]> {
    match phase {
        "SER" => Some(&[1.0]),
        "BCC" => Some(&[1.0, 1.0]),
        "FCC" => Some(&[1.0, 3.0]),
        "HCP" => Some(&[2.0, 6.0]),
        _ => None,
    }
}

/// Known element symbols (standard + VA for vacancies).
/// Sorted for binary search via `is_element`.
const ELEMENTS: &[&str] = &[
    "AC", "AG", "AL", "AM", "AR", "AS", "AT", "AU", "B", "BA", "BE", "BH", "BI", "BK", "BR",
    "C", "CA", "CD", "CE", "CF", "CL", "CM", "CN", "CO", "CR", "CS", "CU", "DB", "DS", "DY",
    "ER", "ES", "EU", "F", "FE", "FL", "FM", "FR", "GA", "GD", "GE", "H", "HE", "HF", "HG",
    "HO", "HS", "I", "IN", "IR", "K", "KR", "LA", "LI", "LR", "LU", "LV", "MC", "MD", "MG",
    "MN", "MO", "MT", "N", "NA", "NB", "ND", "NE", "NH", "NI", "NO", "NP", "O", "OG", "OS",
    "P", "PA", "PB", "PD", "PM", "PO", "PR", "PT", "PU", "RA", "RB", "RE", "RF", "RG", "RH",
    "RN", "RU", "S", "SB", "SC", "SE", "SG", "SI", "SM", "SN", "SR", "TA", "TB", "TC", "TE",
    "TH", "TI", "TL", "TM", "TS", "U", "V", "VA", "W", "XE", "Y", "YB", "ZN", "ZR",
];

/// Check whether a string is a known element symbol (case-sensitive, uppercase).
pub fn is_element(s: &str) -> bool {
    ELEMENTS.binary_search(&s).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f_const() {
        assert!((F_CONST - 96485.0).abs() < 1.0);
    }

    #[test]
    fn test_t_min_max() {
        assert!(T_MIN < T_MAX);
    }

    #[test]
    fn test_phase_metrics_known() {
        let bcc = phase_metrics("BCC").unwrap();
        assert_eq!(bcc, &[1.0, 1.0]);
        let fcc = phase_metrics("FCC").unwrap();
        assert_eq!(fcc, &[1.0, 3.0]);
        let ser = phase_metrics("SER").unwrap();
        assert_eq!(ser, &[1.0]);
    }

    #[test]
    fn test_phase_metrics_unknown() {
        assert!(phase_metrics("OTH").is_none());
        assert!(phase_metrics("XYZ").is_none());
    }

    #[test]
    fn test_is_element_common() {
        assert!(is_element("FE"));
        assert!(is_element("AL"));
        assert!(is_element("CR"));
        assert!(is_element("NI"));
        assert!(is_element("TI"));
        assert!(is_element("V"));
        assert!(is_element("W"));
        assert!(is_element("VA"));
    }

    #[test]
    fn test_is_element_rejects() {
        assert!(!is_element("AA"));
        assert!(!is_element("ZZ"));
        assert!(!is_element(""));
        assert!(!is_element("ABC"));
        assert!(!is_element("fe")); // lowercase
    }
}
