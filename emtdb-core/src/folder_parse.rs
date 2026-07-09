use crate::config::is_element;

/// Parse a subfolder name into phase, elements, and atom count.
///
/// Supported formats:
/// - `BCC-Al-Al-2`     → phase=BCC,  elems=[Al, Al], atom_num=2
/// - `BCC-TiNb-2`      → phase=BCC,  elems=[Ti, Nb], atom_num=2
/// - `BCC-Ta2-Al-8`    → phase=BCC,  elems=[Ta, Al], atom_num=8
/// - `SER-Nb-2atoms`   → phase=SER,  elems=[Nb],    atom_num=2
/// - `OTH-Al-Ti-Nb-4`  → phase=OTH,  elems=[Al,Ti,Nb], atom_num=4
/// - `FCC-Fe-Mn`       → phase=FCC,  elems=[Fe, Mn], atom_num=1
///
/// Returns `None` on parse failure (unknown phase token, unrecognised element, etc.).
pub fn parse_folder_name(name: &str) -> Option<(String, Vec<String>, usize)> {
    let parts: Vec<&str> = name.split('-').collect();
    if parts.len() < 2 {
        return None;
    }

    let phase = parts[0].to_uppercase();

    // Walk from the right to find an optional atom_num suffix
    let mut atom_num: usize = 1;
    let elem_end = if parts.len() > 1 {
        let last = parts[parts.len() - 1];
        if let Ok(n) = parse_atom_num(last) {
            atom_num = n;
            parts.len() - 1
        } else {
            parts.len()
        }
    } else {
        parts.len()
    };

    // Collect element tokens (the middle parts between phase and atom_num)
    let raw_tokens: Vec<&str> = parts[1..elem_end].to_vec();

    let mut elements: Vec<String> = Vec::new();
    for tok in &raw_tokens {
        let upper = tok.to_uppercase();
        if is_element(&upper) {
            elements.push(upper);
        } else {
            // Try to split concatenated token (e.g. "TiNb" → [Ti, Nb])
            let split = split_elements(tok)?;
            elements.extend(split);
        }
    }

    if elements.is_empty() {
        return None;
    }

    Some((phase, elements, atom_num))
}

/// Try to parse a suffix like `"2"`, `"8"`, `"2atoms"`, `"4atoms"` as atom count.
fn parse_atom_num(s: &str) -> Result<usize, ()> {
    let stripped = s
        .strip_suffix("atoms")
        .or_else(|| s.strip_suffix("atom"))
        .or_else(|| s.strip_suffix("ATOMS"))
        .or_else(|| s.strip_suffix("ATOM"))
        .unwrap_or(s);
    stripped.parse::<usize>().map_err(|_| ())
}

/// Split a concatenated string like "TiNb", "AlNb3", "MnW" into element symbols.
///
/// Rules:
/// 1. Each element: one uppercase letter + optional one lowercase letter
/// 2. Digits following an element are skipped (they encode ratios, not used here)
/// 3. Works on original case; each candidate is uppercased for validation
/// 4. Returns `None` if any unrecognised element code is encountered
fn split_elements(s: &str) -> Option<Vec<String>> {
    let chars: Vec<char> = s.chars().collect();
    let mut result = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        if !chars[i].is_ascii_uppercase() {
            return None;
        }

        let mut elem = String::new();
        elem.push(chars[i]);
        i += 1;

        // Optionally consume one lowercase letter (e.g. "Ti", "Nb")
        if i < chars.len() && chars[i].is_ascii_lowercase() {
            elem.push(chars[i]);
            i += 1;
        }

        // Skip any following digits (ratio markers, e.g. "Nb3" → "Nb")
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }

        let upper = elem.to_uppercase();
        if !is_element(&upper) {
            return None;
        }
        result.push(upper);
    }

    Some(result)
}

/// Normalise metrics so they sum to 1.0.
///
/// # Example
/// ```
/// use emtdb_core::folder_parse::normalize_metrics;
/// let n = normalize_metrics(&[1.0, 3.0]);
/// assert!((n[0] - 0.25).abs() < 1e-12);
/// assert!((n[1] - 0.75).abs() < 1e-12);
/// ```
pub fn normalize_metrics(metrics: &[f64]) -> Vec<f64> {
    let sum: f64 = metrics.iter().sum();
    if sum == 0.0 {
        return metrics.to_vec();
    }
    metrics.iter().map(|&m| m / sum).collect()
}

/// Generate a BCC symmetric exchange entry (swapped elements and metrics).
///
/// Returns `Some((swapped_elements, swapped_metrics))` when:
/// - exactly two elements, different symbols
/// - their metrics are equal (within float tolerance)
///
/// Returns `None` otherwise.
///
/// # Example
/// ```
/// use emtdb_core::folder_parse::gen_exchange;
/// let elems = vec!["FE".into(), "CR".into()];
/// let metrics = vec![0.5, 0.5];
/// let swapped = gen_exchange(&elems, &metrics);
/// assert!(swapped.is_some());
/// let (e, m) = swapped.unwrap();
/// assert_eq!(e, &["CR", "FE"]);
/// ```
pub fn gen_exchange(elements: &[String], metrics: &[f64]) -> Option<(Vec<String>, Vec<f64>)> {
    if elements.len() != 2
        || elements[0] == elements[1]
        || metrics.len() != 2
    {
        return None;
    }
    // Check metrics are equal within float tolerance
    if (metrics[0] - metrics[1]).abs() > 1e-12 {
        return None;
    }
    Some((
        vec![elements[1].clone(), elements[0].clone()],
        vec![metrics[1], metrics[0]],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_folder_name ──

    #[test]
    fn test_parse_basic() {
        let r = parse_folder_name("BCC-Al-Al-2").unwrap();
        assert_eq!(r.0, "BCC");
        assert_eq!(r.1, &["AL", "AL"]);
        assert_eq!(r.2, 2);
    }

    #[test]
    fn test_parse_concatenated() {
        let r = parse_folder_name("BCC-TiNb-2").unwrap();
        assert_eq!(r.0, "BCC");
        assert_eq!(r.1, &["TI", "NB"]);
        assert_eq!(r.2, 2);
    }

    #[test]
    fn test_parse_number_suffix() {
        let r = parse_folder_name("BCC-Ta2-Al-8").unwrap();
        assert_eq!(r.0, "BCC");
        assert_eq!(r.1, &["TA", "AL"]);
        assert_eq!(r.2, 8);
    }

    #[test]
    fn test_parse_ser() {
        let r = parse_folder_name("SER-Nb-2atoms").unwrap();
        assert_eq!(r.0, "SER");
        assert_eq!(r.1, &["NB"]);
        assert_eq!(r.2, 2);
    }

    #[test]
    fn test_parse_no_atom() {
        let r = parse_folder_name("FCC-Fe-Mn").unwrap();
        assert_eq!(r.0, "FCC");
        assert_eq!(r.1, &["FE", "MN"]);
        assert_eq!(r.2, 1);
    }

    #[test]
    fn test_parse_multi_elem() {
        let r = parse_folder_name("OTH-Al-Ti-Nb-4").unwrap();
        assert_eq!(r.0, "OTH");
        assert_eq!(r.1, &["AL", "TI", "NB"]);
        assert_eq!(r.2, 4);
    }

    #[test]
    fn test_parse_lowercase() {
        let r = parse_folder_name("bcc-ti-nb-4").unwrap();
        assert_eq!(r.0, "BCC");
        assert_eq!(r.1, &["TI", "NB"]);
        assert_eq!(r.2, 4);
    }

    #[test]
    fn test_parse_single_letter_elements() {
        let r = parse_folder_name("BCC-V-V-2").unwrap();
        assert_eq!(r.0, "BCC");
        assert_eq!(r.1, &["V", "V"]);
    }

    #[test]
    fn test_parse_vanadium() {
        let r = parse_folder_name("SER-V-1").unwrap();
        assert_eq!(r.0, "SER");
        assert_eq!(r.1, &["V"]);
    }

    #[test]
    fn test_parse_mixed_case() {
        let r = parse_folder_name("BCC-FeCr-2").unwrap();
        assert_eq!(r.1, &["FE", "CR"]);
    }

    #[test]
    fn test_parse_invalid_phase_only() {
        assert!(parse_folder_name("BCC").is_none());
    }

    #[test]
    fn test_parse_unknown_element() {
        assert!(parse_folder_name("BCC-XX-YY-2").is_none());
    }

    #[test]
    fn test_parse_empty_string() {
        assert!(parse_folder_name("").is_none());
    }

    // ── normalize_metrics ──

    #[test]
    fn test_normalize_equal() {
        let n = normalize_metrics(&[1.0, 1.0]);
        assert!((n[0] - 0.5).abs() < 1e-12);
        assert!((n[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_fcc() {
        let n = normalize_metrics(&[1.0, 3.0]);
        assert!((n[0] - 0.25).abs() < 1e-12);
        assert!((n[1] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_ser() {
        let n = normalize_metrics(&[1.0]);
        assert!((n[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_zero_sum() {
        let n = normalize_metrics(&[0.0, 0.0]);
        assert_eq!(n.len(), 2);
        assert!(n[0].is_nan() || n[0] == 0.0);
    }

    // ── gen_exchange ──

    #[test]
    fn test_gen_exchange_bcc() {
        let elems = vec!["FE".into(), "CR".into()];
        let metrics = vec![0.5, 0.5];
        let (e, m) = gen_exchange(&elems, &metrics).unwrap();
        assert_eq!(e, &["CR", "FE"]);
        assert!((m[0] - 0.5).abs() < 1e-12);
        assert!((m[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_gen_exchange_same_element() {
        let elems = vec!["AL".into(), "AL".into()];
        let metrics = vec![0.5, 0.5];
        assert!(gen_exchange(&elems, &metrics).is_none());
    }

    #[test]
    fn test_gen_exchange_unequal_metrics() {
        let elems = vec!["FE".into(), "CR".into()];
        let metrics = vec![0.25, 0.75];
        assert!(gen_exchange(&elems, &metrics).is_none());
    }

    #[test]
    fn test_gen_exchange_single_elem() {
        let elems = vec!["FE".into()];
        let metrics = vec![1.0];
        assert!(gen_exchange(&elems, &metrics).is_none());
    }
}
