use clap::{Parser, Subcommand};
use std::path::Path;

use emtdb_core::{
    fit, gen_exchange, normalize_metrics, parse_folder_name, phase_metrics, tdb_format::format_tdb_parameter, FitResult,
};

#[derive(Parser)]
#[command(name = "emtdb", version, about = "Gibbs–temperature SGTE polynomial fitter")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fit a single dataset
    Fit {
        /// Path to .dat file or folder containing gibbs-temperature.dat
        #[arg(long, required = true)]
        filepath: String,

        /// Phase name (e.g. BCC, FCC, SER)
        #[arg(long, required = true)]
        phase: String,

        /// Element symbols (e.g. Fe Cr)
        #[arg(long, required = true, num_args = 1..)]
        elem: Vec<String>,

        /// Stoichiometry ratios (e.g. 1 3 for FCC)
        #[arg(long, required = true, num_args = 1..)]
        metrics: Vec<f64>,

        /// Atoms per formula unit
        #[arg(long = "atom-num", required = true)]
        atom_num: usize,
    },
    /// Fit all subfolders matching a phase
    Batch {
        /// Root folder containing subfolders
        #[arg(long, required = true)]
        filepath: String,

        /// Phase name
        #[arg(long, required = true)]
        phase: String,

        /// Stoichiometry ratios (omit to use built-in defaults)
        #[arg(long, num_args = 1..)]
        metrics: Option<Vec<f64>>,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Fit {
            filepath,
            phase,
            elem,
            metrics,
            atom_num,
        } => cmd_fit(&filepath, &phase, &elem, &metrics, atom_num),
        Command::Batch {
            filepath,
            phase,
            metrics,
        } => cmd_batch(&filepath, &phase, metrics.as_deref()),
    }
}

/// Locate a gibbs-temperature.dat file from a path.
/// If `path` is a file, return it directly.
/// If `path` is a directory, recurse for the first match.
fn find_dat(path: &Path) -> Option<std::path::PathBuf> {
    if path.is_file() {
        return Some(path.to_path_buf());
    }
    if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() && p.file_name().unwrap_or_default() == "gibbs-temperature.dat" {
                    return Some(p);
                }
                if p.is_dir() {
                    if let Some(found) = find_dat(&p) {
                        return Some(found);
                    }
                }
            }
        }
    }
    None
}

fn cmd_fit(
    filepath: &str,
    phase: &str,
    elem: &[String],
    metrics: &[f64],
    atom_num: usize,
) {
    let phase_upper = phase.to_uppercase();
    let elements: Vec<String> = elem.iter().map(|e| e.to_uppercase()).collect();
    let metrics_norm = normalize_metrics(metrics);

    let dat_path = find_dat(Path::new(filepath)).unwrap_or_else(|| {
        eprintln!("[ERROR] No gibbs-temperature.dat found in or under {filepath}");
        std::process::exit(1);
    });

    println!("Folder:    {filepath}");
    println!("Dat file:  {}", dat_path.display());
    println!("Phase:     {phase_upper}");
    println!("Elements:  {}", elements.join(","));
    println!("Metrics:   {}", metrics_norm.iter().map(|m| format!("{m:.4}")).collect::<Vec<_>>().join(", "));
    println!("Atom num:  {atom_num}");

    let result = fit::fit(dat_path.to_str().unwrap(), atom_num)
        .unwrap_or_else(|e| { eprintln!("[ERROR] {e}"); std::process::exit(1); });

    println!();
    print_result(&result, &phase_upper, &elements, &metrics_norm);

    // BCC symmetric exchange
    if let Some((swapped, swapped_m)) = gen_exchange(&elements, metrics) {
        println!();
        print_result(&result, &phase_upper, &swapped, &swapped_m);
    }
}

fn cmd_batch(filepath: &str, phase: &str, cli_metrics: Option<&[f64]>) {
    let phase_upper = phase.to_uppercase();
    let metrics_raw: Vec<f64> = match cli_metrics {
        Some(m) => m.to_vec(),
        None => phase_metrics(&phase_upper)
            .map(|m| m.to_vec())
            .unwrap_or_else(|| {
                eprintln!("[ERROR] Unknown phase {phase_upper:?} — please provide --metrics");
                std::process::exit(1);
            }),
    };
    let metrics_norm = normalize_metrics(&metrics_raw);

    let root = Path::new(filepath);
    if !root.is_dir() {
        eprintln!("[ERROR] {filepath} is not a directory");
        std::process::exit(1);
    }

    println!("Phase:   {phase_upper}");
    println!("Metrics: {}", metrics_norm.iter().map(|m| format!("{m:.4}")).collect::<Vec<_>>().join(", "));
    println!("Root:    {filepath}");
    println!();

    let mut results: Vec<(FitResult, String, Vec<String>, Vec<f64>)> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();

    let dir_entries = std::fs::read_dir(root).unwrap_or_else(|e| {
        eprintln!("[ERROR] Cannot read {filepath}: {e}");
        std::process::exit(1);
    });

    for entry in dir_entries.flatten() {
        let subdir = entry.path();
        if !subdir.is_dir() {
            continue;
        }
        let name = subdir.file_name().unwrap().to_string_lossy().to_string();

        let parsed = match parse_folder_name(&name) {
            Some(p) => p,
            None => { skipped.push((name, "cannot parse folder name".into())); continue; }
        };

        if parsed.0 != phase_upper { continue; }
        if parsed.1.len() != metrics_norm.len() {
            skipped.push((name, format!("elems ({}) != metrics ({})", parsed.1.len(), metrics_norm.len())));
            continue;
        }

        let dat = match find_dat(&subdir) {
            Some(p) => p,
            None => { skipped.push((name, "no gibbs-temperature.dat".into())); continue; }
        };

        match fit::fit(dat.to_str().unwrap(), parsed.2) {
            Ok(r) => {
                results.push((r.clone(), name.clone(), parsed.1.clone(), metrics_norm.clone()));
                if let Some((swapped, swapped_m)) = gen_exchange(&parsed.1, &metrics_raw) {
                    let swapped_m_norm = normalize_metrics(&swapped_m);
                    results.push((r, format!("{name}-ex"), swapped, swapped_m_norm));
                }
            }
            Err(e) => skipped.push((name, e)),
        }
    }

    println!("Fitted: {} result(s)", results.len());
    for (r, name, elems, m) in &results {
        println!();
        println!("  {}", "─".repeat(50));
        println!("  Name:      {name}");
        println!("  Elements:  {}", elems.join(","));
        print_result(r, &phase_upper, elems, m);
    }

    if !skipped.is_empty() {
        println!("\n[WARNING] {} folder(s) skipped:", skipped.len());
        for (name, reason) in &skipped {
            println!("  - {name}: {reason}");
        }
    }
}

fn print_result(
    result: &FitResult,
    phase: &str,
    elements: &[String],
    norm_metrics: &[f64],
) {
    println!("  R² =       {:.6}", result.r2);
    println!("  G(T) =     {}", result.expression);
    let tdb = format_tdb_parameter(phase, elements, norm_metrics, &result.expression);
    println!("  TDB =      {tdb}");
}
