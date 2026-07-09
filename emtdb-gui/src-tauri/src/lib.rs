use std::path::Path;

use emtdb_core::fit::fit;
use emtdb_core::folder_parse::normalize_metrics;
use emtdb_core::tdb_format::format_tdb_parameter;
use serde::Serialize;

#[derive(Debug, Serialize)]
struct DatasetInfo {
    folder_name: String,
    file_path: String,
}

/// Recursively search a directory for the first `gibbs-temperature.dat`.
fn find_dat(dir: &Path) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
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
    None
}

#[tauri::command]
fn scan_folder(folder_path: String) -> Result<Vec<DatasetInfo>, String> {
    let dir = std::fs::read_dir(&folder_path)
        .map_err(|e| format!("Cannot read {}: {}", folder_path, e))?;

    let mut results = Vec::new();
    for entry in dir {
        let entry = entry.map_err(|e| format!("Entry error: {}", e))?;
        let path = entry.path();
        if path.is_dir() {
            if let Some(dat) = find_dat(&path) {
                results.push(DatasetInfo {
                    folder_name: path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string(),
                    file_path: dat.to_string_lossy().to_string(),
                });
            }
        }
    }
    Ok(results)
}

#[tauri::command]
fn run_fit(
    filepath: String,
    phase: String,
    elem: Vec<String>,
    metrics: Vec<f64>,
    atom_num: usize,
) -> Result<emtdb_core::FitResult, String> {
    let mut result = fit(&filepath, atom_num)?;

    let elements: Vec<String> = elem.iter().map(|e| e.to_uppercase()).collect();
    let norm_metrics = normalize_metrics(&metrics);

    result.tdb_parameter = format_tdb_parameter(&phase, &elements, &norm_metrics, &result.expression);

    Ok(result)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![run_fit, scan_folder])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
