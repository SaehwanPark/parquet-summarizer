use anyhow::{Context, Result};
use clap::Parser;
use polars::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

/// A CLI tool to summarize Parquet files with shape and statistical information
#[derive(Parser)]
#[command(name = "parquet-summarizer")]
#[command(about = "Analyze and summarize Parquet files efficiently", long_about = None)]
#[command(version)]
struct Args {
  /// Path to the parquet file to analyze
  input_file: PathBuf,

  /// Optional output file path. If not provided, prints to stdout
  #[arg(short, long)]
  output: Option<PathBuf>,

  /// Maximum number of distinct values to consider a column categorical (default: 10)
  #[arg(long, default_value_t = 10)]
  categorical_threshold: usize,

  /// Process file with reduced memory usage (limits parallelism)
  #[arg(long)]
  low_memory: bool,
}

#[derive(Debug)]
struct ColumnSummary {
  name: String,
  data_type: String,
  summary: ColumnStats,
}

#[derive(Debug)]
enum ColumnStats {
  Numerical {
    mean: Option<f64>,
    std_dev: Option<f64>,
    q25: Option<f64>,
    q75: Option<f64>,
    iqr: Option<f64>,
  },
  Categorical {
    frequency_table: Vec<(String, u32)>,
    total_unique: usize,
    showing_top_n: bool,
  },
}

fn main() -> Result<()> {
  let args = Args::parse();

  // Validate input file exists
  if !args.input_file.exists() {
    anyhow::bail!("Input file '{}' does not exist", args.input_file.display());
  }

  // Analyze the parquet file
  let summary = analyze_parquet(&args)?;

  // Generate output
  let output_text = format_summary(&summary);

  // Write to file or stdout
  match args.output {
    Some(output_path) => {
      let mut file = File::create(&output_path)
        .with_context(|| format!("Failed to create output file '{}'", output_path.display()))?;
      file
        .write_all(output_text.as_bytes())
        .with_context(|| format!("Failed to write to output file '{}'", output_path.display()))?;
      println!("Summary written to: {}", output_path.display());
    }
    None => {
      print!("{output_text}");
    }
  }

  Ok(())
}

fn analyze_parquet(args: &Args) -> Result<Vec<ColumnSummary>> {
  // Use lazy loading for efficiency with large files
  let mut scan_args = ScanArgsParquet::default();
  if args.low_memory {
    scan_args.low_memory = true;
  }

  let lazy_frame = LazyFrame::scan_parquet(&args.input_file, scan_args).with_context(|| {
    format!(
      "Failed to scan parquet file '{}'",
      args.input_file.display()
    )
  })?;

  // Collect the dataframe
  let df = lazy_frame
    .collect()
    .with_context(|| "Failed to load parquet data")?;

  let mut summaries = Vec::new();

  // Print shape information
  println!("📊 Parquet File Analysis");
  println!("━━━━━━━━━━━━━━━━━━━━━━━━━");
  println!("📁 File: {}", args.input_file.display());
  println!("📏 Shape: {} rows × {} columns", df.height(), df.width());
  println!();

  // Analyze each column
  for column_name in df.get_column_names() {
    let column = df
      .column(column_name)
      .with_context(|| format!("Failed to get column '{column_name}'"))?;

    // Convert Column to Series
    let series = column.as_materialized_series().clone();

    let data_type = series.dtype();
    let summary = analyze_column(&series, args.categorical_threshold)?;

    summaries.push(ColumnSummary {
      name: column_name.to_string(),
      data_type: format!("{data_type:?}"),
      summary,
    });
  }

  Ok(summaries)
}

fn analyze_column(column: &Series, categorical_threshold: usize) -> Result<ColumnStats> {
  let data_type = column.dtype();

  match data_type {
    // Numerical types
    DataType::UInt8
    | DataType::UInt16
    | DataType::UInt32
    | DataType::UInt64
    | DataType::Int8
    | DataType::Int16
    | DataType::Int32
    | DataType::Int64
    | DataType::Int128
    | DataType::Float32
    | DataType::Float64 => analyze_numerical_column(column),

    // String and categorical types
    DataType::String | DataType::Categorical(_, _) | DataType::Enum(_, _) => {
      analyze_categorical_column(column, categorical_threshold)
    }

    // For other types, treat as categorical if they have reasonable number of unique values
    _ => {
      let unique_count = column
        .n_unique()
        .map_err(|e| anyhow::anyhow!("Failed to count unique values: {}", e))?;

      if unique_count <= categorical_threshold {
        analyze_categorical_column(column, categorical_threshold)
      } else {
        // For complex types with too many unique values, just show basic info
        Ok(ColumnStats::Categorical {
          frequency_table: vec![],
          total_unique: unique_count,
          showing_top_n: false,
        })
      }
    }
  }
}

fn analyze_numerical_column(column: &Series) -> Result<ColumnStats> {
  // Get statistical measures
  let mean = column.mean();
  let std_dev = column.std(1);

  // Calculate quartiles for IQR using quantile_reduce
  let q25 = column
    .quantile_reduce(0.25, QuantileMethod::Nearest)
    .ok()
    .and_then(|scalar| scalar.value().extract::<f64>());

  let q75 = column
    .quantile_reduce(0.75, QuantileMethod::Nearest)
    .ok()
    .and_then(|scalar| scalar.value().extract::<f64>());

  let iqr = match (q25, q75) {
    (Some(q25_val), Some(q75_val)) => Some(q75_val - q25_val),
    _ => None,
  };

  Ok(ColumnStats::Numerical {
    mean,
    std_dev,
    q25,
    q75,
    iqr,
  })
}

fn analyze_categorical_column(
  column: &Series,
  categorical_threshold: usize,
) -> Result<ColumnStats> {
  let unique_count = column
    .n_unique()
    .map_err(|e| anyhow::anyhow!("Failed to count unique values: {}", e))?;

  // Get value counts
  let value_counts_result = column.value_counts(false, true, "count".into(), false);

  match value_counts_result {
    Ok(counts_df) => {
      let mut frequency_table = Vec::new();
      let showing_top_n = unique_count > categorical_threshold;

      // Extract the values and counts
      let values_column = counts_df
        .column(column.name())
        .map_err(|e| anyhow::anyhow!("Failed to get values column: {}", e))?;
      let counts_column = counts_df
        .column("count")
        .map_err(|e| anyhow::anyhow!("Failed to get counts column: {}", e))?;

      let limit = if showing_top_n {
        std::cmp::min(10, unique_count)
      } else {
        unique_count
      };

      for i in 0..std::cmp::min(limit, counts_df.height()) {
        let value = values_column
          .get(i)
          .map_err(|e| anyhow::anyhow!("Failed to get value at index {}: {}", i, e))?;
        let count = counts_column
          .get(i)
          .map_err(|e| anyhow::anyhow!("Failed to get count at index {}: {}", i, e))?;

        let value_str = match value {
          AnyValue::String(s) => s.to_string(),
          AnyValue::Categorical(idx, rev_map, _) => rev_map.get(idx).to_string(),
          _ => format!("{value}"),
        };

        if let AnyValue::UInt32(count_val) = count {
          frequency_table.push((value_str, count_val));
        }
      }

      Ok(ColumnStats::Categorical {
        frequency_table,
        total_unique: unique_count,
        showing_top_n,
      })
    }
    Err(_) => {
      // Fallback: just return unique count
      Ok(ColumnStats::Categorical {
        frequency_table: vec![],
        total_unique: unique_count,
        showing_top_n: false,
      })
    }
  }
}

fn format_summary(summaries: &[ColumnSummary]) -> String {
  let mut output = String::new();

  output.push_str("📋 Column Analysis\n");
  output.push_str("━━━━━━━━━━━━━━━━━━\n\n");

  for (i, summary) in summaries.iter().enumerate() {
    output.push_str(&format!(
      "{}. Column: '{}' ({})\n",
      i + 1,
      summary.name,
      summary.data_type
    ));

    match &summary.summary {
      ColumnStats::Numerical {
        mean,
        std_dev,
        q25,
        q75,
        iqr,
      } => {
        output.push_str("   📈 Numerical Statistics:\n");

        if let Some(mean_val) = mean {
          output.push_str(&format!("      Mean: {mean_val:.6}\n"));
        } else {
          output.push_str("      Mean: N/A (no valid values)\n");
        }

        if let Some(std_val) = std_dev {
          output.push_str(&format!("      Std Dev: {std_val:.6}\n"));
        } else {
          output.push_str("      Std Dev: N/A (no valid values)\n");
        }

        match (q25, q75, iqr) {
          (Some(q25_val), Some(q75_val), Some(iqr_val)) => {
            output.push_str(&format!("      Q1 (25%): {q25_val:.6}\n"));
            output.push_str(&format!("      Q3 (75%): {q75_val:.6}\n"));
            output.push_str(&format!("      IQR: {iqr_val:.6}\n"));
          }
          _ => {
            output.push_str("      Quartiles: N/A (no valid values)\n");
          }
        }
      }

      ColumnStats::Categorical {
        frequency_table,
        total_unique,
        showing_top_n,
      } => {
        if frequency_table.is_empty() {
          output.push_str(&format!(
            "   📊 Categorical: {total_unique} unique values (too many to display)\n"
          ));
        } else {
          if *showing_top_n {
            output.push_str(&format!(
              "   📊 Categorical: {total_unique} total unique values (showing top 10):\n"
            ));
          } else {
            output.push_str(&format!(
              "   📊 Categorical: {total_unique} unique values:\n"
            ));
          }
          for (value, count) in frequency_table {
            let percentage =
              (*count as f64 / frequency_table.iter().map(|(_, c)| *c as f64).sum::<f64>()) * 100.0;
            output.push_str(&format!("      '{value}': {count} ({percentage:.1}%)\n"));
          }
        }
      }
    }

    output.push('\n');
  }

  output.push_str("✅ Analysis complete!\n");

  output
}
