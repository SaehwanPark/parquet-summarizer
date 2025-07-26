# Parquet Summarizer

A high-performance CLI tool for analyzing Parquet files using Polars.

## Usage Examples

```bash
# Basic usage - analyze parquet file and print to stdout
cargo run -- data.parquet

# Save analysis to a file
cargo run -- data.parquet -o analysis.txt

# Use low memory mode for very large files
cargo run -- large_data.parquet --low-memory

# Custom categorical threshold (default is 10)
cargo run -- data.parquet --categorical-threshold 5

# Show help
cargo run --help
```

## Features

✅ **Smart Data Type Detection**: Automatically identifies numerical vs categorical columns

✅ **Efficient Memory Usage**: Uses Polars lazy loading and optional streaming for large files

✅ **Comprehensive Statistics**:
  - Numerical: mean, standard deviation, IQR (Q1, Q3)
  - Categorical: frequency tables with percentages

✅ **Flexible Output**: Print to stdout or save to file

✅ **Large File Support**: Low memory mode for reduced memory usage

✅ **Configurable**: Adjustable categorical threshold

✅ **Error Handling**: Robust error handling with helpful messages

## Smart Strategies for Large Files

- **Lazy Loading**: Uses `LazyFrame::scan_parquet()` to defer computation
- **Low Memory Mode**: Optional `--low-memory` flag for processing large files with reduced parallelism
- **Efficient Statistics**: Leverages Polars' optimized statistical functions
- **Memory Management**: Automatically uses appropriate data types and avoids unnecessary copies
