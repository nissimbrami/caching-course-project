#!/usr/bin/env bash
# run_all.sh - One-command script to reproduce all experimental results.
# Usage: bash scripts/run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configuration (override via environment variables)
CACHE_SIZE="${CACHE_SIZE:-1000}"
NUM_RUNS="${NUM_RUNS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Create output directories
mkdir -p "$OUTPUT_DIR/benchmarks" "$OUTPUT_DIR/plots" "$OUTPUT_DIR/ablation"

# ─── Step 1: Run Unit Tests ─────────────────────────────────────────────────────
print_header "Step 1/4: Running Unit Tests"

if python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing 2>&1; then
    print_success "All unit tests passed."
else
    print_error "Some tests failed. Continuing with benchmarks..."
fi

# ─── Step 2: Run Full Benchmark Suite ───────────────────────────────────────────
print_header "Step 2/4: Running Full Benchmark Suite"

echo "Configuration:"
echo "  Cache size: $CACHE_SIZE"
echo "  Number of runs: $NUM_RUNS"
echo "  Output directory: $OUTPUT_DIR"
echo ""

python -m benchmarks.run_all \
    --n-runs "$NUM_RUNS" \
    --output-dir "$OUTPUT_DIR/benchmarks"

print_success "Benchmarks complete. Results saved to $OUTPUT_DIR/benchmarks/"

# ─── Step 3: Run Ablation Study ─────────────────────────────────────────────────
print_header "Step 3/4: Running Ablation Study (alpha x beta sweep)"

python scripts/run_ablation.py \
    --cache-size "$CACHE_SIZE" \
    --num-runs "$NUM_RUNS" \
    --output-dir "$OUTPUT_DIR/ablation"

print_success "Ablation study complete. Results saved to $OUTPUT_DIR/ablation/"

# ─── Step 4: Generate Plots ─────────────────────────────────────────────────────
print_header "Step 4/4: Generating Publication-Quality Plots"

python scripts/generate_plots.py \
    --input-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR/plots"

print_success "Plots generated in $OUTPUT_DIR/plots/"

# ─── Summary ────────────────────────────────────────────────────────────────────
print_header "Results Summary"

# Print summary table if benchmark results exist
if [ -f "$OUTPUT_DIR/benchmarks/summary.csv" ]; then
    echo ""
    echo "Benchmark Summary (cache_size=$CACHE_SIZE, runs=$NUM_RUNS):"
    echo "─────────────────────────────────────────────────────────────────"
    printf "%-8s │ %10s │ %10s │ %15s │ %12s\n" "Policy" "Hit Rate" "CWHR" "Savings (\$/1K)" "Latency (ms)"
    echo "─────────────────────────────────────────────────────────────────"

    # Skip header line and print each row
    tail -n +2 "$OUTPUT_DIR/benchmarks/summary.csv" | while IFS=',' read -r policy hit_rate cwhr savings latency; do
        printf "%-8s │ %9s%% │ %9s%% │ %15s │ %12s\n" \
            "$policy" "$hit_rate" "$cwhr" "\$$savings" "$latency"
    done

    echo "─────────────────────────────────────────────────────────────────"
    echo ""
else
    print_warning "No summary.csv found. Run benchmarks first."
fi

# List generated files
echo ""
echo "Generated artifacts:"
echo "  Benchmark CSVs: $(find "$OUTPUT_DIR/benchmarks" -name "*.csv" 2>/dev/null | wc -l) files"
echo "  Ablation CSVs:  $(find "$OUTPUT_DIR/ablation" -name "*.csv" 2>/dev/null | wc -l) files"
echo "  Plot images:    $(find "$OUTPUT_DIR/plots" -name "*.png" 2>/dev/null | wc -l) files"
echo ""

print_success "All tasks completed. Results are in $OUTPUT_DIR/"
