#!/bin/bash

# ESG Score EDA Repository Cleanup Script
# This script removes unnecessary files from Git tracking
# Files are removed from Git but kept locally

echo "Starting repository cleanup..."

# Remove archive directories
echo "Removing archive directories..."
git rm -r --cached archive/ 2>/dev/null || echo "archive/ not in git"
git rm -r --cached archive_20250624_122349/ 2>/dev/null || echo "archive_20250624_122349/ not in git"
git rm -r --cached archive_20250624_123458/ 2>/dev/null || echo "archive_20250624_123458/ not in git"
git rm -r --cached archive_20250624_125216/ 2>/dev/null || echo "archive_20250624_125216/ not in git"
git rm -r --cached archive_20250624_130651/ 2>/dev/null || echo "archive_20250624_130651/ not in git"

# Remove log files
echo "Removing log files..."
git rm --cached *.log 2>/dev/null || echo "No log files in git"
git rm --cached missing_values_analysis.log 2>/dev/null || true
git rm --cached outlier_detection.log 2>/dev/null || true
git rm --cached pipeline_validation.log 2>/dev/null || true

# Remove temporary documentation
echo "Removing temporary documentation..."
git rm --cached ARCHIVE_README.txt 2>/dev/null || echo "ARCHIVE_README.txt not in git"
git rm --cached EXECUTION_SUMMARY.md 2>/dev/null || echo "EXECUTION_SUMMARY.md not in git"
git rm --cached FEATURE_TRANSFORMATION_FIX_COMPLETE.md 2>/dev/null || echo "FEATURE_TRANSFORMATION_FIX_COMPLETE.md not in git"
git rm --cached ML_TRANSFORMATION_FIX_REPORT.md 2>/dev/null || echo "ML_TRANSFORMATION_FIX_REPORT.md not in git"
git rm --cached final_validation_report.md 2>/dev/null || echo "final_validation_report.md not in git"
git rm --cached location_update_notes.txt 2>/dev/null || echo "location_update_notes.txt not in git"

# Remove utility scripts (except archive_files.py which user wants to keep)
echo "Removing one-time utility scripts..."
git rm --cached cleanup_repo.sh 2>/dev/null || echo "cleanup_repo.sh not in git"
git rm --cached move_isolation_forest_files.sh 2>/dev/null || echo "move_isolation_forest_files.sh not in git"
git rm --cached prepare_for_github.sh 2>/dev/null || echo "prepare_for_github.sh not in git"

# Remove test results
echo "Removing test results directory..."
git rm -r --cached test_results/ 2>/dev/null || echo "test_results/ not in git"

# Remove Python cache files
echo "Removing Python cache files..."
find . -type d -name __pycache__ -exec git rm -r --cached {} + 2>/dev/null || echo "No __pycache__ in git"
find . -name "*.pyc" -exec git rm --cached {} + 2>/dev/null || echo "No .pyc files in git"

# Remove virtual environment if tracked
echo "Checking for virtual environment..."
git rm -r --cached venv/ 2>/dev/null || echo "venv/ not in git"

# Remove any example_data directory
echo "Removing example_data if present..."
git rm -r --cached example_data/ 2>/dev/null || echo "example_data/ not in git"

echo ""
echo "Cleanup complete! Next steps:"
echo "1. Review the changes with: git status"
echo "2. Commit the changes with:"
echo "   git commit -m 'Clean up repository: remove archives, logs, and temporary files'"
echo "3. After pushing, remove plan.md and this script:"
echo "   git rm --cached plan.md git_cleanup_commands.sh"
echo "   git commit -m 'Remove cleanup planning files'"
echo ""
echo "Note: Files are only removed from Git tracking, not deleted from disk."