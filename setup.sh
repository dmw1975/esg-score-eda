#!/bin/bash
# Setup script for ESG Score EDA environment

# Function to display help message
show_help() {
    echo "ESG Score EDA Setup Script"
    echo ""
    echo "Usage: bash setup.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  --help, -h         Show this help message"
    echo "  --env, -e          Create and set up virtual environment only"
    echo "  --directories, -d  Create directory structure only"
    echo "  --run, -r          Run the full pipeline after setup"
    echo "  --sample, -s       Generate sample data"
    echo ""
    echo "If no options are provided, will set up environment and directories but not run pipeline."
}

# Default flags
CREATE_ENV=true
CREATE_DIRECTORIES=true
RUN_PIPELINE=false
GENERATE_SAMPLE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --env|-e)
            CREATE_ENV=true
            CREATE_DIRECTORIES=false
            RUN_PIPELINE=false
            GENERATE_SAMPLE=false
            ;;
        --directories|-d)
            CREATE_ENV=false
            CREATE_DIRECTORIES=true
            RUN_PIPELINE=false
            GENERATE_SAMPLE=false
            ;;
        --run|-r)
            RUN_PIPELINE=true
            ;;
        --sample|-s)
            GENERATE_SAMPLE=true
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    shift
done

# Setup virtual environment if requested
if $CREATE_ENV; then
    echo "Checking Python and venv..."
    if ! command -v python3 &> /dev/null; then
        echo "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    # Create virtual environment
    echo "Creating virtual environment..."
    python3 -m venv venv || { echo "Failed to create virtual environment"; exit 1; }
    
    # Activate the environment
    echo "Activating virtual environment..."
    source venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }
    
    # Install requirements
    echo "Installing requirements..."
    pip install -r requirements.txt || { echo "Failed to install requirements"; exit 1; }
    
    echo "Virtual environment setup complete!"
fi

# Create directory structure if requested
if $CREATE_DIRECTORIES; then
    echo "Creating directories..."
    
    # Create main directories
    mkdir -p data/{raw,processed,processed/pkl}
    mkdir -p visualizations/{missing_values,imputation,outliers/{boxplots,shap,isolation},one_hot,yeo_johnson,performance}
    
    echo "Directory structure created!"
fi

# Generate sample data if requested
if $GENERATE_SAMPLE; then
    echo "Generating sample data..."
    
    # Ensure directories exist
    mkdir -p data/raw
    
    # Generate sample data
    if [ -f "data/raw/sample_data.py" ]; then
        python data/raw/sample_data.py || { echo "Failed to generate sample data"; exit 1; }
        echo "Sample data generated successfully!"
    else
        echo "Sample data generator script not found at data/raw/sample_data.py"
        exit 1
    fi
fi

# Run the pipeline if requested
if $RUN_PIPELINE; then
    echo "Running data processing pipeline..."
    
    # Check if virtual environment is activated
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        echo "Virtual environment not activated. Activating now..."
        source venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }
    fi
    
    # Run the pipeline
    python main.py --all || { echo "Pipeline execution failed"; exit 1; }
    
    echo "Data processing complete!"
    echo "All processed data is available in the data/ directory."
    echo "Visualizations are available in the visualizations/ directory."
fi

echo "Setup completed successfully!"