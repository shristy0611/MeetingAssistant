#!/bin/bash

# AMPTALK Model Pruning Script
# This script provides a convenient interface for running model pruning
# either locally or with Docker.

# Set default values
CONFIG="configs/default.json"
MODE="docker"
SPARSITY=0.3
MODEL="openai/whisper-large-v3-turbo"
DEVICE="auto"
MAX_SAMPLES=20

# ANSI color codes
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display help message
function show_help {
    echo -e "${CYAN}AMPTALK Model Pruning Script${NC}"
    echo -e "Usage: $0 [options]"
    echo -e ""
    echo -e "Options:"
    echo -e "  -h, --help                  Show this help message"
    echo -e "  -c, --config CONFIG         Specify configuration file (default: $CONFIG)"
    echo -e "  -m, --mode MODE             Run mode: 'docker' or 'local' (default: $MODE)"
    echo -e "  -s, --sparsity SPARSITY     Target sparsity ratio (default: $SPARSITY)"
    echo -e "  --model MODEL               Model to prune (default: $MODEL)"
    echo -e "  -d, --device DEVICE         Computing device: 'auto', 'cuda', 'cpu', or 'mps' (default: $DEVICE)"
    echo -e "  --samples SAMPLES           Maximum number of evaluation samples (default: $MAX_SAMPLES)"
    echo -e ""
    echo -e "Examples:"
    echo -e "  $0 --mode local --sparsity 0.5"
    echo -e "  $0 --model openai/whisper-base --device cpu"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -s|--sparsity)
            SPARSITY="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        --samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check if the config file exists
if [ ! -f "$CONFIG" ] && [[ "$CONFIG" != "configs/"* ]]; then
    # Try with configs/ prefix
    if [ -f "configs/$CONFIG" ]; then
        CONFIG="configs/$CONFIG"
    else
        echo -e "${RED}Error: Configuration file '$CONFIG' not found${NC}"
        exit 1
    fi
fi

# Create necessary directories
mkdir -p models/cache
mkdir -p output

# Function to generate a dynamic configuration
function generate_dynamic_config {
    local dynamic_config="configs/dynamic_config.json"
    
    # If the specified config exists, use it as a base
    if [ -f "$CONFIG" ]; then
        cp "$CONFIG" "$dynamic_config"
        
        # Update the values using jq if available
        if command -v jq &> /dev/null; then
            jq ".pruning.sparsity = $SPARSITY" "$dynamic_config" > temp.json && mv temp.json "$dynamic_config"
            jq ".model.name = \"$MODEL\"" "$dynamic_config" > temp.json && mv temp.json "$dynamic_config"
            jq ".hardware.device = \"$DEVICE\"" "$dynamic_config" > temp.json && mv temp.json "$dynamic_config"
            
            # Update max_samples for each dataset
            jq "(.evaluation.datasets[] | .max_samples) |= $MAX_SAMPLES" "$dynamic_config" > temp.json && mv temp.json "$dynamic_config"
        else
            echo -e "${YELLOW}Warning: jq not found. Using default configuration with command line parameters.${NC}"
        fi
    else
        echo -e "${YELLOW}Warning: Config file not found. Using command line parameters.${NC}"
        
        # Create a basic config file
        cat > "$dynamic_config" << EOF
{
    "model": {
        "name": "$MODEL",
        "cache_dir": "../models/cache"
    },
    "pruning": {
        "method": "magnitude",
        "sparsity": $SPARSITY,
        "target_modules": ["encoder.layers", "decoder.layers"],
        "excluded_modules": ["encoder.embed_positions", "decoder.embed_tokens"],
        "pruning_pattern": "unstructured",
        "apply_outlier_weighting": true,
        "outlier_detection_method": "owl",
        "outlier_hyper_lambda": 0.08,
        "outlier_hyper_m": 5
    },
    "evaluation": {
        "datasets": [
            {
                "name": "librispeech",
                "split": "test-clean",
                "subset": "clean", 
                "max_samples": $MAX_SAMPLES
            }
        ],
        "metrics": ["wer"],
        "batch_size": 8
    },
    "optimization": {
        "quantization": {
            "enabled": true,
            "bits": 8,
            "method": "dynamic"
        },
        "compile": {
            "enabled": true,
            "mode": "reduce-overhead"
        }
    },
    "output": {
        "dir": "../output",
        "save_model": true,
        "log_to_tensorboard": true,
        "save_pruned_checkpoints": true,
        "export_onnx": true
    },
    "hardware": {
        "device": "$DEVICE",
        "use_mps": true,
        "precision": "float16",
        "cpu_threads": 8
    }
}
EOF
    fi
    
    echo "$dynamic_config"
}

# Generate dynamic configuration
DYNAMIC_CONFIG=$(generate_dynamic_config)

# Print execution parameters
echo -e "${GREEN}Starting AMPTALK model pruning with the following parameters:${NC}"
echo -e "  Mode:      ${CYAN}$MODE${NC}"
echo -e "  Model:     ${CYAN}$MODEL${NC}"
echo -e "  Sparsity:  ${CYAN}$SPARSITY${NC}"
echo -e "  Device:    ${CYAN}$DEVICE${NC}"
echo -e "  Config:    ${CYAN}$DYNAMIC_CONFIG${NC}"
echo -e "  Samples:   ${CYAN}$MAX_SAMPLES${NC}"

# Execute based on selected mode
if [ "$MODE" = "docker" ]; then
    echo -e "\n${GREEN}Running pruning in Docker container...${NC}"
    if [ -f "docker-compose.yml" ]; then
        docker-compose run pruning python scripts/prune_whisper.py --config "$DYNAMIC_CONFIG"
    else
        echo -e "${RED}Error: docker-compose.yml not found${NC}"
        exit 1
    fi
elif [ "$MODE" = "local" ]; then
    echo -e "\n${GREEN}Running pruning locally...${NC}"
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    else
        source venv/bin/activate
    fi
    
    # Run the pruning script
    python scripts/prune_whisper.py --config "$DYNAMIC_CONFIG"
    
    # Deactivate virtual environment
    deactivate
else
    echo -e "${RED}Error: Invalid mode '$MODE'. Use 'docker' or 'local'.${NC}"
    exit 1
fi

echo -e "\n${GREEN}Pruning process completed!${NC}" 