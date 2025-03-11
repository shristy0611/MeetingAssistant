#!/bin/bash

# Set script to exit on error
set -e

# Default values
COMPOSE_FILE="docker/docker-compose.yml"
ENV_FILE=".env"
ACTION=""
SERVICE=""
PROFILE=""

# Help message
show_help() {
    echo "AMPTALK Container Management Script"
    echo
    echo "Usage: $0 [options] <action> [service]"
    echo
    echo "Actions:"
    echo "  start       Start services"
    echo "  stop        Stop services"
    echo "  restart     Restart services"
    echo "  build       Build services"
    echo "  logs        View service logs"
    echo "  ps          List services"
    echo "  clean       Clean up containers and volumes"
    echo
    echo "Options:"
    echo "  -p, --profile <name>    Use specific profile (cpu/gpu)"
    echo "  -e, --env <file>        Specify environment file (default: .env)"
    echo "  -h, --help              Show this help message"
    echo
    echo "Examples:"
    echo "  $0 start                Start all services"
    echo "  $0 -p gpu start         Start services with GPU support"
    echo "  $0 logs api             View API service logs"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        -e|--env)
            ENV_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        start|stop|restart|build|logs|ps|clean)
            ACTION="$1"
            shift
            ;;
        *)
            SERVICE="$1"
            shift
            ;;
    esac
done

# Check if action is provided
if [ -z "$ACTION" ]; then
    echo "Error: No action specified"
    show_help
    exit 1
fi

# Load environment variables
if [ -f "$ENV_FILE" ]; then
    export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
fi

# Compose command prefix
COMPOSE_CMD="docker compose -f $COMPOSE_FILE"

# Add profile if specified
if [ ! -z "$PROFILE" ]; then
    COMPOSE_CMD="$COMPOSE_CMD --profile $PROFILE"
fi

# Execute action
case $ACTION in
    start)
        echo "Starting services..."
        $COMPOSE_CMD up -d $SERVICE
        ;;
    stop)
        echo "Stopping services..."
        $COMPOSE_CMD down $SERVICE
        ;;
    restart)
        echo "Restarting services..."
        $COMPOSE_CMD restart $SERVICE
        ;;
    build)
        echo "Building services..."
        $COMPOSE_CMD build $SERVICE
        ;;
    logs)
        if [ -z "$SERVICE" ]; then
            echo "Error: Service name required for logs"
            exit 1
        fi
        echo "Viewing logs for $SERVICE..."
        $COMPOSE_CMD logs -f $SERVICE
        ;;
    ps)
        echo "Listing services..."
        $COMPOSE_CMD ps
        ;;
    clean)
        echo "Cleaning up..."
        $COMPOSE_CMD down -v --remove-orphans
        docker system prune -f
        ;;
    *)
        echo "Error: Unknown action '$ACTION'"
        show_help
        exit 1
        ;;
esac 