#!/bin/bash
# Script to build and deploy documentation

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Change to script directory (project root)
cd "$(dirname "$0")"

echo -e "${GREEN}FPGA Placement FEM Documentation${NC}\n"

COMMAND=${1:-build}

case $COMMAND in
    build)
        echo -e "${YELLOW}Building documentation...${NC}"
        uv run python -m mkdocs build --clean --strict
        echo -e "${GREEN}✓ Built successfully → site/${NC}"
        ;;

    serve)
        echo -e "${YELLOW}Starting local server...${NC}"
        echo -e "Docs at: ${GREEN}http://127.0.0.1:8000${NC}"
        echo -e "Press ${YELLOW}Ctrl+C${NC} to stop\n"
        uv run python -m mkdocs serve
        ;;

    deploy)
        echo -e "${YELLOW}Deploying to GitHub Pages...${NC}"
        uv run python -m mkdocs gh-deploy --clean --force
        echo -e "${GREEN}✓ Deployed to GitHub Pages${NC}"
        ;;

    check)
        echo -e "${YELLOW}Checking documentation...${NC}"
        uv run python -m mkdocs build --strict && echo -e "${GREEN}✓ No errors${NC}" || echo -e "${RED}✗ Has errors${NC}"
        ;;

    clean)
        echo -e "${YELLOW}Cleaning build artifacts...${NC}"
        rm -rf site/
        echo -e "${GREEN}✓ Cleaned${NC}"
        ;;

    install)
        echo -e "${YELLOW}Installing dependencies...${NC}"
        uv run python -m pip install ".[docs]"
        echo -e "${GREEN}✓ Installed${NC}"
        ;;

    *)
        echo "Usage: $0 {build|serve|deploy|check|clean|install}"
        echo ""
        echo "Commands:"
        echo "  build   - Build documentation"
        echo "  serve   - Start local server"
        echo "  deploy  - Deploy to GitHub Pages"
        echo "  check   - Check for errors"
        echo "  clean   - Remove build artifacts"
        echo "  install - Install dependencies"
        exit 1
        ;;
esac
