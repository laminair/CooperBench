#!/usr/bin/env bash
# Build all Docker images for CooperBench tasks in dataset/
#
# Usage:
#   ./scripts/build_task_images.sh               # Build all task images
#   ./scripts/build_task_images.sh --dry-run     # Show what would be built
#   ./scripts/build_task_images.sh --push        # Build and push to registry
#   ./scripts/build_task_images.sh --no-cache    # Build without Docker cache
#   ./scripts/build_task_images.sh --filter llama_index  # Build only matching repos
#
# Naming convention:
#   dataset/{repo}_task/task{id}/Dockerfile  ->  akhatua/cooperbench-{repo-clean}:task{id}
#   where {repo-clean} = {repo} with underscores replaced by hyphens

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATASET_DIR="$REPO_ROOT/dataset"
REGISTRY="${COOPERBENCH_REGISTRY:-akhatua}"
IMAGE_PREFIX="${COOPERBENCH_IMAGE_PREFIX:-cooperbench}"

DRY_RUN=false
PUSH=false
NO_CACHE=false
FILTER=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Build all Docker images for CooperBench task Dockerfiles found under dataset/.

Options:
  --dry-run     Print what would be built without actually building
  --push        Push images to the registry after building
  --no-cache    Build without the Docker build cache
  --filter PAT  Only build tasks whose repo name matches PAT
  -h, --help    Show this help message

Environment:
  COOPERBENCH_REGISTRY        Docker registry (default: akhatua)
  COOPERBENCH_IMAGE_PREFIX    Image name prefix (default: cooperbench)
EOF
    exit 0
}

for arg in "$@"; do
    case "$arg" in
        --dry-run)   DRY_RUN=true ;;
        --push)      PUSH=true ;;
        --no-cache)  NO_CACHE=true ;;
        --filter)    shift; FILTER="$1" ;;
        -h|--help)   usage ;;
        *)           ;;
    esac
    shift 2>/dev/null || true
done

echo "=============================================="
echo "CooperBench Task Image Builder"
echo "=============================================="
echo "Registry:  $REGISTRY"
echo "Dataset:   $DATASET_DIR"
if [ -n "$FILTER" ]; then
    echo "Filter:    $FILTER"
fi
if $DRY_RUN; then
    echo "Mode:      DRY RUN"
fi
if $PUSH; then
    echo "Push:      enabled"
fi
if $NO_CACHE; then
    echo "Cache:     disabled"
fi
echo ""

echo "Scanning dataset for tasks..."
BUILDS=()
for repo_dir in "$DATASET_DIR"/*_task; do
    if [ ! -d "$repo_dir" ]; then
        continue
    fi
    repo_name=$(basename "$repo_dir")

    # Filter by repo name if --filter was given
    if [ -n "$FILTER" ] && [[ "$repo_name" != *"$FILTER"* ]]; then
        continue
    fi

    # Convert repo name: llama_index_task -> llama-index
    image_repo=$(echo "$repo_name" | sed 's/_task$//' | tr '_' '-')

    for task_dir in "$repo_dir"/task*/; do
        if [ ! -d "$task_dir" ]; then
            continue
        fi

        dockerfile="$task_dir/Dockerfile"
        if [ ! -f "$dockerfile" ]; then
            continue
        fi

        task_id=$(basename "$task_dir" | sed 's/task//')
        image_tag="$REGISTRY/$IMAGE_PREFIX-$image_repo:task$task_id"
        BUILDS+=("$repo_name|$task_id|$task_dir|$image_tag")
    done
done

TOTAL=${#BUILDS[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "No tasks found."
    exit 0
fi

echo "Found $TOTAL task(s) to build"
echo "=============================================="
echo ""

SUCCESS=0
FAILED=0
FAILED_LIST=()
COUNT=0

for entry in "${BUILDS[@]}"; do
    IFS='|' read -r repo_name task_id task_dir image_tag <<< "$entry"
    COUNT=$((COUNT + 1))

    echo "[$COUNT/$TOTAL] $image_tag"

    if $DRY_RUN; then
        echo "  Dir: $task_dir"
        if $PUSH; then
            echo "  Would push: $image_tag"
        fi
        SUCCESS=$((SUCCESS + 1))
        echo ""
        continue
    fi

    BUILD_ARGS=("-t" "$image_tag")
    if $NO_CACHE; then
        BUILD_ARGS+=("--no-cache")
    fi

    # Build the image from the task directory
    BUILD_OUT=$(mktemp)
    if docker build "${BUILD_ARGS[@]}" -f "$task_dir/Dockerfile" "$task_dir" > "$BUILD_OUT" 2>&1; then
        SUCCESS=$((SUCCESS + 1))
        echo "  OK (built)"
        rm -f "$BUILD_OUT"

        if $PUSH; then
            echo -n "  Pushing... "
            PUSH_OUT=$(mktemp)
            if docker push "$image_tag" > "$PUSH_OUT" 2>&1; then
                echo "OK (pushed)"
            else
                echo "FAILED"
                FAILED=$((FAILED + 1))
                FAILED_LIST+=("$image_tag (push)")
                echo "  --- push output ---"
                cat "$PUSH_OUT"
                echo "  -------------------"
            fi
            rm -f "$PUSH_OUT"
        fi
    else
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$image_tag (build)")
        echo "  FAILED"
        echo "  --- build output ---"
        cat "$BUILD_OUT"
        echo "  --------------------"
        rm -f "$BUILD_OUT"
    fi
    echo ""
done

echo "=============================================="
echo "Summary"
echo "=============================================="
echo "Total:   $TOTAL"
echo "Success: $SUCCESS"
echo "Failed:  $FAILED"

if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo ""
    echo "Failed tasks:"
    for item in "${FAILED_LIST[@]}"; do
        echo "  - $item"
    done
fi

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Some builds failed."
    exit 1
fi

echo ""
echo "Done."
