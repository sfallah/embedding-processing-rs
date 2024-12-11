#!/bin/bash
set -eu
echo "Downloading"
mkdir -p /models/
echo "Running"
ROCKS_DB_PATH="$PERSISTENT_STORAGE_PATH/rocks_db_data"
mkdir -p $ROCKS_DB_PATH

/usr/local/bin/embedding-server-rs --model-id "$MODEL_ID" --model-type "$MODEL_TYPE" --db-path "$ROCKS_DB_PATH"
