#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Fetching Go dependencies..."
go mod tidy

echo "Building shared library..."
go build -buildmode=c-shared -o libanthropic.so main.go

echo "Generated:"
ls -la libanthropic.so libanthropic.h

echo "Done! Library at: $(pwd)/libanthropic.so"
