# Schwarzschild Black Hole Ray Tracer - Task Runner

# デフォルトタスク: ヘルプを表示
default:
    @just --list

# 環境セットアップ（初回のみ）
setup:
    @echo "Setting up development environment..."
    @echo "1. Creating Python virtual environment..."
    uv venv
    @echo "2. Installing Python dependencies..."
    uv pip install jupyterlab ipykernel
    @echo "3. Installing evcxr_jupyter (Rust kernel)..."
    cargo install evcxr_jupyter
    @echo "4. Registering Jupyter kernel..."
    evcxr_jupyter --install
    @echo "✓ Setup complete!"
    @echo ""
    @echo "Run 'just lab' to start JupyterLab"

# Python 依存関係の更新
update-python:
    @echo "Updating Python dependencies..."
    uv pip install --upgrade jupyterlab ipykernel

# Rust 依存関係の更新
update-rust:
    @echo "Updating Rust dependencies..."
    cargo update

# すべての依存関係を更新
update: update-python update-rust
    @echo "✓ All dependencies updated"

# JupyterLab を起動
lab:
    @echo "Starting JupyterLab..."
    . .venv/bin/activate && jupyter lab notebooks/

# CLI アプリケーションを実行
run:
    @echo "Running CLI application..."
    cargo run --release

# プロジェクトをビルド
build:
    @echo "Building project..."
    cargo build --release

# プロジェクトをビルド（デバッグモード）
build-dev:
    @echo "Building project (debug)..."
    cargo build

# テストを実行
test:
    @echo "Running tests..."
    cargo test

# コードのフォーマット
fmt:
    @echo "Formatting code..."
    cargo fmt

# Linter を実行
lint:
    @echo "Running linter..."
    cargo clippy

# クリーンアップ
clean:
    @echo "Cleaning build artifacts..."
    cargo clean
    @echo "✓ Clean complete"

# 完全クリーンアップ（仮想環境も削除）
clean-all: clean
    @echo "Removing Python virtual environment..."
    rm -rf .venv
    @echo "✓ Complete clean done"

# Jupyter カーネルリストを表示
kernels:
    @echo "Installed Jupyter kernels:"
    . .venv/bin/activate && jupyter kernelspec list

# プロジェクト情報を表示
info:
    @echo "=== Project Information ==="
    @echo "Rust version:"
    @rustc --version
    @echo ""
    @echo "Cargo version:"
    @cargo --version
    @echo ""
    @echo "Python version:"
    @python3 --version
    @echo ""
    @echo "uv version:"
    @uv --version
    @echo ""
    @echo "Virtual environment:"
    @if [ -d .venv ]; then echo "✓ exists"; else echo "✗ not found (run 'just setup')"; fi
