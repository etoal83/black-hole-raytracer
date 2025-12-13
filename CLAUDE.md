# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

WGSL コンピュートシェーダを使用したシュワルツシルトブラックホールのレイトレーシングエンジン。ライブラリとCLIアプリケーションのハイブリッド構成で、JupyterLab統合もサポート。

## ビルドとテストコマンド

このプロジェクトは `just` タスクランナーを使用します：

```bash
# 環境セットアップ（初回のみ）
just setup

# CLIアプリケーションを実行
just run

# リリースビルド
just build

# デバッグビルド
just build-dev

# テストを実行
just test

# コードフォーマット
just fmt

# Linter（Clippy）を実行
just lint

# JupyterLabを起動
just lab

# 依存関係を更新
just update
```

### カーゴコマンド

```bash
# デフォルト（GUIなし）でビルド
cargo build --release

# GUIあり（egui統合）でビルド
cargo build --release --features gui

# 特定のシェーダを使ってアプリを実行
cargo run --release -- --shader src/ray_tracer_euler.wgsl

# デバッグステップ視覚化モード
cargo run --release -- --debug-steps

# パフォーマンスログを記録
cargo run --release -- --perf-log measurements/perf.csv --duration 60.0
```

## アーキテクチャ

### デュアルモード設計

**ライブラリモード** (`src/lib.rs`):
- JupyterLab統合のための設計
- `GpuContext::new()`: Surfaceなしでの実行可能
- `block_on()`: 非同期関数の同期実行ヘルパー
- 画像保存とデータ取得のAPI提供

**アプリケーションモード** (`src/main.rs`):
- winit + wgpu によるウィンドウ管理
- リアルタイムアニメーション（カメラ周回）
- オプショナルGUI（`--features gui`）によるパフォーマンス統計表示
- タイムスタンプクエリによるGPU計算時間の正確な計測

### GPUパイプライン構成

```
Compute Pipeline (レイトレーシング)
  ├─ Binding 0: StorageTexture (出力画像)
  ├─ Binding 1: Uniform (Camera)
  ├─ Binding 2: Uniform (SceneParams)
  ├─ Binding 3: Texture (スカイボックス/スターマップ)
  └─ Binding 4: Sampler

Render Pipeline (ディスプレイ)
  ├─ フルスクリーンクアッド描画
  ├─ Binding 0: Texture (compute結果)
  └─ Binding 1: Sampler

Optional GUI Pipeline (egui)
  └─ パフォーマンス統計のオーバーレイ
```

## 重要なコンセプト

### 1. メモリレイアウト同期

RustとWGSLシェーダ間で完全なメモリレイアウト同期が必要：

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Camera {
    pub position: [f32; 3],
    pub _padding1: f32,
    pub forward: [f32; 3],
    pub _padding2: f32,
    pub up: [f32; 3],
    pub _padding3: f32,
}
```

- `#[repr(C)]` でメモリレイアウトを固定
- WGSL のアライメント要件（16バイト境界）に合わせたパディング
- `bytemuck` クレートの `Pod` と `Zeroable` トレイトで安全な転送を保証

### 2. 物理シミュレーション（Schwarzschild計量）

`src/ray_tracer_euler.wgsl` で実装：

```wgsl
// Schwarzschild計量テンソル成分
fn g_tt(r: f32, rs: f32) -> f32 { -(1.0 - rs / r) }
fn g_rr(r: f32, rs: f32) -> f32 { 1.0 / (1.0 - rs / r) }

// 測地線方程式の数値積分（Euler法）
fn trace_geodesic(pos: vec3<f32>, vel: vec3<f32>, rs: f32, dt: f32)
```

**実装上の特徴**:
- 3次元Cartesian座標での近似（完全な球座標計算ではない）
- Euler法による数値積分（dt = 0.1）
- リアルタイム性能のための簡略化

**シミュレートされる物理効果**:
- 重力レンズ効果
- 光子球（r ≈ 1.5 × schwarzschild_radius）
- イベントホライズン
- アインシュタインリング

詳細な導出は `euler.md` を参照。

### 3. シェーダファイルの動的読み込み

コンパイル時埋め込み（`include_str!`）ではなく、実行時にシェーダファイルを読み込む設計：

```rust
// lib.rs
pub fn load_shader(path: &str) -> anyhow::Result<String> {
    std::fs::read_to_string(path)
        .with_context(|| format!("Failed to load shader from {}", path))
}

// main.rs
let shader_code = load_shader(&args.shader)?;
```

**理由**: シェーダの反復開発を高速化（再コンパイル不要）

### 4. パフォーマンス計測システム

`PerformanceStats` 構造体でフレームタイムと統計を管理：

- **ウォームアップ期間**: 最初の10フレームを除外（初期化コストを排除）
- **GPU時間計測**: タイムスタンプクエリによる正確な計測
- **統計**: 平均/最小/最大FPS、標準偏差
- **CSV出力**: `--perf-log` オプションでログ記録

```rust
pub struct PerformanceStats {
    frame_times: Vec<Duration>,      // CPU時間
    gpu_times: Vec<Duration>,        // GPU時間
    warmup_frames: usize,            // デフォルト10
    // ...
}
```

### 5. テクスチャ処理

**HDR → LDR トーンマッピング**:
- EXR形式（HDR）のスターマップを読み込み
- Reinhardトーンマッピング: `rgb / (1.0 + rgb)`
- RGBA8形式に変換してGPUへ転送

**GPU → CPU データ転送**:
```rust
// ステージングバッファを介した非破壊転送
staging_buffer (COPY_DST | MAP_READ)
  ↓ copy_texture_to_buffer
output_texture (STORAGE_BINDING | COPY_SRC)
  ↓ buffer.slice(..).map_async()
Vec<u8> (RGBA8)
```

## 主要なファイル

- [src/lib.rs](src/lib.rs) - コアライブラリ（GpuContext, BlackHoleRenderer, Camera, SceneParams）
- [src/main.rs](src/main.rs) - CLIアプリケーション（winit統合、リアルタイムアニメーション、パフォーマンス計測）
- [src/ray_tracer_euler.wgsl](src/ray_tracer_euler.wgsl) - 光線追跡コンピュートシェーダ（測地線方程式、物理シミュレーション）
- [src/display.wgsl](src/display.wgsl) - ディスプレイシェーダ（フルスクリーンクアッド描画）
- [euler.md](euler.md) - 測地線方程式の数学的導出
- [notebooks/black_hole_rendering.ipynb](notebooks/black_hole_rendering.ipynb) - Jupyter統合サンプル

## 機能フラグ

```toml
[features]
default = []
gui = ["dep:egui", "dep:egui-winit", "dep:egui-wgpu"]
```

- **デフォルト**: GUIなし（軽量、ヘッドレス実行可能）
- **gui**: egui統合（パフォーマンス統計オーバーレイ）

最近のコミット（e1c4d6b）でデフォルトからGUIを削除。

## トラブルシューティング

### シェーダのコンパイルエラー

WGSLのバリデーションエラーが出た場合：
1. Rustの構造体定義とWGSLのstruct定義を比較
2. パディングフィールド（`_paddingN`）の有無を確認
3. アライメント要件（vec3は16バイト、vec4も16バイト）を確認

### JupyterLabでの実行

`block_on()` を使わずに `.await` を使うとエラーになる：
```rust
// ❌ 間違い（Jupyterでは動作しない）
let context = GpuContext::new().await?;

// ✅ 正しい
let context = block_on(GpuContext::new())?;
```

理由: evcxr_jupyter は非同期ランタイムを提供しないため。

### パフォーマンスが低い

1. リリースビルドを使用しているか確認: `cargo build --release`
2. デバッグステップモード（`--debug-steps`）を無効化
3. `SceneParams.max_steps` を調整（デフォルト500、減らすと速くなるが精度が落ちる）

## 依存関係

- **wgpu 22**: GPU抽象化、コンピュートシェーダ
- **winit 0.30**: ウィンドウ管理とイベントループ
- **tokio**: 非同期ランタイム（バッファマッピング用）
- **exr 1.72**: HDRテクスチャ読み込み
- **image 0.25**: 画像I/O（PNG/JPEG保存）
- **clap 4.5**: CLI引数パーサー
- **csv 1.3**: パフォーマンスログ記録
- **egui 0.29**: オプショナルGUI（feature flag）
