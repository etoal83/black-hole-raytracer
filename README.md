# Schwarzschild Black Hole Ray Tracer

WGSL コンピュートシェーダを使用したシュワルツシルトブラックホールのレイトレーシングエンジン。

## 機能

- **リアルタイムアニメーション**: ブラックホール周りを周回するカメラビュー
- **GPU コンピュートシェーダ**: wgpu を使用した高速レイトレーシング
- **JupyterLab サポート**: インタラクティブなパラメータ調整と可視化
- **ハイブリッド構成**: ライブラリ + バイナリの両方を提供

## 必要要件

- Rust (最新の stable 版)
- Python 3.11 以上
- [just](https://github.com/casey/just) (タスクランナー)
- [uv](https://docs.astral.sh/uv/) (Python パッケージマネージャー)
- GPU (Metal, Vulkan, または DirectX 12 対応)

## クイックスタート

### 1. 必要なツールのインストール

```bash
# just のインストール（macOS）
brew install just

# uv のインストール
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 環境セットアップ（初回のみ）

```bash
# すべての環境をセットアップ
just setup
```

このコマンドで以下が自動的に実行されます：
- Python 仮想環境の作成
- JupyterLab と ipykernel のインストール
- evcxr_jupyter (Rust カーネル) のインストール
- Jupyter カーネルの登録

### 3. JupyterLab の起動

```bash
# JupyterLab を起動
just lab
```

ブラウザが開いて JupyterLab が起動します。`black_hole_rendering.ipynb` を開いて、インタラクティブにレンダリングを試すことができます。

### 4. CLI アプリケーションの実行

```bash
# リアルタイムアニメーションを表示
just run
```

## 利用可能なタスク

```bash
just --list          # すべてのタスクを表示
just setup           # 環境セットアップ（初回のみ）
just lab             # JupyterLab を起動
just run             # CLI アプリケーションを実行
just build           # リリースビルド
just build-dev       # デバッグビルド
just test            # テストを実行
just fmt             # コードをフォーマット
just lint            # Linter を実行
just update          # すべての依存関係を更新
just kernels         # インストール済み Jupyter カーネルを表示
just info            # プロジェクト情報を表示
just clean           # ビルド成果物をクリーンアップ
just clean-all       # 完全クリーンアップ（仮想環境も削除）
```

## プロジェクト構成

```
black-hole-raytracer/
├── Cargo.toml              # Rust プロジェクト設定
├── pyproject.toml          # Python 依存関係
├── justfile                # タスクランナー設定
├── .python-version         # Python バージョン固定
├── .venv/                  # Python 仮想環境 (自動生成)
├── src/
│   ├── lib.rs             # コアライブラリ
│   ├── main.rs            # CLI アプリケーション
│   ├── compute.wgsl       # レイトレーシングシェーダ
│   └── display.wgsl       # ディスプレイシェーダ
├── notebooks/
│   └── black_hole_rendering.ipynb  # サンプルノートブック
└── README.md
```

## ライブラリの使用方法

### Rust から使用

```rust
use black_hole_raytracer::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // レンダラーの作成
    let mut renderer = BlackHoleRenderer::new(800, 600).await?;

    // カメラとシーンの設定
    let camera = Camera::new(
        [15.0, 5.0, 0.0],  // 位置
        [0.0, 0.0, 0.0],   // 注視点
        [0.0, 1.0, 0.0],   // 上方向
    );

    let scene = SceneParams {
        black_hole_position: [0.0, 0.0, 0.0],
        schwarzschild_radius: 2.0,
        screen_width: 800,
        screen_height: 600,
        fov: std::f32::consts::PI / 3.0,
        max_steps: 500,
    };

    // レンダリング実行
    renderer.render_frame(&camera, &scene);

    // 画像を保存
    renderer.save_image("output.png").await?;

    Ok(())
}
```

### JupyterLab から使用

JupyterLab で新しいノートブックを作成し、カーネルを **Rust** に変更してから以下を実行：

```rust
// 依存関係の追加
:dep black_hole_raytracer = { path = ".." }

use black_hole_raytracer::*;

// レンダラーの初期化
let context = block_on(GpuContext::new())?;
let mut renderer = BlackHoleRenderer::new_with_context(context, 800, 600)?;

// カメラとシーンの設定
let camera = Camera::new([15.0, 5.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
let scene = SceneParams {
    black_hole_position: [0.0, 0.0, 0.0],
    schwarzschild_radius: 2.0,
    screen_width: 800,
    screen_height: 600,
    fov: std::f32::consts::PI / 3.0,
    max_steps: 500,
};

// レンダリングと保存
renderer.render_frame(&camera, &scene);
block_on(renderer.save_image("black_hole.png"))?;
```

## API ドキュメント

### 主要な構造体

- **`Camera`**: カメラの位置と向きを定義
- **`SceneParams`**: シーンパラメータ（ブラックホール位置、シュワルツシルト半径など）
- **`GpuContext`**: GPU デバイスとキューを管理
- **`BlackHoleRenderer`**: レイトレーシングエンジンのメインクラス

### 主要なメソッド

- **`BlackHoleRenderer::new(width, height)`**: 新しいレンダラーを作成
- **`render_frame(&camera, &scene)`**: フレームをレンダリング
- **`get_image_data()`**: GPU から画像データを取得
- **`save_image(path)`**: 画像をファイルに保存
- **`block_on(future)`**: JupyterLab 用の同期実行ヘルパー

## トラブルシューティング

### JupyterLab で Rust カーネルが表示されない

```bash
# カーネルを再インストール
evcxr_jupyter --install

# インストール済みカーネルを確認
just kernels
```

### Python 仮想環境が見つからない

```bash
# 環境を再セットアップ
just setup
```

### 依存関係の更新

```bash
# すべての依存関係を更新
just update
```

## ライセンス

このプロジェクトは実験的なものです。
