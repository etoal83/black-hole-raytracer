// Black Hole Ray Tracer - Core Library
//! Schwarzschild ブラックホールのレイトレーシングエンジン
//!
//! このライブラリは、WGSL コンピュートシェーダを使用して
//! ブラックホール周辺の光の軌道をシミュレートします。

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ============================================================================
// 公開 API: 基本データ構造
// ============================================================================

/// カメラ設定（シェーダと同じメモリレイアウト）
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Camera {
    pub position: [f32; 3],
    _padding1: f32,
    pub forward: [f32; 3],
    _padding2: f32,
    pub right: [f32; 3],
    _padding3: f32,
    pub up: [f32; 3],
    _padding4: f32,
}

impl Camera {
    /// 新しいカメラを作成
    ///
    /// # Arguments
    /// * `position` - カメラの位置
    /// * `look_at` - 注視点
    /// * `up` - 上方向ベクトル
    pub fn new(position: [f32; 3], look_at: [f32; 3], up: [f32; 3]) -> Self {
        // forward ベクトルの計算
        let forward = normalize([
            look_at[0] - position[0],
            look_at[1] - position[1],
            look_at[2] - position[2],
        ]);

        // right ベクトルの計算 (forward × up)
        let right = normalize(cross(forward, up));

        // up ベクトルの再計算 (right × forward)
        let up = normalize(cross(right, forward));

        Self {
            position,
            _padding1: 0.0,
            forward,
            _padding2: 0.0,
            right,
            _padding3: 0.0,
            up,
            _padding4: 0.0,
        }
    }
}

/// シーンパラメータ（シェーダと同じメモリレイアウト）
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SceneParams {
    pub black_hole_position: [f32; 3],
    pub _padding1: f32,  // vec3のアライメント調整（16バイト境界）
    pub schwarzschild_radius: f32,
    pub screen_width: u32,
    pub screen_height: u32,
    pub fov: f32,
    pub max_steps: u32,
    pub debug_mode: u32,  // 0=通常モード, 1=ステップ数可視化モード
    pub _padding2: [u32; 6],   // 構造体を64バイトに調整（vec4<u32> + 8バイト追加パディング）
}

/// 頂点データ（フルスクリーンクアッド用）
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        }
    }
}

/// フルスクリーンクアッドの頂点
pub const QUAD_VERTICES: &[Vertex] = &[
    Vertex {
        position: [-1.0, -1.0],
    },
    Vertex {
        position: [1.0, -1.0],
    },
    Vertex {
        position: [-1.0, 1.0],
    },
    Vertex {
        position: [1.0, 1.0],
    },
];

// ============================================================================
// ユーティリティ関数
// ============================================================================

/// ベクトルの正規化
pub fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        v
    }
}

/// ベクトルの外積
pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ============================================================================
// GPU コンテキスト
// ============================================================================

/// GPU デバイスとキューを管理する構造体
///
/// Surface に依存せず、コンピュートシェーダ専用の GPU コンテキストを提供します。
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// 新しい GPU コンテキストを作成（Surface なし）
    ///
    /// この関数は Jupyter Notebook などの非ウィンドウ環境で使用します。
    pub async fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find a suitable GPU adapter"))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await?;

        Ok(Self { device, queue })
    }

    /// Surface を持つ GPU コンテキストを作成
    ///
    /// ウィンドウアプリケーション用のコンテキストを作成します。
    pub async fn new_with_surface(
        surface: &wgpu::Surface<'_>,
    ) -> anyhow::Result<(Self, wgpu::Adapter)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find a suitable GPU adapter"))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await?;

        Ok((Self { device, queue }, adapter))
    }
}

// ============================================================================
// テクスチャ読み込みヘルパー
// ============================================================================

/// 画像ファイルからテクスチャを作成（PNG/JPEG/EXR対応）
fn load_texture_from_file(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &str,
) -> anyhow::Result<wgpu::Texture> {
    use std::path::Path;

    let path_obj = Path::new(path);
    let extension = path_obj
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    let (width, height, rgba) = if extension.eq_ignore_ascii_case("exr") {
        // EXRファイルの読み込み
        load_exr_image(path)?
    } else {
        // 通常の画像フォーマット（PNG、JPEG等）の読み込み
        let img = image::open(path)?.to_rgba8();
        let (w, h) = img.dimensions();
        (w, h, img.into_raw())
    };

    // テクスチャを作成
    let texture_size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Skybox Texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // テクスチャにデータを書き込み
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        texture_size,
    );

    Ok(texture)
}

/// EXRファイルを読み込んでRGBA8に変換
fn load_exr_image(path: &str) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    use exr::prelude::*;

    // EXRファイルを読み込み
    let image = read_first_rgba_layer_from_file(
        path,
        |resolution, _| {
            // ピクセルデータを格納するバッファを作成
            vec![vec![(0.0, 0.0, 0.0, 0.0); resolution.width()]; resolution.height()]
        },
        |buffer, position, (r, g, b, a): (f32, f32, f32, f32)| {
            // HDRピクセルデータを読み込み
            buffer[position.y()][position.x()] = (r, g, b, a);
        },
    )?;

    let width = image.layer_data.size.width() as u32;
    let height = image.layer_data.size.height() as u32;

    // HDR -> LDR変換（簡易トーンマッピング）
    let mut rgba8 = Vec::with_capacity((width * height * 4) as usize);

    for row in image.layer_data.channel_data.pixels.iter() {
        for &(r, g, b, a) in row.iter() {
            // 簡易Reinhardトーンマッピング: x / (1 + x)
            let tone_map = |x: f32| -> u8 {
                let mapped = x / (1.0 + x);
                (mapped.clamp(0.0, 1.0) * 255.0) as u8
            };

            rgba8.push(tone_map(r));
            rgba8.push(tone_map(g));
            rgba8.push(tone_map(b));
            rgba8.push((a.clamp(0.0, 1.0) * 255.0) as u8);
        }
    }

    Ok((width, height, rgba8))
}

// ============================================================================
// Black Hole Renderer
// ============================================================================

/// ブラックホールレイトレーシングエンジン
///
/// コンピュートシェーダを使用してブラックホール周辺の光の軌道をシミュレートします。
pub struct BlackHoleRenderer {
    context: GpuContext,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    output_texture: wgpu::Texture,
    output_texture_view: wgpu::TextureView,
    camera_buffer: wgpu::Buffer,
    scene_buffer: wgpu::Buffer,
    skybox_texture: wgpu::Texture,
    skybox_texture_view: wgpu::TextureView,
    skybox_sampler: wgpu::Sampler,
    width: u32,
    height: u32,
}

impl BlackHoleRenderer {
    /// 新しいレンダラーを作成（新しい GPU コンテキストを作成）
    ///
    /// # Arguments
    /// * `width` - 出力画像の幅
    /// * `height` - 出力画像の高さ
    pub async fn new(width: u32, height: u32) -> anyhow::Result<Self> {
        let context = GpuContext::new().await?;
        Self::new_with_context(context, width, height)
    }

    /// 既存の GPU コンテキストを使用してレンダラーを作成
    ///
    /// # Arguments
    /// * `context` - 既存の GPU コンテキスト
    /// * `width` - 出力画像の幅
    /// * `height` - 出力画像の高さ
    pub fn new_with_context(context: GpuContext, width: u32, height: u32) -> anyhow::Result<Self> {

        // 初期カメラとシーンの設定
        let camera = Camera::new(
            [0.0, 5.0, 15.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        );

        let scene = SceneParams {
            black_hole_position: [0.0, 0.0, 0.0],
            _padding1: 0.0,
            schwarzschild_radius: 2.0,
            screen_width: width,
            screen_height: height,
            fov: std::f32::consts::PI / 3.0,
            max_steps: 500,
            debug_mode: 0,
            _padding2: [0, 0, 0, 0, 0, 0],
        };

        // ユニフォームバッファの作成
        let camera_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let scene_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scene Buffer"),
            contents: bytemuck::cast_slice(&[scene]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 出力テクスチャの作成
        let output_texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let output_texture_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // スカイボックステクスチャの読み込み
        let skybox_texture = load_texture_from_file(
            &context.device,
            &context.queue,
            "assets/starmap_2020_4k.exr",
        )?;
        let skybox_texture_view = skybox_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // スカイボックスサンプラーの作成
        let skybox_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // コンピュートシェーダのロード
        let compute_shader = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ray_tracer_euler.wgsl").into()),
        });

        // コンピュートパイプラインのバインドグループレイアウト
        let compute_bind_group_layout =
            context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // コンピュートバインドグループの作成
        let compute_bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&output_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scene_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&skybox_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&skybox_sampler),
                },
            ],
        });

        // コンピュートパイプラインの作成
        let compute_pipeline_layout =
            context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            context,
            compute_pipeline,
            compute_bind_group,
            output_texture,
            output_texture_view,
            camera_buffer,
            scene_buffer,
            skybox_texture,
            skybox_texture_view,
            skybox_sampler,
            width,
            height,
        })
    }

    /// フレームをレンダリング
    ///
    /// カメラとシーンパラメータを更新して、レイトレーシングを実行します。
    /// オプショナルでタイムスタンプクエリセットを受け取り、GPU時間を測定できます。
    pub fn render_frame(
        &mut self,
        camera: &Camera,
        scene: &SceneParams,
        timestamp_query: Option<&wgpu::QuerySet>,
    ) {
        // バッファを更新
        self.context.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[*camera]));
        self.context.queue.write_buffer(&self.scene_buffer, 0, bytemuck::cast_slice(&[*scene]));

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // コンピュートパス：レイトレーシング
        {
            let timestamp_writes = timestamp_query.map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes,
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);

            // ワークグループ数の計算（8x8のワークグループサイズ）
            let workgroup_count_x = (self.width + 7) / 8;
            let workgroup_count_y = (self.height + 7) / 8;
            compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
        }

        self.context.queue.submit(std::iter::once(encoder.finish()));
    }

    /// 出力テクスチャのビューを取得
    ///
    /// レンダリングパイプラインでテクスチャを表示する際に使用します。
    pub fn output_texture_view(&self) -> &wgpu::TextureView {
        &self.output_texture_view
    }

    /// GPU デバイスへの参照を取得
    pub fn device(&self) -> &wgpu::Device {
        &self.context.device
    }

    /// GPU キューへの参照を取得
    pub fn queue(&self) -> &wgpu::Queue {
        &self.context.queue
    }

    /// テクスチャから画像データを取得
    ///
    /// GPU テクスチャのデータを CPU メモリにコピーして、RGBA8 形式のバイト配列として返します。
    /// Jupyter Notebook で画像を表示する際に使用します。
    pub async fn get_image_data(&self) -> anyhow::Result<Vec<u8>> {
        // テクスチャのサイズを計算
        let bytes_per_pixel = 4; // RGBA8
        let unpadded_bytes_per_row = self.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
        let buffer_size = (padded_bytes_per_row * self.height) as u64;

        // ステージングバッファの作成
        let staging_buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // テクスチャからバッファにコピー
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Encoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.context.queue.submit(std::iter::once(encoder.finish()));

        // バッファをマップして読み取り（tokio で非同期に待つ）
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = tokio::sync::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // デバイスをポーリングしてマッピングを完了
        self.context.device.poll(wgpu::Maintain::Wait);
        receiver.await??;

        let data = buffer_slice.get_mapped_range();

        // パディングを除去して画像データを抽出
        let mut image_data = Vec::with_capacity((self.width * self.height * bytes_per_pixel) as usize);
        for row in 0..self.height {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + unpadded_bytes_per_row as usize;
            image_data.extend_from_slice(&data[start..end]);
        }

        drop(data);
        staging_buffer.unmap();

        Ok(image_data)
    }

    /// 画像をファイルに保存
    ///
    /// # Arguments
    /// * `path` - 保存先のファイルパス（拡張子で形式を自動判定）
    pub async fn save_image(&self, path: &str) -> anyhow::Result<()> {
        let image_data = self.get_image_data().await?;

        // image クレートで画像を保存
        let img = image::RgbaImage::from_raw(self.width, self.height, image_data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image from data"))?;

        img.save(path)?;

        Ok(())
    }
}

// ============================================================================
// Jupyter Notebook 用ヘルパー
// ============================================================================

/// async 関数をブロッキング実行するヘルパー関数
///
/// Jupyter Notebook 内で async 関数を簡単に実行するために使用します。
pub fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Runtime::new()
        .expect("Failed to create Tokio runtime")
        .block_on(future)
}
