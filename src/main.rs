use black_hole_raytracer::*;
use clap::Parser;
use std::collections::VecDeque;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

/// Black Hole Raytracer - コマンドライン引数
#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// パフォーマンスログを記録する（バージョンタグを指定）
    #[arg(long)]
    perf_log: Option<String>,

    /// 指定秒数後に自動終了（ベンチマーク用）
    #[arg(long, value_name = "SECONDS")]
    duration: Option<f32>,

    /// デバッグモード: 計算ステップ数を可視化（ヒートマップ）
    #[arg(long)]
    debug_steps: bool,

    /// コンピュートシェーダのファイルパス
    #[arg(long, default_value = "src/ray_tracer_euler.wgsl")]
    shader: String,
}


/// パフォーマンス統計情報
struct PerformanceStats {
    /// 直近のフレームタイム（ミリ秒）
    frame_times: VecDeque<f32>,
    /// 直近のCPU時間（ミリ秒）
    cpu_times: VecDeque<f32>,
    /// 直近のGPU時間（ミリ秒、測定可能な場合）
    gpu_times: VecDeque<f32>,
    /// 保存する最大サンプル数
    max_samples: usize,
    /// 前フレームの時刻
    last_frame_time: std::time::Instant,
    /// 現在のFPS
    current_fps: f32,
    /// 現在のフレームタイム（ms）
    current_frame_time: f32,
    /// 現在のCPU時間（ms）
    current_cpu_time: f32,
    /// 現在のGPU時間（ms、測定可能な場合）
    current_gpu_time: Option<f32>,
    /// 全期間の最小FPS（最も遅かったフレーム）
    all_time_min_fps: f32,
    /// 全期間の最大FPS（最も速かったフレーム）
    all_time_max_fps: f32,
    /// ウォームアップ用のフレームカウンター（最初の数フレームを除外）
    warmup_frames_remaining: u32,
}

impl PerformanceStats {
    fn new(max_samples: usize) -> Self {
        Self {
            frame_times: VecDeque::with_capacity(max_samples),
            cpu_times: VecDeque::with_capacity(max_samples),
            gpu_times: VecDeque::with_capacity(max_samples),
            max_samples,
            last_frame_time: std::time::Instant::now(),
            current_fps: 0.0,
            current_frame_time: 0.0,
            current_cpu_time: 0.0,
            current_gpu_time: None,
            all_time_min_fps: f32::INFINITY,
            all_time_max_fps: 0.0,
            warmup_frames_remaining: 10, // 最初の10フレームをウォームアップ期間とする
        }
    }

    fn update_frame_time(&mut self) {
        let now = std::time::Instant::now();

        // ウォームアップ期間中は統計から除外
        if self.warmup_frames_remaining > 0 {
            self.warmup_frames_remaining -= 1;
            self.last_frame_time = now;

            // 最後のウォームアップフレームの場合、通知
            if self.warmup_frames_remaining == 0 {
                println!("Warmup complete. Starting performance measurement.");
            }
            return;
        }

        let delta = now.duration_since(self.last_frame_time);
        self.last_frame_time = now;

        let frame_time_ms = delta.as_secs_f32() * 1000.0;
        self.current_frame_time = frame_time_ms;
        self.current_fps = if frame_time_ms > 0.0 {
            1000.0 / frame_time_ms
        } else {
            0.0
        };

        // 全期間の最小/最大FPSを更新
        if self.current_fps > 0.0 {
            if self.current_fps < self.all_time_min_fps {
                self.all_time_min_fps = self.current_fps;
            }
            if self.current_fps > self.all_time_max_fps {
                self.all_time_max_fps = self.current_fps;
            }
        }

        self.frame_times.push_back(frame_time_ms);
        if self.frame_times.len() > self.max_samples {
            self.frame_times.pop_front();
        }
    }

    fn update_cpu_time(&mut self, cpu_time_ms: f32) {
        self.current_cpu_time = cpu_time_ms;
        self.cpu_times.push_back(cpu_time_ms);
        if self.cpu_times.len() > self.max_samples {
            self.cpu_times.pop_front();
        }
    }

    fn update_gpu_time(&mut self, gpu_time_ms: f32) {
        self.current_gpu_time = Some(gpu_time_ms);
        self.gpu_times.push_back(gpu_time_ms);
        if self.gpu_times.len() > self.max_samples {
            self.gpu_times.pop_front();
        }
    }

    /// 平均FPSを計算
    fn avg_fps(&self) -> f32 {
        if self.frame_times.is_empty() {
            return 0.0;
        }
        let avg_frame_time: f32 = self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32;
        if avg_frame_time > 0.0 {
            1000.0 / avg_frame_time
        } else {
            0.0
        }
    }

    /// 最小FPS（全期間で最も遅かったフレーム）を返す
    fn min_fps(&self) -> f32 {
        if self.all_time_min_fps == f32::INFINITY {
            0.0
        } else {
            self.all_time_min_fps
        }
    }

    /// 最大FPS（全期間で最も速かったフレーム）を返す
    fn max_fps(&self) -> f32 {
        self.all_time_max_fps
    }

    /// FPSの標準偏差を計算（安定性の指標）
    fn std_dev_fps(&self) -> f32 {
        if self.frame_times.len() < 2 {
            return 0.0;
        }
        let avg = self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32;
        let variance = self.frame_times
            .iter()
            .map(|&t| {
                let diff = t - avg;
                diff * diff
            })
            .sum::<f32>() / self.frame_times.len() as f32;
        variance.sqrt()
    }

    /// 平均CPU時間を計算
    fn avg_cpu_time(&self) -> f32 {
        if self.cpu_times.is_empty() {
            return 0.0;
        }
        self.cpu_times.iter().sum::<f32>() / self.cpu_times.len() as f32
    }

    /// 平均GPU時間を計算
    fn avg_gpu_time(&self) -> f32 {
        if self.gpu_times.is_empty() {
            return 0.0;
        }
        self.gpu_times.iter().sum::<f32>() / self.gpu_times.len() as f32
    }
}

/// CSVパフォーマンスロガー
struct PerfLogger {
    writer: csv::Writer<std::fs::File>,
    version_tag: String,
    start_time: std::time::Instant,
}

impl PerfLogger {
    fn new(version_tag: String) -> std::io::Result<Self> {
        // measurementsディレクトリを作成（存在しない場合）
        std::fs::create_dir_all("measurements")?;

        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let filename = format!("measurements/perf_log_{}_{}.csv", version_tag, timestamp);
        let file = std::fs::File::create(&filename)?;
        let mut writer = csv::Writer::from_writer(file);

        // CSVヘッダーを書き込む
        writer.write_record(&[
            "elapsed_sec",
            "version",
            "fps",
            "frame_time_ms",
            "cpu_time_ms",
            "gpu_time_ms",
            "avg_fps",
            "min_fps",
            "max_fps",
            "std_dev_fps",
            "avg_cpu_time_ms",
            "avg_gpu_time_ms",
        ])?;

        println!("Performance log created: {}", filename);

        Ok(Self {
            writer,
            version_tag,
            start_time: std::time::Instant::now(),
        })
    }

    fn log_frame(&mut self, stats: &PerformanceStats) -> std::io::Result<()> {
        let elapsed = self.start_time.elapsed().as_secs_f32();

        self.writer.write_record(&[
            format!("{:.3}", elapsed),
            self.version_tag.clone(),
            format!("{:.2}", stats.current_fps),
            format!("{:.2}", stats.current_frame_time),
            format!("{:.2}", stats.current_cpu_time),
            format!("{:.2}", stats.current_gpu_time.unwrap_or(0.0)),
            format!("{:.2}", stats.avg_fps()),
            format!("{:.2}", stats.min_fps()),
            format!("{:.2}", stats.max_fps()),
            format!("{:.2}", stats.std_dev_fps()),
            format!("{:.2}", stats.avg_cpu_time()),
            format!("{:.2}", stats.avg_gpu_time()),
        ])?;

        self.writer.flush()?;
        Ok(())
    }
}

struct State {
    renderer: BlackHoleRenderer,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    // レンダリングパイプライン関連（ディスプレイ用）
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,

    // アニメーション
    start_time: std::time::Instant,

    // パフォーマンス統計
    perf_stats: PerformanceStats,

    // GPU タイムスタンプクエリ関連
    timestamp_query_set: wgpu::QuerySet,
    timestamp_buffer: wgpu::Buffer,
    timestamp_result_buffer: wgpu::Buffer,
    timestamp_period: f32,

    // egui関連（guiフィーチャーが有効な場合のみ）
    #[cfg(feature = "gui")]
    egui_context: egui::Context,
    #[cfg(feature = "gui")]
    egui_state: egui_winit::State,
    #[cfg(feature = "gui")]
    egui_renderer: Option<egui_wgpu::Renderer>,

    window: Arc<Window>,

    // パフォーマンスロガー
    perf_logger: Option<PerfLogger>,

    // ベンチマーク自動終了
    first_frame_time: Option<std::time::Instant>,
    benchmark_duration: Option<f32>,
    should_close: bool,

    // デバッグモード
    debug_steps: bool,
}

impl State {
    async fn new(window: Arc<Window>, perf_log: Option<String>, duration: Option<f32>, debug_steps: bool, shader_path: &str) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        // 同じ instance を使って adapter を request
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find a suitable GPU adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Main Device"),
                    required_features: wgpu::Features::TIMESTAMP_QUERY,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();

        let context = GpuContext { device, queue };

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&context.device, &config);

        // BlackHoleRenderer の作成
        let renderer = BlackHoleRenderer::new_with_context(
            context,
            size.width,
            size.height,
            shader_path
        ).unwrap();

        // 描画用のシェーダ（フルスクリーンクアッド）
        let display_shader = renderer.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Display Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("display.wgsl").into()),
        });

        // サンプラーの作成
        let sampler = renderer.device().create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // レンダリングパイプラインのバインドグループレイアウト
        let render_bind_group_layout =
            renderer.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Render Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // レンダリングバインドグループの作成
        let render_bind_group = renderer.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(renderer.output_texture_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // レンダリングパイプラインの作成
        let render_pipeline_layout =
            renderer.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = renderer.device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &display_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &display_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // 頂点バッファの作成
        let vertex_buffer = renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // パフォーマンス統計の初期化
        let perf_stats = PerformanceStats::new(60); // 60フレーム分のデータを保持

        // egui の初期化（guiフィーチャーが有効な場合のみ）
        #[cfg(feature = "gui")]
        let egui_context = egui::Context::default();

        #[cfg(feature = "gui")]
        let egui_state = egui_winit::State::new(
            egui_context.clone(),
            egui::viewport::ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        );

        #[cfg(feature = "gui")]
        let egui_renderer = egui_wgpu::Renderer::new(
            renderer.device(),
            config.format,
            None,
            1,
            false,
        );

        // GPU タイムスタンプクエリの初期化
        let timestamp_query_set = renderer.device().create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Timestamp Query Set"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });

        let timestamp_buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let timestamp_result_buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp Result Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let timestamp_period = 1.0_f32;

        // パフォーマンスロガーの初期化
        let perf_logger = perf_log.and_then(|tag| {
            match PerfLogger::new(tag) {
                Ok(logger) => Some(logger),
                Err(e) => {
                    eprintln!("Failed to create performance logger: {}", e);
                    None
                }
            }
        });

        Self {
            renderer,
            surface,
            config,
            size,
            render_pipeline,
            render_bind_group,
            vertex_buffer,
            start_time: std::time::Instant::now(),
            perf_stats,
            timestamp_query_set,
            timestamp_buffer,
            timestamp_result_buffer,
            timestamp_period,
            #[cfg(feature = "gui")]
            egui_context,
            #[cfg(feature = "gui")]
            egui_state,
            #[cfg(feature = "gui")]
            egui_renderer: Some(egui_renderer),
            window,
            perf_logger,
            first_frame_time: None,
            benchmark_duration: duration,
            should_close: false,
            debug_steps,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(self.renderer.device(), &self.config);

            // TODO: renderer のリサイズ処理を実装
            // 現状では、ウィンドウリサイズに対応していません
        }
    }

    /// stats.js風のシンプルなグラフを描画
    #[cfg(feature = "gui")]
    fn draw_mini_graph(
        ui: &mut egui::Ui,
        label: &str,
        values: &VecDeque<f32>,
        current_value: f32,
        unit: &str,
        color: egui::Color32,
        height: f32,
    ) {
        ui.label(format!("{}: {:.1}{}", label, current_value, unit));

        if values.is_empty() {
            return;
        }

        let max_value = values.iter().fold(0.0f32, |a, &b| a.max(b)).max(1.0);
        let (response, painter) = ui.allocate_painter(
            egui::Vec2::new(ui.available_width(), height),
            egui::Sense::hover(),
        );

        let rect = response.rect;
        let bg_color = egui::Color32::from_black_alpha(128);
        painter.rect_filled(rect, 0.0, bg_color);

        if values.len() > 1 {
            let points: Vec<egui::Pos2> = values
                .iter()
                .enumerate()
                .map(|(i, &value)| {
                    let x = rect.left() + (i as f32 / (values.len() - 1) as f32) * rect.width();
                    let normalized = (value / max_value).clamp(0.0, 1.0);
                    let y = rect.bottom() - normalized * rect.height();
                    egui::Pos2::new(x, y)
                })
                .collect();

            painter.add(egui::Shape::line(points, egui::Stroke::new(1.5, color)));
        }

        // グリッドライン（中央）
        let mid_y = rect.center().y;
        painter.line_segment(
            [egui::Pos2::new(rect.left(), mid_y), egui::Pos2::new(rect.right(), mid_y)],
            egui::Stroke::new(0.5, egui::Color32::from_white_alpha(32)),
        );
    }

    /// ステップ数凡例UIを描画（横向きバー）
    #[cfg(feature = "gui")]
    fn draw_legend_ui(ctx: &egui::Context, max_steps: u32) {
        egui::Window::new("Step Count Legend")
            .default_pos([10.0, 200.0])
            .default_width(240.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.style_mut().text_styles.insert(
                    egui::TextStyle::Body,
                    egui::FontId::new(10.0, egui::FontFamily::Proportional),
                );

                ui.label("Computation Steps");
                ui.add_space(4.0);

                // グラデーションバーの描画（横向き）
                let bar_width = 200.0;
                let bar_height = 25.0;
                let (response, painter) = ui.allocate_painter(
                    egui::Vec2::new(bar_width, bar_height + 30.0),
                    egui::Sense::hover(),
                );

                let rect = response.rect;
                let bar_rect = egui::Rect::from_min_size(
                    rect.min,
                    egui::Vec2::new(bar_width, bar_height),
                );

                // グラデーションを描画（左から右へ：青→シアン→緑→黄→赤）
                let segments = 100;
                for i in 0..segments {
                    let t = i as f32 / segments as f32;
                    let next_t = (i + 1) as f32 / segments as f32;

                    // ステップ数に対応する色を計算（compute.wgslと同じロジック）
                    let color = if t < 0.25 {
                        let local_t = t * 4.0;
                        let r = 0.0;
                        let g = local_t;
                        let b = 1.0;
                        egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
                    } else if t < 0.5 {
                        let local_t = (t - 0.25) * 4.0;
                        let r = 0.0;
                        let g = 1.0;
                        let b = 1.0 - local_t;
                        egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
                    } else if t < 0.75 {
                        let local_t = (t - 0.5) * 4.0;
                        let r = local_t;
                        let g = 1.0;
                        let b = 0.0;
                        egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
                    } else {
                        let local_t = (t - 0.75) * 4.0;
                        let r = 1.0;
                        let g = 1.0 - local_t;
                        let b = 0.0;
                        egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
                    };

                    let x_start = bar_rect.min.x + (t * bar_width);
                    let x_end = bar_rect.min.x + (next_t * bar_width);
                    let segment_rect = egui::Rect::from_min_max(
                        egui::Pos2::new(x_start, bar_rect.min.y),
                        egui::Pos2::new(x_end, bar_rect.max.y),
                    );
                    painter.rect_filled(segment_rect, 0.0, color);
                }

                // 枠線
                painter.rect_stroke(bar_rect, 0.0, egui::Stroke::new(1.0, egui::Color32::WHITE));

                // ラベルを追加（バーの下）
                let label_y = bar_rect.max.y + 3.0;
                let labels = [
                    (0.0, "0".to_string(), "Blue"),
                    (0.25, format!("{}", (max_steps as f32 * 0.25) as u32), "Cyan"),
                    (0.5, format!("{}", (max_steps as f32 * 0.5) as u32), "Green"),
                    (0.75, format!("{}", (max_steps as f32 * 0.75) as u32), "Yellow"),
                    (1.0, format!("{}", max_steps), "Red"),
                ];

                for (t, steps, desc) in &labels {
                    let x = bar_rect.min.x + (t * bar_width);

                    // ティックマーク
                    painter.line_segment(
                        [
                            egui::Pos2::new(x, bar_rect.max.y),
                            egui::Pos2::new(x, bar_rect.max.y + 3.0),
                        ],
                        egui::Stroke::new(1.0, egui::Color32::WHITE),
                    );

                    let text = if desc.is_empty() {
                        steps.to_string()
                    } else {
                        format!("{}\n{}", steps, desc)
                    };

                    let align = if *t == 0.0 {
                        egui::Align2::LEFT_TOP
                    } else if *t == 1.0 {
                        egui::Align2::RIGHT_TOP
                    } else {
                        egui::Align2::CENTER_TOP
                    };

                    painter.text(
                        egui::Pos2::new(x, label_y + 2.0),
                        align,
                        text,
                        egui::FontId::new(8.0, egui::FontFamily::Proportional),
                        egui::Color32::WHITE,
                    );
                }

                ui.add_space(4.0);
                ui.label("← Less steps    More steps →");
                ui.label("Stronger gravitational lensing →");
            });
    }

    /// パフォーマンス統計UIを描画
    #[cfg(feature = "gui")]
    fn draw_performance_ui(ctx: &egui::Context, perf_stats: &PerformanceStats) {
        egui::Window::new("Perf")
            .default_pos([10.0, 10.0])
            .default_width(160.0)
            .resizable(false)
            .show(ctx, |ui| {
                // フォントサイズを小さく
                ui.style_mut().text_styles.insert(
                    egui::TextStyle::Body,
                    egui::FontId::new(10.0, egui::FontFamily::Proportional),
                );
                ui.style_mut().text_styles.insert(
                    egui::TextStyle::Button,
                    egui::FontId::new(10.0, egui::FontFamily::Proportional),
                );
                ui.style_mut().spacing.item_spacing.y = 2.0;

                // FPS - 現在値と統計
                ui.label(format!("FPS: {:.1}", perf_stats.current_fps));
                ui.label(format!("  Avg: {:.1}", perf_stats.avg_fps()));
                ui.label(format!("  Min: {:.1}", perf_stats.min_fps()));
                ui.label(format!("  Max: {:.1}", perf_stats.max_fps()));

                Self::draw_mini_graph(
                    ui,
                    "Frame",
                    &perf_stats.frame_times,
                    perf_stats.current_frame_time,
                    "ms",
                    egui::Color32::from_rgb(0, 255, 0),
                    28.0,
                );

                ui.add_space(2.0);

                // CPU Time
                Self::draw_mini_graph(
                    ui,
                    "CPU",
                    &perf_stats.cpu_times,
                    perf_stats.current_cpu_time,
                    "ms",
                    egui::Color32::from_rgb(255, 255, 0),
                    28.0,
                );

                ui.add_space(2.0);

                // GPU Time（測定可能な場合）
                if let Some(gpu_time) = perf_stats.current_gpu_time {
                    Self::draw_mini_graph(
                        ui,
                        "GPU",
                        &perf_stats.gpu_times,
                        gpu_time,
                        "ms",
                        egui::Color32::from_rgb(0, 191, 255),
                        28.0,
                    );
                } else {
                    ui.label("GPU: N/A");
                }
            });
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // フレーム時間の更新
        self.perf_stats.update_frame_time();

        // CPU時間測定開始
        let cpu_start = std::time::Instant::now();

        // ベンチマーク自動終了チェック
        if let Some(duration) = self.benchmark_duration {
            if self.first_frame_time.is_none() {
                // 最初のフレームの時刻を記録
                self.first_frame_time = Some(std::time::Instant::now());
                println!("Benchmark started. Will run for {} seconds.", duration);
            } else if let Some(start) = self.first_frame_time {
                let elapsed = start.elapsed().as_secs_f32();
                if elapsed >= duration {
                    println!("Benchmark duration reached ({:.2}s). Exiting...", elapsed);
                    self.should_close = true;
                }
            }
        }

        // カメラの位置を更新（ブラックホール周りを周回）
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let rotation_speed = 0.3; // 回転速度（ラジアン/秒）
        let angle = elapsed * rotation_speed;
        let radius = 15.0; // ブラックホールからの距離
        let height = 5.0; // Y座標（高さ）

        // 赤道面に沿った周回（XZ平面）
        let camera_pos = [
            radius * angle.cos(),
            height,
            radius * angle.sin(),
        ];

        // カメラを更新（常にブラックホールの中心を見る）
        let camera = Camera::new(
            camera_pos,
            [0.0, 0.0, 0.0], // ブラックホールの中心を注視
            [0.0, 1.0, 0.0], // 上方向
        );

        let scene = SceneParams {
            black_hole_position: [0.0, 0.0, 0.0],
            _padding1: 0.0,
            schwarzschild_radius: 2.0,
            screen_width: self.size.width,
            screen_height: self.size.height,
            fov: std::f32::consts::PI / 3.0,
            max_steps: 200,
            debug_mode: if self.debug_steps { 1 } else { 0 },
            _padding2: [0, 0, 0, 0, 0, 0],
        };

        // レイトレーシングを実行（タイムスタンプクエリを有効化）
        self.renderer.render_frame(&camera, &scene, Some(&self.timestamp_query_set));

        // 前フレームのGPU時間を読み取り
        {
            let slice = self.timestamp_result_buffer.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.renderer.device().poll(wgpu::Maintain::Wait);

            let data = slice.get_mapped_range();
            if data.len() >= 16 {
                let timestamps: &[u64; 2] = bytemuck::from_bytes(&data[0..16]);
                let start = timestamps[0];
                let end = timestamps[1];

                if end > start {
                    let duration_ns = (end - start) as f32 * self.timestamp_period;
                    let duration_ms = duration_ns / 1_000_000.0;
                    self.perf_stats.update_gpu_time(duration_ms);
                }
            }
            drop(data);
            self.timestamp_result_buffer.unmap();
        }

        // クエリ結果を解決（次フレームで読み取り）
        {
            let mut encoder = self
                .renderer
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Timestamp Resolve Encoder"),
                });

            encoder.resolve_query_set(&self.timestamp_query_set, 0..2, &self.timestamp_buffer, 0);
            encoder.copy_buffer_to_buffer(&self.timestamp_buffer, 0, &self.timestamp_result_buffer, 0, 16);

            self.renderer.queue().submit(std::iter::once(encoder.finish()));
        }

        // Surface に描画
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .renderer
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // レンダーパス：テクスチャを画面に描画
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.render_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..4, 0..1);
        }

        // 最初のエンコーダをサブミット
        self.renderer.queue().submit(std::iter::once(encoder.finish()));

        // egui の描画（guiフィーチャーが有効な場合のみ）
        #[cfg(feature = "gui")]
        {
            let raw_input = self.egui_state.take_egui_input(&self.window);
            let perf_stats = &self.perf_stats;
            let debug_steps = self.debug_steps;
            let max_steps = scene.max_steps;
            let full_output = self.egui_context.run(raw_input, |ctx| {
                Self::draw_performance_ui(ctx, perf_stats);
                if debug_steps {
                    Self::draw_legend_ui(ctx, max_steps);
                }
            });

            self.egui_state.handle_platform_output(&self.window, full_output.platform_output);

            let screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.size.width, self.size.height],
                pixels_per_point: self.window.scale_factor() as f32,
            };

            let paint_jobs = self.egui_context.tessellate(full_output.shapes, full_output.pixels_per_point);

            // レンダラーを一時的に取り出して使用
            if let Some(mut renderer) = self.egui_renderer.take() {
                for (id, image_delta) in &full_output.textures_delta.set {
                    renderer.update_texture(self.renderer.device(), self.renderer.queue(), *id, image_delta);
                }

                // 新しいエンコーダを作成
                let mut egui_encoder = self
                    .renderer
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("egui Encoder"),
                    });

                renderer.update_buffers(
                    self.renderer.device(),
                    self.renderer.queue(),
                    &mut egui_encoder,
                    &paint_jobs,
                    &screen_descriptor,
                );

                // RenderPassを作成し、forget_lifetime()で'staticライフタイムに変換
                {
                    let render_pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("egui Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });

                    // egui-wgpuのrenderメソッドは'static RenderPassを要求するため、
                    // forget_lifetime()を使用（公式の使用方法）
                    let mut render_pass_static = render_pass.forget_lifetime();
                    renderer.render(&mut render_pass_static, &paint_jobs[..], &screen_descriptor);
                }

                for id in &full_output.textures_delta.free {
                    renderer.free_texture(id);
                }

                // eguiエンコーダをサブミット
                self.renderer.queue().submit(std::iter::once(egui_encoder.finish()));

                // レンダラーを戻す
                self.egui_renderer = Some(renderer);
            }
        }

        // CPU時間を測定
        let cpu_elapsed = cpu_start.elapsed().as_secs_f32() * 1000.0;
        self.perf_stats.update_cpu_time(cpu_elapsed);

        // パフォーマンスログを記録
        if let Some(logger) = &mut self.perf_logger {
            if let Err(e) = logger.log_frame(&self.perf_stats) {
                eprintln!("Failed to log performance: {}", e);
            }
        }

        output.present();

        Ok(())
    }
}

struct App {
    state: Option<State>,
    args: Args,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title("Schwarzschild Black Hole Ray Tracer")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let state = rt.block_on(State::new(window, self.args.perf_log.clone(), self.args.duration, self.args.debug_steps, &self.args.shader));
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };

        // eguiにイベントを渡す（guiフィーチャーが有効な場合のみ）
        #[cfg(feature = "gui")]
        let egui_response = state.egui_state.on_window_event(&state.window, &event);

        // eguiが処理していないイベントのみ処理する
        #[cfg(feature = "gui")]
        let consumed = egui_response.consumed;

        #[cfg(not(feature = "gui"))]
        let consumed = false;

        if !consumed {
            match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::Resized(physical_size) => {
                    state.resize(physical_size);
                }
                _ => {}
            }
        }

        // RedrawRequestedは常に処理
        if matches!(event, WindowEvent::RedrawRequested) {
            state.window.request_redraw();

            match state.render() {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                Err(e) => eprintln!("{:?}", e),
            }

            // ベンチマーク自動終了チェック
            if state.should_close {
                event_loop.exit();
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App {
        state: None,
        args,
    };
    event_loop.run_app(&mut app).unwrap();
}
