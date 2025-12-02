use black_hole_raytracer::*;
use std::collections::VecDeque;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};


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
        }
    }

    fn update_frame_time(&mut self) {
        let now = std::time::Instant::now();
        let delta = now.duration_since(self.last_frame_time);
        self.last_frame_time = now;

        let frame_time_ms = delta.as_secs_f32() * 1000.0;
        self.current_frame_time = frame_time_ms;
        self.current_fps = if frame_time_ms > 0.0 {
            1000.0 / frame_time_ms
        } else {
            0.0
        };

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

    // egui関連（guiフィーチャーが有効な場合のみ）
    #[cfg(feature = "gui")]
    egui_context: egui::Context,
    #[cfg(feature = "gui")]
    egui_state: egui_winit::State,
    #[cfg(feature = "gui")]
    egui_renderer: Option<egui_wgpu::Renderer>,

    window: Arc<Window>,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
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
            .request_device(&wgpu::DeviceDescriptor::default(), None)
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
        let renderer = BlackHoleRenderer::new_with_context(context, size.width, size.height)
            .unwrap();

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
            #[cfg(feature = "gui")]
            egui_context,
            #[cfg(feature = "gui")]
            egui_state,
            #[cfg(feature = "gui")]
            egui_renderer: Some(egui_renderer),
            window,
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

                // FPS
                ui.label(format!("FPS: {:.1}", perf_stats.current_fps));
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
            schwarzschild_radius: 2.0,
            screen_width: self.size.width,
            screen_height: self.size.height,
            fov: std::f32::consts::PI / 3.0,
            max_steps: 500,
        };

        // レイトレーシングを実行
        self.renderer.render_frame(&camera, &scene);

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
            let full_output = self.egui_context.run(raw_input, |ctx| {
                Self::draw_performance_ui(ctx, perf_stats);
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

        output.present();

        Ok(())
    }
}

struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title("Schwarzschild Black Hole Ray Tracer")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let state = rt.block_on(State::new(window));
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
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App { state: None };
    event_loop.run_app(&mut app).unwrap();
}
