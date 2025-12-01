use black_hole_raytracer::*;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

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

    window: Arc<Window>,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
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
            .request_device(&wgpu::DeviceDescriptor::default())
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
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &display_shader,
                entry_point: Some("fs_main"),
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

        Self {
            renderer,
            surface,
            config,
            size,
            render_pipeline,
            render_bind_group,
            vertex_buffer,
            start_time: std::time::Instant::now(),
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

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
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
                    depth_slice: None,
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

        self.renderer.queue().submit(std::iter::once(encoder.finish()));
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

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                state.resize(physical_size);
            }
            WindowEvent::RedrawRequested => {
                state.window.request_redraw();

                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            _ => {}
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
