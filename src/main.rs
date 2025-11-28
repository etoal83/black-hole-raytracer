use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

// カメラ設定（シェーダと同じメモリレイアウト）
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Camera {
    position: [f32; 3],
    _padding1: f32,
    forward: [f32; 3],
    _padding2: f32,
    right: [f32; 3],
    _padding3: f32,
    up: [f32; 3],
    _padding4: f32,
}

impl Camera {
    fn new(position: [f32; 3], look_at: [f32; 3], up: [f32; 3]) -> Self {
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

// シーンパラメータ（シェーダと同じメモリレイアウト）
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SceneParams {
    black_hole_position: [f32; 3],
    schwarzschild_radius: f32,
    screen_width: u32,
    screen_height: u32,
    fov: f32,
    max_steps: u32,
}

// ベクトル演算ヘルパー関数
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        v
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// 頂点データ（フルスクリーンクアッド用）
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
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

// フルスクリーンクアッドの頂点
const QUAD_VERTICES: &[Vertex] = &[
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

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    // コンピュートシェーダ関連
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    output_texture: wgpu::Texture,

    // レンダリングパイプライン関連
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,

    // ユニフォームバッファ
    camera_buffer: wgpu::Buffer,
    scene_buffer: wgpu::Buffer,

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

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

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

        surface.configure(&device, &config);

        // カメラの初期化（ブラックホールを見る位置）
        let camera = Camera::new(
            [0.0, 5.0, 15.0],  // カメラ位置
            [0.0, 0.0, 0.0],   // 注視点（ブラックホールの位置）
            [0.0, 1.0, 0.0],   // 上方向
        );

        // シーンパラメータの初期化
        let scene = SceneParams {
            black_hole_position: [0.0, 0.0, 0.0],
            schwarzschild_radius: 2.0,  // ブラックホールの半径
            screen_width: size.width,
            screen_height: size.height,
            fov: std::f32::consts::PI / 3.0,  // 60度
            max_steps: 500,  // レイトレーシングの最大ステップ数
        };

        // ユニフォームバッファの作成
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let scene_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scene Buffer"),
            contents: bytemuck::cast_slice(&[scene]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 出力テクスチャの作成
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output Texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // コンピュートシェーダのロード
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute.wgsl").into()),
        });

        // コンピュートパイプラインのバインドグループレイアウト
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                ],
            });

        // コンピュートバインドグループの作成
        let output_texture_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            ],
        });

        // コンピュートパイプラインの作成
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // 描画用のシェーダ（フルスクリーンクアッド）
        let display_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Display Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("display.wgsl").into()),
        });

        // サンプラーの作成
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
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
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&output_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // レンダリングパイプラインの作成
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            compute_pipeline,
            compute_bind_group,
            output_texture,
            render_pipeline,
            render_bind_group,
            vertex_buffer,
            camera_buffer,
            scene_buffer,
            start_time: std::time::Instant::now(),
            window,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // 出力テクスチャを再作成
            self.output_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Output Texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            // バインドグループを再作成（省略 - 必要に応じて実装）
            // シーンパラメータのサイズを更新
            let scene = SceneParams {
                black_hole_position: [0.0, 0.0, 0.0],
                schwarzschild_radius: 2.0,
                screen_width: new_size.width,
                screen_height: new_size.height,
                fov: std::f32::consts::PI / 3.0,
                max_steps: 500,
            };
            self.queue.write_buffer(&self.scene_buffer, 0, bytemuck::cast_slice(&[scene]));
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

        // カメラバッファを更新
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera]));

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // コンピュートパス：レイトレーシング
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);

            // ワークグループ数の計算（8x8のワークグループサイズ）
            let workgroup_count_x = (self.size.width + 7) / 8;
            let workgroup_count_y = (self.size.height + 7) / 8;
            compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
        }

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

        self.queue.submit(std::iter::once(encoder.finish()));
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
