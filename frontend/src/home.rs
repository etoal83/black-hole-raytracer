use web_sys::HtmlCanvasElement;
use wgpu::*;
use zoon::{println, named_color::*, *};

const CANVAS_WIDTH: u32 = 640;
const CANVAS_HEIGHT: u32 = 480;


struct GraphicContext {
    surface: Surface,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    render_pipeline: RenderPipeline,
}


#[static_ref]
fn graphic_context() -> &'static Mutable<Option<SendWrapper<GraphicContext>>> {
    Mutable::new(None)
}

#[static_ref]
fn animation_loop() -> &'static Mutable<Option<SendWrapper<AnimationLoop>>> {
    Mutable::new(None)
}


fn set_graphic_context(canvas: HtmlCanvasElement) {
    Task::start(async {
        // let instance = wgpu::Instance::new(Backends::all());
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            dx12_shader_compiler: Default::default(),
        });
        let surface = instance.create_surface_from_canvas(canvas).unwrap();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    features: Features::empty(),
                    limits: Limits::downlevel_webgl2_defaults(),
                    label: None,
                },
                None
            )
            .await
            .unwrap();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: CANVAS_WIDTH,
            height: CANVAS_HEIGHT,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(include_wgsl!("./shader/shader.wgsl"));
        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Render pipeline layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: config.format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                polygon_mode: PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        graphic_context().set(Some(SendWrapper::new(GraphicContext {
            surface,
            device,
            queue,
            config,
            render_pipeline,
        })));

        start_animation();
        println!("[INFO] GraphicContext set");
    });
}

fn render() {
    graphic_context().use_ref(|ctx| {
        let ctx = ctx.as_ref().unwrap_throw();
        let output = ctx.surface.get_current_texture().unwrap_throw();
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render encoder") });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.,
                        }),
                        store: true,
                    }
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&ctx.render_pipeline);
            render_pass.draw(0..3, 0..1);
        }

        ctx.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    })
}

fn remove_canvas_context() {
    graphic_context().take();
    animation_loop().take();
}

fn start_animation() {
    let rendering_loop = AnimationLoop::new(|_| {
        render();
    });
    animation_loop().set(Some(SendWrapper::new(rendering_loop)));
}

// ------ ------
//     View
// ------ ------

pub fn page_content() -> impl Element {
    Column::new()
        .item(Canvas::new()
            .width(CANVAS_WIDTH)
            .height(CANVAS_HEIGHT)
            .s(Background::new().color(hsluv!(0., 0., 0.)))
            .s(Borders::all(Border::new().color(GRAY_7)))
            .after_insert(set_graphic_context)
            .after_remove(|_| remove_canvas_context()))
}
