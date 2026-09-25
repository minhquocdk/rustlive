// src/main.rs
use bytemuck::{Pod, Zeroable};
use glam::{EulerRot, Mat4, Quat, Vec2, Vec3, Vec4};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

// ----------------------------------------------------------------------------
// Cấu hình
// ----------------------------------------------------------------------------
const PARTICLE_COUNT: usize = 20_000;
const SYMBOL_CELL: u32 = 64;

const CORE_COLORS: [[f32; 3]; 4] = [
    [0.000, 0.725, 0.361], // #00B95C
    [1.000, 0.800, 0.000], // #FFCC00
    [1.000, 0.275, 0.255], // #FF4641
    [0.192, 0.525, 1.000], // #3186FF
];

const ACCENT_COLORS: [[f32; 3]; 4] = [
    [0.000, 0.647, 0.718], // #00A5B7
    [1.000, 0.420, 0.169], // #FF6B2B
    [0.847, 0.384, 0.494], // #D8627E
    [0.663, 0.459, 0.667], // #A975AA
];

const SCATTER_TOP: f32 = 1.0;
const SCATTER_BOTTOM: f32 = 0.15;
const SHRINK_SPEED: f32 = 15.0;
const ENTRANCE_DELAY: f32 = 0.5;
const ENTRANCE_GROW_SPEED: f32 = 1.2;
const SPARK_CURVE_N: f32 = 0.7;
const ROUNDING_FACTOR: f32 = 0.18;

// Runes: ᚵ ᛟ ᚼ ᛡ ᛞ ᚦ ᛝ ᚦ ᛈ  (định nghĩa bằng nét vẽ, tọa độ chuẩn hoá 0..1)
const GLYPHS: &[&[[f32; 4]]] = &[
    // ᚵ  (Gyfu)
    &[[0.22, 0.15, 0.78, 0.85], [0.78, 0.15, 0.22, 0.85]],
    // ᛟ  (Othala)
    &[
        [0.50, 0.10, 0.85, 0.45],
        [0.85, 0.45, 0.50, 0.90],
        [0.50, 0.90, 0.15, 0.45],
        [0.15, 0.45, 0.50, 0.10],
        [0.50, 0.90, 0.50, 0.98],
    ],
    // ᚼ  (Hagall)
    &[
        [0.25, 0.12, 0.25, 0.88],
        [0.75, 0.12, 0.75, 0.88],
        [0.25, 0.50, 0.75, 0.50],
    ],
    // ᛡ  (Ior)
    &[
        [0.50, 0.12, 0.50, 0.88],
        [0.50, 0.36, 0.80, 0.16],
        [0.50, 0.64, 0.80, 0.84],
    ],
    // ᛞ  (Dagaz)
    &[
        [0.20, 0.20, 0.80, 0.80],
        [0.80, 0.20, 0.20, 0.80],
        [0.20, 0.20, 0.20, 0.80],
        [0.80, 0.20, 0.80, 0.80],
    ],
    // ᚦ  (Thurisaz)
    &[
        [0.50, 0.10, 0.50, 0.90],
        [0.50, 0.32, 0.80, 0.60],
        [0.50, 0.32, 0.20, 0.60],
    ],
    // ᛝ  (Ing)
    &[
        [0.50, 0.12, 0.82, 0.50],
        [0.82, 0.50, 0.50, 0.88],
        [0.50, 0.88, 0.18, 0.50],
        [0.18, 0.50, 0.50, 0.12],
    ],
    // ᚦ
    &[
        [0.50, 0.10, 0.50, 0.90],
        [0.50, 0.32, 0.80, 0.60],
        [0.50, 0.32, 0.20, 0.60],
    ],
    // ᛈ  (Wunjo)
    &[
        [0.50, 0.12, 0.50, 0.88],
        [0.50, 0.12, 0.80, 0.36],
        [0.50, 0.50, 0.80, 0.74],
    ],
];

// ----------------------------------------------------------------------------
// RNG nhỏ gọn (xorshift32)
// ----------------------------------------------------------------------------
struct Rng(u32);
impl Rng {
    fn new(seed: u32) -> Self {
        Self(if seed == 0 { 0x9E3779B9 } else { seed })
    }
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x >> 8) as f32 / 16_777_216.0
    }
}

// ----------------------------------------------------------------------------
// Atlas ký tự
// ----------------------------------------------------------------------------
fn seg_dist(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let vx = bx - ax;
    let vy = by - ay;
    let wx = px - ax;
    let wy = py - ay;
    let len2 = vx * vx + vy * vy;
    let t = if len2 > 1e-6 {
        ((wx * vx + wy * vy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let dx = wx - vx * t;
    let dy = wy - vy * t;
    (dx * dx + dy * dy).sqrt()
}

fn build_atlas() -> (Vec<u8>, u32, u32) {
    let count = GLYPHS.len() as u32;
    let w = SYMBOL_CELL * count;
    let h = SYMBOL_CELL;
    let mut data = vec![0u8; (w * h * 4) as usize];

    for (gi, strokes) in GLYPHS.iter().enumerate() {
        let base_x = gi as u32 * SYMBOL_CELL;
        for y in 0..SYMBOL_CELL {
            for x in 0..SYMBOL_CELL {
                let fx = (x as f32 + 0.5) / SYMBOL_CELL as f32;
                let fy = (y as f32 + 0.5) / SYMBOL_CELL as f32;
                let mut d = f32::MAX;
                for s in strokes.iter() {
                    d = d.min(seg_dist(fx, fy, s[0], s[1], s[2], s[3]));
                }
                let dpx = d * SYMBOL_CELL as f32;
                let alpha = ((2.4 - dpx) / 1.4).clamp(0.0, 1.0);
                let idx = (((y * w) + base_x + x) * 4) as usize;
                data[idx] = 255;
                data[idx + 1] = 255;
                data[idx + 2] = 255;
                data[idx + 3] = (alpha * 255.0) as u8;
            }
        }
    }
    (data, w, h)
}

// ----------------------------------------------------------------------------
// GPU structs
// ----------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Instance {
    pos: [f32; 3],
    color: [f32; 3],
    size: f32,
    sym: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    model: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    px_scale: f32,
    symbol_count: f32,
    _pad: [f32; 2],
}

struct Wave {
    radius: f32,
    state_index: usize,
    width: f32,
    speed: f32,
}

// ----------------------------------------------------------------------------
// State
// ----------------------------------------------------------------------------
struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group0: wgpu::BindGroup,
    bind_group1: wgpu::BindGroup,
    corner_buf: wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    instances: Vec<Instance>,

    angles: Vec<f32>,
    radii: Vec<f32>,
    z_offsets: Vec<f32>,
    speed_scales: Vec<f32>,
    symbol_indices: Vec<f32>,

    anim_time: f32,
    shape_morph_angle: f32,
    entrance_scale: f32,
    state_index: usize,
    waves: Vec<Wave>,

    mouse_ndc: Vec2,
    hovered: bool,

    yaw: f32,
    pitch: f32,
    distance: f32,
    last_cursor: Option<Vec2>,
    dragging: bool,

    pixel_ratio: f32,
    last_time: Instant,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .expect("không tạo được surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("không tìm thấy adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("gemini-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("không tạo được device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // ---- Atlas texture ----
        let (atlas_data, atlas_w, atlas_h) = build_atlas();
        let tex_size = wgpu::Extent3d {
            width: atlas_w,
            height: atlas_h,
            depth_or_array_layers: 1,
        };
        let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("symbol-atlas"),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &atlas_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(atlas_w * 4),
                rows_per_image: Some(atlas_h),
            },
            tex_size,
        );
        let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // ---- Uniforms ----
        let uniforms = Uniforms {
            model: Mat4::IDENTITY.to_cols_array_2d(),
            view: Mat4::IDENTITY.to_cols_array_2d(),
            proj: Mat4::IDENTITY.to_cols_array_2d(),
            px_scale: 1.0,
            symbol_count: GLYPHS.len() as f32,
            _pad: [0.0; 2],
        };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl-uniform"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg-uniform"),
            layout: &bgl0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl-texture"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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
        let bind_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg-texture"),
            layout: &bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // ---- Shader ----
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline-layout"),
            bind_group_layouts: &[&bgl0, &bgl1],
            push_constant_ranges: &[],
        });

        const CORNER_ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];
        const INSTANCE_ATTRS: [wgpu::VertexAttribute; 4] =
            wgpu::vertex_attr_array![1 => Float32x3, 2 => Float32x3, 3 => Float32, 4 => Float32];

        let corner_layout = wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &CORNER_ATTRS,
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_ATTRS,
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[corner_layout, instance_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ---- Buffers ----
        let corners: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];
        let corner_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("corners"),
            contents: bytemuck::cast_slice(&corners),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let instances = vec![
            Instance {
                pos: [0.0; 3],
                color: [0.0; 3],
                size: 1.0,
                sym: 0.0,
            };
            PARTICLE_COUNT
        ];
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // ---- Particle data ----
        let mut rng = Rng::new(0xC0FFEE);
        let mut angles = Vec::with_capacity(PARTICLE_COUNT);
        let mut radii = Vec::with_capacity(PARTICLE_COUNT);
        let mut z_offsets = Vec::with_capacity(PARTICLE_COUNT);
        let mut speed_scales = Vec::with_capacity(PARTICLE_COUNT);
        let mut symbol_indices = Vec::with_capacity(PARTICLE_COUNT);
        let glyph_count = GLYPHS.len() as f32;

        for _ in 0..PARTICLE_COUNT {
            angles.push(rng.next_f32() * std::f32::consts::TAU);
            radii.push(rng.next_f32().powi(5));
            z_offsets.push(rng.next_f32() + rng.next_f32() + rng.next_f32() - 1.5);
            speed_scales.push(0.8 + rng.next_f32() * 1.2);
            symbol_indices.push((rng.next_f32() * glyph_count).floor().min(glyph_count - 1.0));
        }

        let pixel_ratio = window.scale_factor().clamp(1.0, 2.0) as f32;

        State {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            uniform_buf,
            bind_group0,
            bind_group1,
            corner_buf,
            instance_buf,
            instances,
            angles,
            radii,
            z_offsets,
            speed_scales,
            symbol_indices,
            anim_time: 0.0,
            shape_morph_angle: -std::f32::consts::FRAC_PI_2,
            entrance_scale: 0.0,
            state_index: 0,
            waves: Vec::new(),
            mouse_ndc: Vec2::new(-9999.0, -9999.0),
            hovered: false,
            yaw: std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
            distance: 240.0,
            last_cursor: None,
            dragging: false,
            pixel_ratio,
            last_time: Instant::now(),
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.pixel_ratio = self.window.scale_factor().clamp(1.0, 2.0) as f32;
    }

    fn on_cursor_moved(&mut self, x: f32, y: f32) {
        let w = self.config.width as f32;
        let h = self.config.height as f32;
        self.mouse_ndc = Vec2::new((x / w) * 2.0 - 1.0, -((y / h) * 2.0 - 1.0));
        self.hovered = true;

        if self.dragging {
            if let Some(last) = self.last_cursor {
                let dx = x - last.x;
                let dy = y - last.y;
                self.yaw -= dx * 0.005;
                self.pitch = (self.pitch + dy * 0.005).clamp(-1.45, 1.45);
            }
        }
        self.last_cursor = Some(Vec2::new(x, y));
    }

    fn spawn_wave(&mut self) {
        self.state_index = (self.state_index + 1) % CORE_COLORS.len();
        self.waves.push(Wave {
            radius: 0.0,
            state_index: self.state_index,
            width: 80.0,
            speed: 650.0,
        });
    }

    fn update(&mut self, dt: f32) {
        self.anim_time += dt;

        // --- Camera ---
        let aspect = self.config.width as f32 / self.config.height as f32;
        let fov = 75f32.to_radians();
        let dir = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        );
        let eye = dir * self.distance;
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(fov, aspect, 0.1, 2000.0);

        // --- Entrance grow ---
        if self.anim_time > ENTRANCE_DELAY {
            self.entrance_scale += (1.0 - self.entrance_scale) * 0.05 * ENTRANCE_GROW_SPEED;
            if self.entrance_scale > 0.99 {
                self.entrance_scale = 1.0;
            }
        }

        // --- Model matrix (rotation + scale) ---
        let r = self.anim_time * 0.25;
        let rot_x = (r * 2.0).sin().powi(3) * 0.4;
        let rot_y = r.sin().powi(3) * 0.6;
        let rot_z = r.sin().powi(3) * -0.3;
        let rot = Quat::from_euler(EulerRot::XYZ, rot_x, rot_y, rot_z);
        let model = Mat4::from_scale_rotation_translation(
            Vec3::splat(self.entrance_scale.max(0.0001)),
            rot,
            Vec3::ZERO,
        );

        // --- Hover target trong local space ---
        let target_local = if self.hovered {
            let inv_vp = (proj * view).inverse();
            let n = self.mouse_ndc;
            let p0 = inv_vp * Vec4::new(n.x, n.y, 0.0, 1.0);
            let p1 = inv_vp * Vec4::new(n.x, n.y, 1.0, 1.0);
            let p0 = p0.truncate() / p0.w;
            let p1 = p1.truncate() / p1.w;
            let rd = (p1 - p0).normalize_or_zero();
            if rd.z.abs() > 1e-5 {
                let t = -p0.z / rd.z;
                let hit = p0 + rd * t;
                model.inverse().transform_point3(hit)
            } else {
                Vec3::splat(-9999.0)
            }
        } else {
            Vec3::splat(-9999.0)
        };

        // --- Waves ---
        for w in self.waves.iter_mut() {
            w.radius += dt * w.speed;
        }
        self.waves.retain(|w| w.radius < 1200.0);

        // --- Shape morph ---
        self.shape_morph_angle += dt * 0.5;
        let shape_param = (3.5 + SPARK_CURVE_N) / 2.0
            + ((3.5 - SPARK_CURVE_N) / 2.0) * self.shape_morph_angle.sin();
        let inv_shape = 2.0 / shape_param;

        let core: [Vec3; 4] = CORE_COLORS.map(|c| Vec3::new(c[0], c[1], c[2]));
        let accent: [Vec3; 4] = ACCENT_COLORS.map(|c| Vec3::new(c[0], c[1], c[2]));
        let next_state = (self.state_index + 1) % CORE_COLORS.len();

        for i in 0..PARTICLE_COUNT {
            let angle = self.angles[i];
            let rad = self.radii[i];
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let sign_cos = if cos_a > 0.0 {
                1.0
            } else if cos_a < 0.0 {
                -1.0
            } else {
                0.0
            };
            let sign_sin = if sin_a > 0.0 {
                1.0
            } else if sin_a < 0.0 {
                -1.0
            } else {
                0.0
            };

            let o = cos_a.abs().powf(inv_shape) * sign_cos;
            let s = sin_a.abs().powf(inv_shape) * sign_sin;
            let rounding = ROUNDING_FACTOR * (2.0 * angle).cos().powi(2);

            let scale_xy = 120.0 * (1.0 + rad - 0.04);
            let x = (o * (1.0 - rounding) + cos_a * rounding) * scale_xy;
            let y = (s * (1.0 - rounding) + sin_a * rounding) * scale_xy;
            let z = self.z_offsets[i] * 120.0 * 0.8;

            let dist = (x * x + y * y + z * z).sqrt().max(0.001);

            let mut active_state = self.state_index;
            let mut intensity_boost = 1.0f32;

            for w in self.waves.iter() {
                if dist < w.radius {
                    active_state = w.state_index;
                }
                let diff = (dist - w.radius).abs();
                if diff < w.width {
                    let factor = 1.0 - diff / w.width;
                    intensity_boost = intensity_boost.max(1.0 + factor * 2.0);
                }
            }

            let mut hover_intensity = 0.0f32;
            if self.hovered {
                let dx = x - target_local.x;
                let dy = y - target_local.y;
                let dz = z - target_local.z;
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                if d < 75.0 {
                    hover_intensity = 1.0 - d / 75.0;
                    intensity_boost = intensity_boost.max(1.0 + hover_intensity * 1.5);
                }
            }

            let fade = 1.0 - ((dist - 40.0) / 220.0).clamp(0.0, 1.0);
            let size = self.speed_scales[i]
                * (0.8 + fade * 1.8)
                * if intensity_boost > 1.0 { 1.4 } else { 1.0 };

            let pc = core[active_state];
            let sc = accent[active_state];
            let mut col = pc + (sc - pc) * rad;

            if hover_intensity > 0.0 {
                let hp = core[next_state];
                let hs = accent[next_state];
                let hc = hp + (hs - hp) * rad;
                col += (hc - col) * hover_intensity;
            }

            if intensity_boost > 1.0 {
                col = (col * intensity_boost).min(Vec3::ONE);
            }

            self.instances[i] = Instance {
                pos: [x, y, z],
                color: [col.x, col.y, col.z],
                size,
                sym: self.symbol_indices[i],
            };
        }

        // --- Uniform ---
        let h_px = self.config.height as f32;
        let px_scale =
            self.pixel_ratio * 300.0 * (2.0 * (fov * 0.5).tan()) / h_px.max(1.0);

        let uniforms = Uniforms {
            model: model.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            px_scale,
            symbol_count: GLYPHS.len() as f32,
            _pad: [0.0; 2],
        };

        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        self.queue.write_buffer(
            &self.instance_buf,
            0,
            bytemuck::cast_slice(&self.instances),
        );
    }

    fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            Err(_) => return,
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group0, &[]);
            rpass.set_bind_group(1, &self.bind_group1, &[]);
            rpass.set_vertex_buffer(0, self.corner_buf.slice(..));
            rpass.set_vertex_buffer(1, self.instance_buf.slice(..));
            rpass.draw(0..4, 0..PARTICLE_COUNT as u32);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_time).as_secs_f32().min(0.05);
        self.last_time = now;

        self.update(dt);
        self.render();
    }
}

// ----------------------------------------------------------------------------
// App
// ----------------------------------------------------------------------------
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Gemini 3D Symbol Particle Fusion")
                        .with_inner_size(LogicalSize::new(1280.0, 800.0)),
                )
                .expect("không tạo được window"),
        );
        let state = pollster::block_on(State::new(window));
        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let Key::Named(NamedKey::Escape) = event.logical_key {
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::ScaleFactorChanged { .. } => {
                let s = state.window.inner_size();
                state.resize(s.width, s.height);
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.on_cursor_moved(position.x as f32, position.y as f32);
            }
            WindowEvent::CursorLeft { .. } => {
                state.hovered = false;
                state.mouse_ndc = Vec2::new(-9999.0, -9999.0);
            }
            WindowEvent::MouseInput {
                state: bs, button, ..
            } => {
                if button == MouseButton::Left {
                    if bs == ElementState::Pressed {
                        state.spawn_wave();
                        state.dragging = true;
                    } else {
                        state.dragging = false;
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                state.frame();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("không tạo được event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { state: None };
    event_loop.run_app(&mut app).expect("lỗi event loop");
}

// ----------------------------------------------------------------------------
// WGSL
// ----------------------------------------------------------------------------
const SHADER: &str = r#"
struct Uniforms {
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    px_scale: f32,
    symbol_count: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var atlas_tex: texture_2d<f32>;
@group(1) @binding(1) var atlas_smp: sampler;

struct VsIn {
    @location(0) corner: vec2<f32>,
    @location(1) i_pos: vec3<f32>,
    @location(2) i_color: vec3<f32>,
    @location(3) i_size: f32,
    @location(4) i_sym: f32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) sym: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;

    let view_pos = u.view * u.model * vec4<f32>(in.i_pos, 1.0);
    let half = 0.5 * in.i_size * u.px_scale;
    let offset = vec4<f32>(in.corner * half, 0.0, 0.0);

    out.clip = u.proj * (view_pos + offset);
    out.uv = vec2<f32>((in.corner.x + 1.0) * 0.5, (1.0 - in.corner.y) * 0.5);
    out.color = in.i_color;
    out.sym = in.i_sym;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = vec2<f32>((in.uv.x + in.sym) / u.symbol_count, in.uv.y);
    let t = textureSample(atlas_tex, atlas_smp, uv);
    if (t.a < 0.1) {
        discard;
    }
    return vec4<f32>(in.color * t.rgb, t.a);
}
"#;