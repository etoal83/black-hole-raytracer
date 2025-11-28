// Schwarzschild ブラックホールのレイトレーシングコンピュートシェーダ

// カメラとシーン設定
struct Camera {
    position: vec3<f32>,
    _padding1: f32,
    forward: vec3<f32>,
    _padding2: f32,
    right: vec3<f32>,
    _padding3: f32,
    up: vec3<f32>,
    _padding4: f32,
}

struct SceneParams {
    black_hole_position: vec3<f32>,
    schwarzschild_radius: f32,  // rs = 2GM/c^2
    screen_width: u32,
    screen_height: u32,
    fov: f32,
    max_steps: u32,
}

@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> scene: SceneParams;

// Schwarzschild 計量テンソルの00成分
fn g_tt(r: f32, rs: f32) -> f32 {
    return -(1.0 - rs / r);
}

// Schwarzschild 計量テンソルのrr成分
fn g_rr(r: f32, rs: f32) -> f32 {
    return 1.0 / (1.0 - rs / r);
}

// 測地線方程式に基づく光線の位置と速度の更新
// Schwarzschild時空での光線追跡
struct GeodesicResult {
    position: vec3<f32>,
    velocity: vec3<f32>,
    is_active: f32,  // 1.0=継続, 0.0=イベントホライズン到達
}

fn trace_geodesic(
    pos: vec3<f32>,
    vel: vec3<f32>,
    rs: f32,
    dt: f32
) -> GeodesicResult {
    let r = length(pos);

    var result: GeodesicResult;

    // イベントホライズンに近づいたら停止
    if r < rs * 1.05 {
        result.position = pos;
        result.velocity = vel;
        result.is_active = 0.0;
        return result;
    }

    // 重力による速度の変化（簡略化された測地線方程式）
    let r_vec = pos / r;
    let v_radial = dot(vel, r_vec);

    // Schwarzschild時空での重力加速度
    let factor = rs / (2.0 * r * r * (1.0 - rs / r));
    let accel = -factor * (
        vel * (1.0 - rs / r) -
        r_vec * v_radial * (1.0 + rs / r)
    );

    let new_vel = vel + accel * dt;

    // 位置の更新（新しい速度を使用）
    let new_pos = pos + new_vel * dt;

    result.position = new_pos;
    result.velocity = new_vel;
    result.is_active = 1.0;

    return result;
}

// スカイボックス/背景色の計算
fn get_background_color(direction: vec3<f32>) -> vec3<f32> {
    let dir = normalize(direction);

    // グラデーションの空
    let t = dir.y * 0.5 + 0.5;
    let sky_color = mix(
        vec3<f32>(0.05, 0.05, 0.15),  // 暗い青
        vec3<f32>(0.01, 0.01, 0.03),  // ほぼ黒
        t
    );

    // より安定した星のパターン
    // 球面座標を使用
    let theta = atan2(dir.z, dir.x);
    let phi = asin(dir.y);

    // グリッド状の星パターン
    let grid_scale = 20.0;
    let star_x = fract(theta * grid_scale);
    let star_y = fract(phi * grid_scale);

    // 星の位置をハッシュ化
    let star_hash = fract(sin(dot(vec2<f32>(floor(theta * grid_scale), floor(phi * grid_scale)),
                                   vec2<f32>(12.9898, 78.233))) * 43758.5453);

    // 星を配置（確率的に）
    if star_hash > 0.95 {
        // 星の中心からの距離
        let star_center = vec2<f32>(0.5, 0.5);
        let dist_to_center = length(vec2<f32>(star_x, star_y) - star_center);

        if dist_to_center < 0.1 {
            let brightness = 1.0 - (dist_to_center / 0.1);
            return sky_color + vec3<f32>(1.0, 1.0, 1.0) * brightness * brightness;
        }
    }

    return sky_color;
}

// レイトレーシングのメイン関数
fn trace_ray(origin: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    var pos = origin;
    var vel = normalize(direction);

    let dt = 0.1;  // タイムステップ
    let rs = scene.schwarzschild_radius;
    let bh_pos = scene.black_hole_position;

    for (var i = 0u; i < scene.max_steps; i++) {
        let relative_pos = pos - bh_pos;
        let dist = length(relative_pos);

        // 光線がブラックホールから十分離れたら背景を返す
        if dist > 100.0 {
            return get_background_color(vel);
        }

        let result = trace_geodesic(relative_pos, vel, rs, dt);

        // イベントホライズンに到達した場合
        if result.is_active < 0.5 {
            return vec3<f32>(0.0, 0.0, 0.0);  // 黒
        }

        pos = result.position + bh_pos;
        vel = normalize(result.velocity);
    }

    // 最大ステップ数に達した場合も背景を返す
    return get_background_color(vel);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_coords = vec2<u32>(global_id.xy);

    // 画面外のピクセルは無視
    if pixel_coords.x >= scene.screen_width || pixel_coords.y >= scene.screen_height {
        return;
    }

    // ピクセル座標を正規化 [-1, 1]
    let uv = vec2<f32>(
        (f32(pixel_coords.x) / f32(scene.screen_width) - 0.5) * 2.0,
        (f32(pixel_coords.y) / f32(scene.screen_height) - 0.5) * -2.0  // Y軸を反転
    );

    // アスペクト比の補正
    let aspect_ratio = f32(scene.screen_width) / f32(scene.screen_height);
    let uv_corrected = vec2<f32>(uv.x * aspect_ratio, uv.y);

    // レイの方向を計算
    let fov_factor = tan(scene.fov * 0.5);
    let ray_direction = normalize(
        camera.forward +
        camera.right * uv_corrected.x * fov_factor +
        camera.up * uv_corrected.y * fov_factor
    );

    // レイトレーシング
    let color = trace_ray(camera.position, ray_direction);

    // 出力テクスチャに書き込み
    textureStore(output_texture, pixel_coords, vec4<f32>(color, 1.0));
}
