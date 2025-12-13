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
    _padding1: f32,
    schwarzschild_radius: f32,  // rs = 2GM/c^2
    screen_width: u32,
    screen_height: u32,
    fov: f32,
    max_steps: u32,
    debug_mode: u32,  // 0=通常モード, 1=ステップ数可視化モード
    _padding2: vec4<u32>,
}

@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> scene: SceneParams;
@group(0) @binding(3) var skybox_texture: texture_2d<f32>;
@group(0) @binding(4) var skybox_sampler: sampler;

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

// 方向ベクトルからEquirectangular UV座標を計算
fn direction_to_equirectangular_uv(dir: vec3<f32>) -> vec2<f32> {
    let normalized = normalize(dir);
    let u = 0.5 + atan2(normalized.z, normalized.x) / (2.0 * 3.14159265359);
    let v = 0.5 - asin(normalized.y) / 3.14159265359;
    return vec2<f32>(u, v);
}

// スカイボックス/背景色の計算（テクスチャから取得）
fn get_background_color(direction: vec3<f32>) -> vec3<f32> {
    let uv = direction_to_equirectangular_uv(direction);
    // コンピュートシェーダーではtextureSampleLevelを使用
    return textureSampleLevel(skybox_texture, skybox_sampler, uv, 0.0).rgb;
}

// レイトレーシング結果
struct TraceResult {
    color: vec3<f32>,
    steps: u32,
}

// ステップ数を色にマッピング（ヒートマップ）
fn steps_to_color(steps: u32, max_steps: u32) -> vec3<f32> {
    let t = f32(steps) / f32(max_steps);

    // 青 -> シアン -> 緑 -> 黄 -> 赤のヒートマップ
    if t < 0.25 {
        // 青 -> シアン
        let local_t = t * 4.0;
        return mix(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0), local_t);
    } else if t < 0.5 {
        // シアン -> 緑
        let local_t = (t - 0.25) * 4.0;
        return mix(vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), local_t);
    } else if t < 0.75 {
        // 緑 -> 黄
        let local_t = (t - 0.5) * 4.0;
        return mix(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), local_t);
    } else {
        // 黄 -> 赤
        let local_t = (t - 0.75) * 4.0;
        return mix(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), local_t);
    }
}

// レイトレーシングのメイン関数
fn trace_ray(origin: vec3<f32>, direction: vec3<f32>) -> TraceResult {
    var pos = origin;
    var vel = normalize(direction);

    let dt = 0.1;  // タイムステップ
    let rs = scene.schwarzschild_radius;
    let bh_pos = scene.black_hole_position;

    var steps_taken = 0u;

    for (var i = 0u; i < scene.max_steps; i++) {
        steps_taken = i + 1u;
        let relative_pos = pos - bh_pos;
        let dist = length(relative_pos);

        // 光線がブラックホールから十分離れたら背景を返す
        if dist > 100.0 {
            return TraceResult(get_background_color(vel), steps_taken);
        }

        let result = trace_geodesic(relative_pos, vel, rs, dt);

        // イベントホライズンに到達した場合
        if result.is_active < 0.5 {
            return TraceResult(vec3<f32>(0.0, 0.0, 0.0), steps_taken);  // 黒
        }

        pos = result.position + bh_pos;
        vel = normalize(result.velocity);
    }

    // 最大ステップ数に達した場合も背景を返す
    return TraceResult(get_background_color(vel), steps_taken);
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
    let result = trace_ray(camera.position, ray_direction);

    // debug_modeに応じて出力を切り替え
    var output_color: vec3<f32>;
    if scene.debug_mode == 1u {
        // ステップ数可視化モード
        output_color = steps_to_color(result.steps, scene.max_steps);
    } else {
        // 通常モード
        output_color = result.color;
    }

    // 出力テクスチャに書き込み
    textureStore(output_texture, pixel_coords, vec4<f32>(output_color, 1.0));
}
