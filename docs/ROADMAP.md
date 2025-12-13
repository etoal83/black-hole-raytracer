# Kerrブラックホール リアルタイムレイトレーサー 実装ロードマップ

**プロジェクト**: black-hole-raytracer (Rust + wgpu)  
**現状**: Schwarzschild BH（降着円盤なし）、Euler法、平均54 FPS  
**最終目標**: Kerr BH（降着円盤あり）、安定60+ FPS

---

## エグゼクティブサマリー

本ロードマップは、調査した既存文献（2015-2025）の技術的知見を統合し、現状から最終目標までの最適な実装経路を提示する。全6フェーズ、推定10-14週間（フルタイム換算約3-3.5ヶ月）で完了予定。

**重要な実装順序の変更**：
- 当初希望：降着円盤 → RK4 → Kerr → 最適化
- **推奨順序：RK4 → 降着円盤 → Kerr → 最適化**

理由：Euler法の数値不安定性が後続実装で問題化するリスクを回避し、RK4移行でステップ数削減による性能バッファを確保する。

---

## 文献調査から得られた核心的知見

### 1. 数値積分と性能のトレードオフ

| 手法 | 精度 | 安定性 | GPU適性 | 推奨用途 |
|-----|------|--------|---------|----------|
| Euler | 1次 | 低 | 優秀 | **非推奨** |
| RK4 | 4次 | 高 | 優秀 | **リアルタイムレンダリング** |
| Leapfrog | 2次 | 中（symplectic） | 優秀 | エネルギー保存重視時 |
| RKF45 | 4-5次 | 最高 | 良好 | 適応ステップ + 高精度 |

**重要知見**：
- Euler→RK4で**5-10倍のステップサイズ増**が可能
- 適応ステップサイズ `dt ∝ (r - r_s)²` で地平線近傍の安定性確保
- Symplectic積分器（Leapfrog/FANTASY）は長時間軌道には有利だが、リアルタイム用途ではRK4で十分

### 2. Kerrへの拡張の計算コスト

- **演算量増加**: Schwarzschild比で2-3倍（Carter定数、ポテンシャル関数R(r)・Θ(θ)の計算）
- **座標系選択**:
  - Boyer-Lindquist: 文献豊富、実装容易、ただし地平線で特異点
  - Kerr-Schild: 特異点なし、実装複雑
- **実測参考**: geodesic_raytracingは汎用メトリックでも30+ fps達成（RX 6700XT）

### 3. 降着円盤の実装コスト

| 実装レベル | 計算コスト | 視覚効果 |
|-----------|-----------|---------|
| 最小（テクスチャのみ） | ほぼゼロ | 基本形状のみ |
| 中程度（Doppler + 重力赤方偏移） | 数十演算/交差 | 物理的非対称性 |
| 完全（温度 + 黒体放射 + ビーミング） | 数百演算/交差 | 高精度物理レンダリング |

**重要知見**：
- 赤道面交差判定は軽量（`if (old_z * new_z < 0)`）
- Dopplerシフトのみでも視覚的インパクト大
- 温度プロファイル T(r) ∝ r^(-3/4) と黒体放射LUTで高品質化

### 4. 最適化技術の効果

| 手法 | 高速化倍率 | 実装難易度 | 適用条件 |
|------|-----------|-----------|---------|
| マルチ解像度レンダリング | 4-9倍 | 中 | 常時 |
| ニューラルネットワーク加速 | 15-26倍 | 高 | 学習済みモデル必要 |
| プリコンピュテーション | 10-100倍 | 中 | カメラ静止時のみ |
| Early Termination | 1.1-1.2倍 | 低 | 常時 |
| Warp Divergence削減 | 1.05-1.15倍 | 低 | 常時 |

---

## 実装ロードマップ全体像

```
Phase 0: 現状分析とベースライン確立（1週間）
    ↓
Phase 1: Euler → RK4/Leapfrog + 適応ステップ（2-3週間）★最優先
    ↓
Phase 2: 降着円盤追加（3-4週間）
    ├─ Phase 2a: 最小実装（1週間）
    └─ Phase 2b: 物理モデル追加（1-2週間）
    ↓
Phase 3: Schwarzschild → Kerr拡張（3-4週間）
    ├─ Phase 3a: Boyer-Lindquist Kerr測地線（2週間）
    ├─ Phase 3b: フレームドラッギング視覚効果（1週間）
    └─ Phase 3c: Kerr-Schild座標への移行（オプション、1-2週間）
    ↓
Phase 4: 最適化で60+ FPS達成（2-3週間）
    ├─ Phase 4-1: マルチ解像度レンダリング（1週間）★最重要
    ├─ Phase 4-2: Early Termination最適化（3日）
    ├─ Phase 4-3: Warp Divergence削減（5日）
    └─ Phase 4-4: プリコンピュテーション（オプション、1週間）
    ↓
Phase 5: ニューラル加速の実験（オプション、2-4週間）
    ↓
Phase 6: 研究レポート作成（2-3週間）
```

**総工数**: 10-14週間（フルタイム換算）

---

## Phase 0: 現状分析とベースライン確立

**期間**: 1週間  
**目的**: ボトルネック特定と性能測定基盤構築

### 実装項目

1. **プロファイリング実装**
   - wgpu timestamp queriesの導入
   - フレームあたりのGPU計算時間測定
   - ステージ別（レイ生成、積分、環境マップサンプリング）の時間計測

2. **性能特性分析**
   - ステップ数とFPSの関係測定（128, 256, 512, 1024ステップ）
   - 解像度とFPSの関係測定（720p, 1080p, 1440p, 4K）
   - GPU使用率、メモリ帯域幅の監視

3. **数値精度評価**
   - 保存量（エネルギー E = -p_t）の追跡
   - 地平線到達までの軌道長での誤差蓄積測定
   - 既知解（円軌道、光子球）との比較

### 参考文献

- なし（基盤作業）

### 成功基準

- [ ] 光線あたりの平均ステップ数と計算時間の相関取得
- [ ] ボトルネック（GPU計算 vs メモリ帯域 vs CPU↔GPU転送）の特定
- [ ] Euler法の数値誤差の定量評価完了

### 期待される発見

- 現状のステップ数上限（おそらく256-512程度）
- GPU計算律速 vs メモリ帯域律速の判別
- 最も時間を消費しているシェーダステージの特定

---

## Phase 1: Euler → RK4/Leapfrog + 適応ステップ

**期間**: 2-3週間  
**目的**: 数値積分の安定性と精度を確保し、性能バッファ獲得  
**優先度**: ★★★ 最優先（降着円盤より先）

### なぜ最初に実装すべきか

1. **数値安定性の確保**: 降着円盤やKerrの実装中にEuler法の発散が顕在化すると原因切り分けが困難
2. **性能バッファ**: RK4でステップ数を1/5に削減可能 → 降着円盤追加の余裕
3. **実装の独立性**: 他フェーズへの影響が小さく、後戻りリスクなし

### 実装項目

#### 1. RK4積分器実装（1週間）

**WGSL Compute Shader内での実装**:

```wgsl
// 状態ベクトル: (r, θ, φ, p_r, p_θ, p_φ)
fn geodesic_derivative(state: vec6<f32>) -> vec6<f32> {
    let r = state.x;
    let theta = state.y;
    // ... メトリック係数計算
    // ... 測地線方程式の右辺
    return vec6(dr_dlambda, dtheta_dlambda, dphi_dlambda, 
                dpr_dlambda, dptheta_dlambda, dpphi_dlambda);
}

fn rk4_step(state: vec6<f32>, h: f32) -> vec6<f32> {
    let k1 = geodesic_derivative(state);
    let k2 = geodesic_derivative(state + 0.5 * h * k1);
    let k3 = geodesic_derivative(state + 0.5 * h * k2);
    let k4 = geodesic_derivative(state + h * k3);
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}
```

#### 2. Leapfrog積分器実装（比較用、3日）

```wgsl
// Schwarzschild特化のBinet方程式版
fn leapfrog_step(u: f32, u_dot: f32, step: f32) -> vec2<f32> {
    let u_ddot = 1.5 * u * u - u;  // u'' = (3/2)u³ - u
    let new_u_dot = u_dot + u_ddot * step;
    let new_u = u + new_u_dot * step;
    return vec2(new_u, new_u_dot);
}
```

#### 3. 適応ステップサイズ（1週間）

**距離依存ステップサイズ**:

```wgsl
fn compute_step_size(r: f32, r_schwarzschild: f32, base_dt: f32) -> f32 {
    let dist_from_horizon = r - r_schwarzschild;
    let k = 0.1;  // 調整パラメータ
    return base_dt * clamp(dist_from_horizon * k, 0.01, 1.0);
}
```

**ステップサイズLUT（分岐回避版）**:

```wgsl
// プリコンピュートテーブル: r → step_size
@group(0) @binding(5) var step_size_lut: texture_1d<f32>;

fn get_adaptive_step(r: f32) -> f32 {
    let normalized_r = (r - r_min) / (r_max - r_min);
    return textureSampleLevel(step_size_lut, sampler, normalized_r, 0.0).r;
}
```

#### 4. 保存量モニタリング（3日）

```wgsl
// エネルギー保存のチェック
let E_initial = -p_t;  // 初期化時
// ... 積分ループ
let E_current = -p_t;
let energy_error = abs(E_current - E_initial) / E_initial;
if (energy_error > 0.01) {
    // デバッグ出力または警告色
}
```

### 参考文献

#### 主要参考

- **oseiskar実装** (`integrator.js`, `schwarzschild.glsl`)
  - Leapfrog実装の直接的GLSLコード
  - 適応ステップサイズのヒューリスティック
  - 保存量チェックのパターン

- **Lepsima/Black-Hole-Simulation**
  - HLSL Compute Shaderの実装パターン
  - ピクセルシェーダからCompute Shaderへの移行理由
  - 300 fps達成の最適化技術

#### 補助参考

- **geodesic_raytracing**
  - 2次Verlet + 適応ステップサイズ
  - 双対数による自動微分（Christoffel記号計算の自動化）

- **FANTASY**
  - Symplectic積分器の理論背景
  - エネルギー保存の重要性

### 技術的注意点

1. **RK4のメモリ使用量**: k1, k2, k3, k4を保持するため、レジスタ圧迫の可能性
   - 対策: k1-k4を逐次計算し、一時変数を再利用

2. **Leapfrogの座標系制限**: 赤道面（θ = π/2）に限定
   - Kerr拡張時にRK4が必須になる

3. **適応ステップのワープダイバージェンス**: 隣接スレッドで異なるステップ数
   - 対策: 固定イテレーション with masking（Phase 4で詳細実装）

### 成功基準

- [ ] 同精度でステップ数**50%以上削減**（例: Euler 512ステップ → RK4 256ステップ）
- [ ] または、同ステップ数で数値誤差**1/10以下**
- [ ] **60+ FPS達成**（降着円盤追加の余裕確保）
- [ ] 保存量の相対誤差 < 1%（長時間積分でも）

### 期待される性能改善

- FPS: 54 → **65-75** （ステップ数削減効果）
- 数値誤差: 10^-2 → **10^-4** （RK4の高精度）

---

## Phase 2: 降着円盤追加

**期間**: 3-4週間  
**目的**: 視覚的インパクト最大化と物理モデル構築  
**優先度**: ★★ 高（RK4実装後）

### Phase 2a: 最小実装（1週間）

#### 目的
単純テクスチャマッピング、物理効果なしで基本形状を可視化

#### 実装項目

1. **赤道面交差判定**

```wgsl
fn check_disk_intersection(
    old_pos: vec3<f32>, 
    new_pos: vec3<f32>,
    r_isco: f32,
    r_outer: f32
) -> option<DiskIntersection> {
    // θ = π/2 平面（Cartesianではz=0）との交差
    if (old_pos.z * new_pos.z < 0.0) {
        let t = -old_pos.z / (new_pos.z - old_pos.z);
        let intersection_pos = old_pos + t * (new_pos - old_pos);
        let r = length(intersection_pos);
        
        if (r >= r_isco && r <= r_outer) {
            return some(DiskIntersection {
                position: intersection_pos,
                radius: r,
                t: t
            });
        }
    }
    return none;
}
```

2. **プロシージャルカラー**

```wgsl
fn disk_color_simple(r: f32, r_isco: f32, r_outer: f32) -> vec3<f32> {
    let normalized = (r - r_isco) / (r_outer - r_isco);
    // 内側: オレンジ、外側: 赤茶色
    return mix(vec3(1.0, 0.5, 0.0), vec3(0.5, 0.2, 0.0), normalized);
}
```

3. **ISCO定義**

```wgsl
const R_SCHWARZSCHILD: f32 = 1.0;  // 幾何単位
const R_ISCO: f32 = 6.0 * R_SCHWARZSCHILD;  // Schwarzschildの場合
const R_DISK_OUTER: f32 = 20.0 * R_SCHWARZSCHILD;
```

#### 参考文献

- **oseiskar実装** (`accretion.glsl`)
  - 赤道面交差判定の実装パターン
  - テクスチャ座標の計算（r, φ → UV）

- **Sean Holloway Unity実装**
  - Fractional Brownian Motionノイズによるボリューメトリック効果（オプション）

#### 成功基準

- [ ] 降着円盤が視覚的に認識可能（円環状）
- [ ] 重力レンズ効果による円盤の歪みが確認可能
- [ ] FPS低下 **< 5%**（交差判定は軽量なはず）

### Phase 2b: 物理モデル追加（1-2週間）

#### 目的
Dopplerシフト、重力赤方偏移、温度プロファイルによる物理的リアリズム

#### 実装項目

1. **Keplerian速度場（3日）**

```wgsl
fn keplerian_velocity(r: f32, M: f32) -> f32 {
    return sqrt(M / (r * r * r));  // 幾何単位
}

fn disk_velocity_vector(pos: vec3<f32>) -> vec3<f32> {
    let r = length(pos);
    let v_magnitude = keplerian_velocity(r, M);
    // 反時計回り（右手系でz軸周り）
    return v_magnitude * normalize(vec3(-pos.y, pos.x, 0.0));
}
```

2. **g-factor計算（5日）**

```wgsl
fn compute_redshift_factor(
    disk_pos: vec3<f32>,
    ray_direction: vec3<f32>,
    observer_r: f32
) -> f32 {
    let r_disk = length(disk_pos);
    
    // Doppler成分
    let v_disk = disk_velocity_vector(disk_pos);
    let beta = length(v_disk);
    let cos_theta = dot(normalize(v_disk), ray_direction);
    let doppler = (1.0 - beta * cos_theta) / sqrt(1.0 - beta * beta);
    
    // 重力赤方偏移成分
    let grav_emit = sqrt(1.0 - R_SCHWARZSCHILD / r_disk);
    let grav_obs = sqrt(1.0 - R_SCHWARZSCHILD / observer_r);
    let grav_shift = grav_emit / grav_obs;
    
    return doppler * grav_shift;
}
```

3. **温度プロファイル（3日）**

```wgsl
const T_ISCO: f32 = 10000.0;  // Kelvin（可視化用に低温化）

fn disk_temperature(r: f32) -> f32 {
    return T_ISCO * pow(r / R_ISCO, -0.75);
}
```

4. **黒体放射LUT生成（Pythonで事前計算、3日）**

```python
import numpy as np
from colour import planck_law, XYZ_to_sRGB

def generate_blackbody_lut(T_min=1000, T_max=30000, steps=512):
    """温度→RGB変換テーブル生成"""
    temperatures = np.linspace(T_min, T_max, steps)
    wavelengths = np.linspace(380e-9, 780e-9, 100)  # 可視光範囲
    
    colors = []
    for T in temperatures:
        # Planck分布
        spectrum = planck_law(wavelengths, T)
        # XYZ色空間へ変換（CIE色感度関数との畳み込み）
        XYZ = spectrum_to_XYZ(spectrum, wavelengths)
        # sRGBへ変換、正規化
        RGB = XYZ_to_sRGB(XYZ)
        RGB_normalized = RGB / np.max(RGB)
        colors.append(RGB_normalized)
    
    return np.array(colors)

# 1Dテクスチャとして保存
lut = generate_blackbody_lut()
save_as_texture("blackbody_lut.png", lut)
```

**WGSL側でのサンプリング**:

```wgsl
@group(0) @binding(6) var blackbody_lut: texture_1d<f32>;
@group(0) @binding(7) var lut_sampler: sampler;

fn temperature_to_color(T: f32) -> vec3<f32> {
    let T_normalized = (T - T_MIN) / (T_MAX - T_MIN);
    return textureSampleLevel(blackbody_lut, lut_sampler, T_normalized, 0.0).rgb;
}
```

5. **相対論的ビーミング（3日）**

```wgsl
fn disk_emission(r: f32, g_factor: f32) -> vec3<f32> {
    let T_emit = disk_temperature(r);
    let T_observed = T_emit * g_factor;
    let base_color = temperature_to_color(T_observed);
    
    // 相対論的ビーミング: I_obs = g^3 × I_emit (ボロメトリック)
    // または g^4 (単色)
    let beaming = pow(g_factor, 3.0);
    
    return base_color * beaming;
}
```

#### 参考文献

##### 主要参考

- **DNGR Interstellar論文** (Classical and Quantum Gravity 32, 2015)
  - Section II.C: g-factorの完全な定義（式28-30）
  - Section III: 降着円盤モデル
  - Appendix: FIDO観測者とカメラ座標系

- **conjLob/BlackHoleShader**
  - `generateLookupTexture.ipynb`: 温度→RGB変換の実装
  - 赤方偏移を考慮したLUT生成のPythonコード
  - Unity HLSLシェーダでのサンプリング例

##### 補助参考

- **oseiskar実装**
  - Doppler色のルックアップテーブル実装
  - 簡易的なg-factor計算

- **Starless (rantonels)**
  - 温度プロファイル T ∝ r^(-3/4) の理論背景
  - 黒体放射の色変換アルゴリズム

#### 技術的注意点

1. **g-factorの符号**: approachingとrecedingで異なる
   - cos_theta > 0: approaching（青方偏移）
   - cos_theta < 0: receding（赤方偏移）

2. **温度スケール**: 現実的な温度（10^7 K）では全てX線領域
   - 可視化のため10^4 K程度に人為的に下げる（DNGR論文も同様）

3. **ビーミングの飽和**: g^4は極端な値を取りうる
   - HDR処理またはtone mappingが必要

#### 成功基準

- [ ] approaching側（回転方向）が青白く、receding側が赤暗く見える
- [ ] 温度勾配が視覚的に確認可能（内側が明るい）
- [ ] FPS **50以上維持**（目標60には届かなくてもOK、Phase 3で回復）
- [ ] DNGR論文Figure 9との定性的一致

### 期待される視覚効果

- **最小実装後**: 基本的な円環形状と重力レンズ効果
- **物理モデル後**: approaching/receding非対称性、内縁の明るさ、リアルな色彩

---

## Phase 3: Schwarzschild → Kerr拡張

**期間**: 3-4週間  
**目的**: 回転ブラックホールの測地線計算とフレームドラッギング効果  
**優先度**: ★★ 高（最も計算コスト増）

### Phase 3a: Boyer-Lindquist座標でのKerr測地線（2週間）

#### 目的
スピンパラメータ a/M を導入し、Kerr時空での光線追跡を実現

#### 実装項目

1. **メトリック係数の更新（3日）**

Boyer-Lindquist座標でのKerr計量:

```wgsl
struct KerrMetric {
    g_tt: f32,
    g_tphi: f32,
    g_rr: f32,
    g_thth: f32,
    g_phiphi: f32,
    Sigma: f32,
    Delta: f32,
}

fn kerr_metric(r: f32, theta: f32, a: f32, M: f32) -> KerrMetric {
    let cos_th = cos(theta);
    let sin_th = sin(theta);
    
    let Sigma = r * r + a * a * cos_th * cos_th;
    let Delta = r * r - 2.0 * M * r + a * a;
    
    let A = (r * r + a * a) * (r * r + a * a) - Delta * a * a * sin_th * sin_th;
    
    var metric: KerrMetric;
    metric.g_tt = -(1.0 - 2.0 * M * r / Sigma);
    metric.g_tphi = -2.0 * M * a * r * sin_th * sin_th / Sigma;
    metric.g_rr = Sigma / Delta;
    metric.g_thth = Sigma;
    metric.g_phiphi = A * sin_th * sin_th / Sigma;
    metric.Sigma = Sigma;
    metric.Delta = Delta;
    
    return metric;
}
```

2. **保存量の計算（3日）**

```wgsl
struct ConservedQuantities {
    E: f32,      // エネルギー
    L: f32,      // 角運動量
    Q: f32,      // Carter定数
}

fn compute_conserved_quantities(
    r: f32, theta: f32, 
    p_r: f32, p_theta: f32, p_phi: f32,
    metric: KerrMetric,
    a: f32
) -> ConservedQuantities {
    let p_t = -(metric.g_tt * E + metric.g_tphi * p_phi);  // ここでEは未知
    
    // 逆算でEを求める（nullness条件 g^μν p_μ p_ν = 0 から）
    let E = -p_t;  // 実際にはより複雑な初期化が必要
    let L = p_phi;
    
    let Q = p_theta * p_theta + 
            cos(theta) * cos(theta) * (
                a * a * E * E + L * L / (sin(theta) * sin(theta))
            );
    
    return ConservedQuantities(E, L, Q);
}
```

3. **ポテンシャル関数R(r), Θ(θ)（5日）**

```wgsl
fn potential_R(r: f32, E: f32, L: f32, Q: f32, a: f32, M: f32) -> f32 {
    let r2 = r * r;
    let a2 = a * a;
    let Delta = r2 - 2.0 * M * r + a2;
    
    let P = (r2 + a2) - a * L / E;  // 簡略化された表現
    let R_pot = P * P - Delta * ((L - a * E) * (L - a * E) + Q);
    
    return R_pot;
}

fn potential_Theta(theta: f32, E: f32, L: f32, Q: f32, a: f32) -> f32 {
    let cos_th = cos(theta);
    let sin_th = sin(theta);
    
    let Theta_pot = Q + a * a * E * E * cos_th * cos_th - 
                    L * L * cos_th * cos_th / (sin_th * sin_th);
    
    return Theta_pot;
}
```

4. **一階形式測地線方程式（5日）**

```wgsl
fn kerr_geodesic_rhs(
    state: vec4<f32>,  // (r, theta, p_r, p_theta)
    conserved: ConservedQuantities,
    a: f32, M: f32
) -> vec4<f32> {
    let r = state.x;
    let theta = state.y;
    let p_r = state.z;
    let p_theta = state.w;
    
    let metric = kerr_metric(r, theta, a, M);
    
    // dr/dλ = ±√R / Σ
    let R_val = potential_R(r, conserved.E, conserved.L, conserved.Q, a, M);
    let dr_dlambda = select(-1.0, 1.0, p_r > 0.0) * sqrt(max(R_val, 0.0)) / metric.Sigma;
    
    // dθ/dλ = ±√Θ / Σ
    let Theta_val = potential_Theta(theta, conserved.E, conserved.L, conserved.Q, a);
    let dtheta_dlambda = select(-1.0, 1.0, p_theta > 0.0) * sqrt(max(Theta_val, 0.0)) / metric.Sigma;
    
    // dp_r/dλ, dp_θ/dλ は複雑な導関数（要実装）
    let dpr_dlambda = 0.0;  // TODO: 実装
    let dptheta_dlambda = 0.0;  // TODO: 実装
    
    return vec4(dr_dlambda, dtheta_dlambda, dpr_dlambda, dptheta_dlambda);
}
```

#### 参考文献

##### 主要参考

- **DNGR Interstellar論文**
  - Section II.B: 完全な測地線方程式（式9-15）
  - Appendix A: 保存量の導出
  - Appendix B: カメラから測地線への写像

- **Verbraeck-Eisemann論文**
  - Section 3.1: Boyer-Lindquist実装の詳細
  - 数値積分の安定性に関する議論

##### 補助参考

- **GRay2論文** (Chan et al., 2018)
  - Kerr-Schild座標の代替案（Phase 3cで参照）

- **KerrGeoPy** (arXiv:2406.01413)
  - 解析解の理論的背景
  - 楕円積分による厳密解（将来的な最適化に有用）

#### 数値的注意点

1. **√R, √Θの符号選択**
   - 初期条件（カメラの視線方向）から決定
   - 符号が反転するturning pointの処理が必要

2. **Δ → 0での特異性**
   - r < r_+ + ε で強制終了（r_+ = M + √(M² - a²) は外部地平線）
   - εは10^-3程度

3. **Carter定数の検証**
   - 積分中にQが保存されているか定期チェック
   - 相対誤差 > 1%なら警告

#### 成功基準

- [ ] a = 0 でSchwarzschildと完全一致
- [ ] a ≠ 0 でフレームドラッギング効果が視覚的に確認可能
- [ ] 光球面の半径がスピン依存（a増加で減少）
- [ ] FPS **30-40程度**（2-3倍の計算増加は想定内）
- [ ] Carter定数の保存（相対誤差 < 1%）

### Phase 3b: フレームドラッギング視覚効果（1週間）

#### 目的
ISCO半径の変化、エルゴスフィア可視化、非対称性の強調

#### 実装項目

1. **スピン依存ISCO（3日）**

```wgsl
fn kerr_isco_radius(a: f32, M: f32, prograde: bool) -> f32 {
    let a_normalized = a / M;
    
    // Z1, Z2関数（Bardeen et al. 1972）
    let Z1 = 1.0 + pow(1.0 - a_normalized * a_normalized, 1.0/3.0) * 
             (pow(1.0 + a_normalized, 1.0/3.0) + pow(1.0 - a_normalized, 1.0/3.0));
    let Z2 = sqrt(3.0 * a_normalized * a_normalized + Z1 * Z1);
    
    if (prograde) {
        return M * (3.0 + Z2 - sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)));
    } else {
        return M * (3.0 + Z2 + sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)));
    }
}
```

2. **エルゴスフィア境界（2日）**

```wgsl
fn ergosphere_radius(theta: f32, a: f32, M: f32) -> f32 {
    return M + sqrt(M * M - a * a * cos(theta) * cos(theta));
}

// エルゴスフィアの可視化（オプション）
fn draw_ergosphere(r: f32, theta: f32, a: f32, M: f32) -> vec3<f32> {
    let r_ergo = ergosphere_radius(theta, a, M);
    if (abs(r - r_ergo) < 0.1) {
        return vec3(0.5, 0.7, 1.0);  // 青色のワイヤーフレーム
    }
    return vec3(0.0);
}
```

3. **FIDO観測者基底（2日）**

```wgsl
// Locally Non-Rotating Frame（慣性系に近い）
fn fido_basis(r: f32, theta: f32, a: f32, M: f32) -> mat3x3<f32> {
    let metric = kerr_metric(r, theta, a, M);
    let omega = -metric.g_tphi / metric.g_phiphi;  // フレームドラッギング角速度
    
    // 正規直交基底の構成
    // ... 複雑な計算（DNGR Appendix参照）
    
    return mat3x3(e_r, e_theta, e_phi);
}
```

#### 参考文献

- **SpaceEngine blog** (2022/07/05, 2022/08/30)
  - ISCO計算の実装コード断片
  - エルゴスフィア描画の視覚化技術
  - ボリューメトリック降着円盤のアニメーション

- **DNGR論文 Appendix**
  - FIDO基底の詳細な導出
  - カメラ座標系との変換

#### 成功基準

- [ ] スピンパラメータUIで a/M = 0.0 → 0.99 の変化を観察可能
- [ ] 順行ISCOが a 増加で縮小（6M → 1M）
- [ ] 降着円盤内縁がISCO半径に追従
- [ ] エルゴスフィアが視覚的に確認可能（オプション）

### Phase 3c: Kerr-Schild座標への移行（オプション、1-2週間）

#### 実装判断基準

**実装すべき場合**:
- Boyer-Lindquistで地平線近傍の数値不安定性が深刻
- 高スピン（a/M > 0.95）で頻繁にクラッシュ

**実装不要な場合**:
- r < r_+ + 0.01 での早期終了で問題なし
- 視覚的に許容範囲の精度

#### 実装項目（実施する場合）

1. **Cartesian Kerr-Schild座標でのメトリック**

```wgsl
fn kerr_schild_metric(x: f32, y: f32, z: f32, a: f32, M: f32) -> mat4x4<f32> {
    let r2 = x * x + y * y + z * z - a * a;
    let r = sqrt(0.5 * (r2 + sqrt(r2 * r2 + 4.0 * a * a * z * z)));
    
    let l_x = (r * x + a * y) / (r * r + a * a);
    let l_y = (r * y - a * x) / (r * r + a * a);
    let l_z = z / r;
    
    let f = 2.0 * M * r * r * r / (r * r * r * r + a * a * z * z);
    
    // g_μν = η_μν + f l_μ l_ν
    var g: mat4x4<f32> = mat4x4(
        vec4(-1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(0.0, 0.0, 0.0, 1.0)
    );
    
    // ... f l_μ l_ν を加算
    
    return g;
}
```

#### 参考文献

- **GRay2論文** (Chan et al., 2018, ApJ 867)
  - 完全な方程式とベンチマーク
  - Boyer-Lindquistとの性能比較

#### 成功基準（実施した場合）

- [ ] 地平線通過時の数値安定性向上
- [ ] 高スピン（a/M > 0.95）での安定動作
- [ ] Boyer-Lindquistと視覚的に同一の結果

---

## Phase 4: 最適化で60+ FPS達成

**期間**: 2-3週間  
**目的**: Kerr + 降着円盤で安定60 FPS  
**優先度**: ★★★ 最重要（目標達成の最終段階）

### Phase 4-1: マルチ解像度レンダリング（1週間）

#### 目的
SpaceEngine手法による4-9倍高速化

#### 実装項目

1. **2パスアーキテクチャ（3日）**

**Pass 1: 低解像度測地線計算**

```wgsl
// 1/3解像度（例: 1080p → 360p）で測地線を計算
@compute @workgroup_size(8, 8)
fn geodesic_lowres(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let resolution_divisor = 3u;
    let pixel = id.xy;
    
    // 測地線積分
    let final_direction = trace_ray_geodesic(pixel);
    
    // 偏向ベクトルをテクスチャに書き込み
    textureStore(deflection_texture, pixel, vec4(final_direction, 1.0));
}
```

**Pass 2: フル解像度ワープ**

```wgsl
@fragment
fn warp_fullres(
    @location(0) uv: vec2<f32>
) -> @location(0) vec4<f32> {
    // 低解像度偏向テクスチャからバイキュービック補間
    let deflection = textureSampleLevel(
        deflection_texture, 
        bicubic_sampler, 
        uv, 
        0.0
    ).xyz;
    
    // 環境マップまたは降着円盤をサンプル
    return textureSample(environment_map, sampler, deflection);
}
```

2. **シャドウエッジ検出と再計算（2日）**

```wgsl
fn detect_shadow_edge(uv: vec2<f32>, deflection_tex: texture_2d<f32>) -> bool {
    // 隣接ピクセルとの偏向差を計算
    let center = textureSample(deflection_tex, sampler, uv).xyz;
    let neighbors = array<vec3<f32>, 4>(
        textureSample(deflection_tex, sampler, uv + vec2(pixel_size, 0.0)).xyz,
        textureSample(deflection_tex, sampler, uv - vec2(pixel_size, 0.0)).xyz,
        textureSample(deflection_tex, sampler, uv + vec2(0.0, pixel_size)).xyz,
        textureSample(deflection_tex, sampler, uv - vec2(0.0, pixel_size)).xyz
    );
    
    for (var i = 0; i < 4; i++) {
        if (length(neighbors[i] - center) > EDGE_THRESHOLD) {
            return true;  // エッジ検出
        }
    }
    return false;
}

// エッジピクセルのみフル解像度で再計算
if (detect_shadow_edge(uv, deflection_texture)) {
    return trace_ray_geodesic_fullres(pixel);
}
```

3. **補間品質の選択（2日）**

```wgsl
// バイリニア（高速）
fn bilinear_sample(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    return textureSample(tex, linear_sampler, uv);
}

// バイキュービック（高品質）
fn bicubic_sample(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    // Catmull-Rom補間
    // ... 16サンプルの重み付き平均
}

// Lanczos（最高品質、重い）
fn lanczos_sample(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    // ... Lanczos-3カーネル
}
```

#### 参考文献

- **SpaceEngine blog** (2022/07/05)
  - アルゴリズムの詳細説明
  - 性能測定結果（GTX 1060で100+ fps）
  - シャドウエッジ処理の重要性

- **Verbraeck-Eisemann論文**
  - 適応グリッドの理論的基盤
  - 補間誤差の評価

#### 期待効果

- **解像度1/2**: 4倍高速化
- **解像度1/3**: 9倍高速化
- **シャドウエッジ再計算**: 品質低下を5%未満に抑制

#### 成功基準

- [ ] 1080pで**60+ FPS**達成（Kerr + 降着円盤）
- [ ] 視覚的品質低下が知覚困難（SSIM > 0.95）
- [ ] シャドウエッジのジャギーなし

### Phase 4-2: Early Termination最適化（3日）

#### 目的
無駄な計算の削減による10-20%高速化

#### 実装項目

```wgsl
struct RayState {
    ACTIVE: u32,
    ESCAPED: u32,
    ABSORBED: u32,
    HIT_DISK: u32,
}

fn trace_ray_with_early_termination(initial_state: GeodesicState) -> vec3<f32> {
    var state = initial_state;
    var ray_state = RayState.ACTIVE;
    var accumulated_color = vec3(0.0);
    
    for (var step = 0u; step < MAX_STEPS; step++) {
        if (ray_state != RayState.ACTIVE) {
            break;  // 早期終了
        }
        
        // ホライズン通過判定
        if (state.r < r_horizon + 0.01) {
            ray_state = RayState.ABSORBED;
            accumulated_color = vec3(0.0);  // 黒色
            break;
        }
        
        // 脱出判定
        if (state.r > R_ESCAPE && state.dr_dlambda > 0.0) {
            ray_state = RayState.ESCAPED;
            accumulated_color = sample_environment(state.direction);
            break;
        }
        
        // 降着円盤との交差
        if (check_disk_intersection(state)) {
            ray_state = RayState.HIT_DISK;
            accumulated_color = compute_disk_emission(state);
            break;
        }
        
        // 積分ステップ
        state = rk4_step(state, adaptive_step_size(state.r));
    }
    
    // デバッグ：最大ステップ到達時は警告色
    if (step >= MAX_STEPS - 1) {
        return vec3(1.0, 0.0, 1.0);  // マゼンタ
    }
    
    return accumulated_color;
}
```

#### 参考文献

- **Lepsima/Black-Hole-Simulation**
  - 3状態分類（escaped, absorbed, hit disk）
  - C#側でのステート管理パターン

#### 期待効果

- 平均ステップ数: 256 → **200**程度（約20%削減）
- 特に光球面外の光線で効果大

#### 成功基準

- [ ] FPS向上**10-20%**
- [ ] デバッグカラー（マゼンタ）の出現頻度 < 1%

### Phase 4-3: Warp Divergence削減（5日）

#### 目的
GPU並列効率向上による5-15%高速化

#### 実装項目

1. **固定イテレーション with Masking（3日）**

```wgsl
fn trace_ray_fixed_iteration(initial_state: GeodesicState) -> vec3<f32> {
    var state = initial_state;
    var active = true;
    var accumulated_color = vec3(0.0);
    
    // 全スレッドが同じ回数ループ（分岐なし）
    for (var step = 0u; step < MAX_STEPS; step++) {
        // マスクで無効化されたスレッドも計算は継続
        if (active) {
            // 終了条件チェック（ただしbreakしない）
            if (state.r < r_horizon || state.r > R_ESCAPE) {
                active = false;
            }
            
            // activeがfalseでも計算は継続（結果は破棄）
            state = rk4_step(state, adaptive_step_size(state.r));
        }
    }
    
    return accumulated_color;
}
```

2. **ステップサイズLUTで分岐回避（2日）**

```wgsl
// CPU側で事前計算
fn generate_step_size_lut() -> [f32; 256] {
    var lut: [f32; 256];
    for (var i = 0; i < 256; i++) {
        let r = r_min + (r_max - r_min) * f32(i) / 255.0;
        let dist_from_horizon = r - r_schwarzschild;
        lut[i] = base_dt * clamp(dist_from_horizon * 0.1, 0.01, 1.0);
    }
    return lut;
}

// WGSL側でルックアップ（分岐なし）
@group(0) @binding(8) var<storage, read> step_size_lut: array<f32, 256>;

fn get_adaptive_step_branchless(r: f32) -> f32 {
    let index = u32(clamp((r - r_min) / (r_max - r_min) * 255.0, 0.0, 255.0));
    return step_size_lut[index];
}
```

#### 参考文献

- **Kerrpy論文** (arXiv:2111.03824)
  - CUDA instruction-level parallelism
  - Warp効率最大化技術
  - Coalesced memory access

- **GRay2論文**
  - GPU最適化のベストプラクティス

#### 期待効果

- Warp効率: 60% → **80%以上**
- FPS向上: **5-15%**

#### 成功基準

- [ ] プロファイラでwarp divergenceの減少確認
- [ ] FPS向上測定

### Phase 4-4: プリコンピュテーション（オプション、1週間）

#### 実装判断基準

**実装すべき場合**:
- Phase 4-1～4-3でも60 FPS未達成
- カメラ静止時のユースケースが多い（デモ、スクリーンショット）

**実装不要な場合**:
- 既に60+ FPS達成
- インタラクティブなカメラ移動が主用途

#### 実装項目（実施する場合）

1. **固定視点での測地線グリッド事前計算（CPU、3日）**

```rust
// Rayon並列化
use rayon::prelude::*;

fn precompute_geodesic_grid(
    observer_pos: Vec3,
    grid_resolution: usize,
    fov: f32
) -> DeflectionGrid {
    let pixels: Vec<_> = (0..grid_resolution)
        .flat_map(|y| (0..grid_resolution).map(move |x| (x, y)))
        .collect();
    
    let deflections: Vec<_> = pixels.par_iter()
        .map(|(x, y)| {
            let ray_dir = pixel_to_ray_direction(*x, *y, grid_resolution, fov);
            let final_dir = trace_geodesic(observer_pos, ray_dir);
            final_dir
        })
        .collect();
    
    DeflectionGrid {
        resolution: grid_resolution,
        deflections,
    }
}
```

2. **GPUテクスチャバッファにキャッシュ（2日）**

```rust
// wgpu側
let deflection_texture = device.create_texture(&wgpu::TextureDescriptor {
    size: wgpu::Extent3d {
        width: grid_resolution as u32,
        height: grid_resolution as u32,
        depth_or_array_layers: 1,
    },
    format: wgpu::TextureFormat::Rgba32Float,
    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    ..Default::default()
});

// データ転送
queue.write_texture(
    deflection_texture.as_image_copy(),
    bytemuck::cast_slice(&grid.deflections),
    wgpu::ImageDataLayout { .. },
    texture_size,
);
```

3. **カメラ移動時の複数グリッド補間（2日）**

```wgsl
// 最近傍4グリッド間の補間
fn interpolate_grids(
    camera_pos: vec3<f32>,
    uv: vec2<f32>,
    grid_cache: array<texture_2d<f32>, 8>
) -> vec3<f32> {
    // カメラ位置から最近傍グリッドを選択
    let grid_indices = find_nearest_grids(camera_pos, 4);
    let weights = compute_interpolation_weights(camera_pos, grid_indices);
    
    var deflection = vec3(0.0);
    for (var i = 0; i < 4; i++) {
        deflection += weights[i] * textureSample(
            grid_cache[grid_indices[i]], 
            sampler, 
            uv
        ).xyz;
    }
    
    return deflection;
}
```

#### 参考文献

- **Verbraeck-Eisemann論文**
  - 適応グリッドプリコンピュテーション
  - グリッド補間の精度評価

- **Eric Bruneton's black_hole_shader**
  - O(1)ルックアップの実装
  - 2Dテクスチャベースのアプローチ

#### 期待効果（カメラ静止時）

- FPS: 40 → **200+**（10-100倍高速化）
- カメラ移動時: 補間オーバーヘッドで30-40 fps程度

#### 成功基準（実施した場合）

- [ ] カメラ静止時に100+ FPS
- [ ] グリッド補間による視覚的アーティファクトなし
- [ ] グリッドキャッシュのVRAM使用量 < 500 MB

---

## Phase 5: ニューラル加速の実験（オプション）

**期間**: 2-4週間  
**目的**: 最先端手法の評価  
**優先度**: 低（Phase 4で60 FPS未達成時のみ）

### 実装判断基準

**実装すべき場合**:
- Phase 4の全最適化でも60 FPS未達成
- 学術的興味（研究レポートへの組み込み）
- デプロイ複雑さを許容できる

**実装不要な場合**:
- 既に60+ FPS達成
- 推論ランタイムの依存関係を避けたい

### 実装項目（実施する場合）

1. **GravLensXアプローチの理解（1週間）**

- 論文精読とアーキテクチャ理解
- 学習データ要件の見積もり
- 推論性能の予測

2. **簡易MLPによる測地線近似（1週間）**

```python
import torch
import torch.nn as nn

class GeodesicMLP(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, hidden_dim),  # 入力: (r, θ, φ, p_r, p_θ)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 出力: 最終方向ベクトル
        )
    
    def forward(self, initial_conditions):
        return self.network(initial_conditions)

# 学習
model = GeodesicMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(100):
    for batch in dataloader:
        initial, target = batch
        pred = model(initial)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

3. **学習データ生成（1週間）**

```rust
// 正確な測地線計算結果でデータセット生成
fn generate_training_data(num_samples: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..num_samples).into_par_iter()
        .map(|_| {
            let initial = random_initial_condition();
            let final_direction = trace_geodesic_accurate(initial);
            (initial.to_vec(), final_direction.to_vec())
        })
        .collect()
}
```

4. **ONNX推論デプロイ（4日）**

```rust
use onnxruntime::{environment::Environment, GraphOptimizationLevel, LoggingLevel};

let environment = Environment::builder()
    .with_name("geodesic_predictor")
    .with_log_level(LoggingLevel::Warning)
    .build()?;

let session = environment
    .new_session_builder()?
    .with_optimization_level(GraphOptimizationLevel::All)?
    .with_model_from_file("geodesic_mlp.onnx")?;

// 推論
let input = ndarray::arr1(&[r, theta, phi, p_r, p_theta]).into_dyn();
let outputs = session.run(vec![input])?;
let final_direction: Vec<f32> = outputs[0].try_extract()?.view().to_slice()?.to_vec();
```

### 参考文献

- **GravLensX論文** (arXiv:2507.15775)
  - ニューラルネットワークアーキテクチャ
  - 15-26倍高速化の詳細

- **BH-NeRF**
  - Neural Radiance Fieldsの適用
  - 別アプローチの参考

### 期待効果

- **推論速度**: 従来の測地線積分比15-26倍高速
- **精度**: 視覚的に区別不可能（SSIM > 0.99）

### 成功基準（実施した場合）

- [ ] 学習済みモデルの推論時間 < 測地線積分の1/10
- [ ] 視覚的精度が許容範囲（ユーザースタディで区別困難）
- [ ] ONNX推論のwgpuパイプラインへの統合成功

---

## Phase 6: 研究レポート作成

**期間**: 2-3週間  
**目的**: 技術的知見の体系化と共有

### レポート構成案

#### 1. Abstract（1ページ）

- プロジェクトの目的と成果の要約
- 達成したFPS（最終値）
- 主要な技術的貢献

#### 2. Introduction（2ページ）

- ブラックホール可視化の背景
- 既存手法の課題
- 本実装の目標と新規性

#### 3. Methodology（5-8ページ）

##### 3.1 数値積分手法の比較

- Euler vs RK4 vs Leapfrogの性能評価
- 適応ステップサイズの効果
- 数値誤差の定量分析

##### 3.2 Schwarzschild → Kerr拡張

- 実装の詳細（Boyer-Lindquist vs Kerr-Schild）
- 計算コスト増加の分析
- 視覚効果の比較（スクリーンショット）

##### 3.3 降着円盤の物理モデル

- 最小実装 vs 完全物理モデルのトレードオフ
- Dopplerシフトと重力赤方偏移の視覚的影響
- 温度プロファイルの実装

##### 3.4 最適化技術

- マルチ解像度レンダリングの効果（定量評価）
- Early Terminationの寄与
- Warp Divergence削減の測定
- プリコンピュテーション（実施した場合）
- ニューラル加速（実施した場合）

#### 4. Results（3-5ページ）

##### 4.1 性能評価

| フェーズ | 構成 | FPS (1080p) | FPS (4K) |
|---------|------|------------|----------|
| Phase 0 | Schwarzschild + Euler | 54 | - |
| Phase 1 | Schwarzschild + RK4 | 65-75 | - |
| Phase 2 | + 降着円盤（物理） | 50-55 | - |
| Phase 3 | Kerr + 降着円盤 | 30-40 | - |
| Phase 4-1 | + マルチ解像度 | 60-80 | 30-40 |
| Phase 4-2,3 | + 全最適化 | 65-90 | 35-50 |

##### 4.2 視覚的品質評価

- Schwarzschild vs Kerrの比較画像
- スピンパラメータ依存性（a/M = 0, 0.5, 0.9, 0.99）
- 降着円盤の物理効果（approaching/receding非対称性）
- 既存実装（Starless, DNGR）との定性比較

##### 4.3 最適化手法の寄与度

積み上げ棒グラフで各最適化のFPS寄与を可視化

#### 5. Discussion（2-3ページ）

- wgpu/Rustエコシステムの利点と課題
- 既存文献との比較（何が新しいか）
- 今後の拡張可能性（Kerr-Newman, ワームホール等）

#### 6. Conclusion（1ページ）

- 達成した成果の要約
- コミュニティへの貢献
- オープンソース化の宣言

#### 7. References

調査した全文献のリスト

### 付録

- Appendix A: WGSL シェーダコード（主要部分）
- Appendix B: Kerr測地線方程式の導出
- Appendix C: ベンチマーク環境の詳細

### 成果物

1. **技術レポート PDF** (15-25ページ)
   - LaTeX執筆、arXivフォーマット
   - または Markdown → Pandoc → PDF

2. **GitHubリポジトリの整備**
   - README.md with スクリーンショット
   - 使用方法ドキュメント
   - ビルド手順
   - ライセンス（MIT推奨）

3. **デモ動画**
   - YouTube等で公開
   - 各最適化の前後比較
   - インタラクティブデモ

4. **技術ブログ記事**（オプション）
   - Medium/Zenn/Qiita等
   - より平易な解説
   - 実装の苦労話

### 参考フォーマット

- **oseiskar/rantonelsのブログ形式**
  - 技術的深さ + 視覚的魅力
  - コード断片と数式のバランス

- **学術論文形式**
  - ApJ Supplements, MNRAS等への投稿も視野
  - 査読プロセスを経ることで信頼性向上

### 成功基準

- [ ] 技術レポート完成（15ページ以上）
- [ ] GitHubリポジトリのスター数 > 100（公開後3ヶ月）
- [ ] 技術ブログのビュー数 > 1,000
- [ ] コミュニティからのフィードバック取得

---

## 参考文献マトリクス

| Phase | 主要参考文献 | 補助参考文献 | コード参照先 |
|-------|------------|------------|-----------|
| **Phase 0** | - | - | - |
| **Phase 1** | oseiskar, Lepsima | geodesic_raytracing, FANTASY | oseiskar `integrator.js`, Lepsima HLSL |
| **Phase 2a** | oseiskar | Sean Holloway blog | oseiskar `accretion.glsl` |
| **Phase 2b** | DNGR Interstellar, conjLob | oseiskar, Starless | conjLob `generateLookupTexture.ipynb` |
| **Phase 3a** | DNGR Interstellar, Verbraeck-Eisemann | GRay2, KerrGeoPy | - |
| **Phase 3b** | SpaceEngine blog | DNGR Appendix | - |
| **Phase 3c** | GRay2 | - | - |
| **Phase 4-1** | SpaceEngine blog | Verbraeck-Eisemann | - |
| **Phase 4-2** | Lepsima | - | Lepsima C# |
| **Phase 4-3** | Kerrpy, GRay2 | - | - |
| **Phase 4-4** | Verbraeck-Eisemann, Bruneton | - | Bruneton GLSL |
| **Phase 5** | GravLensX | BH-NeRF | - |
| **Phase 6** | oseiskar/rantonels blog形式 | 学術論文例 | - |

---

## リスク評価と緊急時の代替案

### 高リスクポイントと対策

#### リスク1: Kerr実装でFPS 30台突入

**発生確率**: 中（50%）  
**影響度**: 高（目標未達成）

**対策**:
1. **即座実施**: Phase 4-1（マルチ解像度）を前倒し
2. **部分的妥協**: スピンパラメータを低スピン（a/M ≤ 0.5）に制限
3. **最終手段**: Schwarzschild + 降着円盤で目標達成とし、Kerrは「今後の拡張」扱い

#### リスク2: 降着円盤の物理モデルが重すぎる

**発生確率**: 低（20%）  
**影響度**: 中（視覚品質低下）

**対策**:
1. **簡略化**: g⁴ビーミングを省略、Dopplerシフトのみに
2. **LOD導入**: 遠方の降着円盤は簡易モデル、近傍のみ完全物理
3. **LUT高速化**: 黒体放射LUTを低解像度化（512 → 128）

#### リスク3: wgpu/WSLの制約

**発生確률**: 低（15%）  
**影響度**: 中（実装変更必要）

**例**:
- Texture atomics非対応
- Compute Shaderのworkgroup size制限
- 倍精度浮動小数点なし

**対策**:
1. **代替実装**: バッファ直接書き込みに変更
2. **Workaround**: CPU側で一部処理
3. **最終手段**: Vulkan/SPIR-Vへの移行検討

#### リスク4: Phase 4の全最適化でも60 FPS未達成

**発生確率**: 低（10%）  
**影響度**: 高（目標未達成）

**対策**:
1. **Phase 5実施**: ニューラル加速の導入
2. **ハードウェア前提変更**: RTX 3060以上を推奨環境に
3. **目標再定義**: 30 fps を「安定動作」とし、60 fpsは「高性能環境」扱い

### 工数見積もりの信頼区間

| Phase | 楽観的 | 現実的 | 悲観的 |
|-------|--------|--------|--------|
| Phase 0 | 3日 | 1週間 | 2週間 |
| Phase 1 | 1.5週間 | 2-3週間 | 4週間 |
| Phase 2 | 2週間 | 3-4週間 | 5週間 |
| Phase 3 | 2週間 | 3-4週間 | 6週間 |
| Phase 4 | 1週間 | 2-3週間 | 4週間 |
| Phase 5 | - | 2-4週間 | 6週間 |
| Phase 6 | 1週間 | 2-3週間 | 4週間 |
| **合計** | **8週間** | **13-21週間** | **31週間** |

**フルタイム換算**: 2ヶ月（楽観的） → **3-5ヶ月（現実的）** → 8ヶ月（悲観的）

---

## 実装順序の最終推奨

### 推奨順序（技術的最適解）

```
Phase 0 (1週) → Phase 1 (2-3週) → Phase 2 (3-4週) → 
Phase 3 (3-4週) → Phase 4 (2-3週) → Phase 6 (2-3週)
```

**Phase 5（ニューラル加速）は Phase 4 後の性能次第で判断**

### あなたの当初希望順序との比較

| 順位 | あなたの希望 | 推奨順序 | 変更理由 |
|-----|------------|---------|---------|
| 1 | 降着円盤追加 | **Phase 1: RK4移行** | 数値安定性優先、性能バッファ確保 |
| 2 | RK4で60FPS回復 | **Phase 2: 降着円盤** | RK4後なら降着円盤追加でもFPS維持 |
| 3 | Kerr拡張 | **Phase 3: Kerr拡張** | 同じ |
| 4 | 最適化で60FPS | **Phase 4: 最適化** | 同じ |
| 5 | レポート作成 | **Phase 6: レポート** | 同じ |

**変更点は Phase 1 と 2 の入れ替えのみ**

### 判断根拠

#### Phase 1（RK4）を最初にすべき理由

1. **数値安定性の確保**
   - 降着円盤実装中にEuler法の発散問題が出ると原因切り分けが困難
   - RK4は「基盤技術」であり、他の全フェーズがこれに依存

2. **性能バッファの獲得**
   - Euler 512ステップ → RK4 256ステップで同精度
   - これにより降着円盤追加（~10%負荷増）を吸収可能

3. **実装の独立性**
   - RK4移行は他フェーズに影響しない単独タスク
   - 後戻りリスクが最小

4. **学習曲線**
   - RK4実装で測地線積分の理解が深まる
   - Kerr実装時に応用が効く

#### あなたの順序も合理的な理由

- **視覚的フィードバック重視**: 降着円盤の視覚効果は大きく、モチベーション維持に有効
- **段階的複雑化**: Euler → 降着円盤 → RK4 という順序でも論理的

**結論**: 両方の順序に利点があるため、**最終判断はあなた次第**。ただし技術的リスクを最小化するなら推奨順序、視覚的フィードバックを優先するなら当初希望順序。

---

## まとめ：成功への道筋

### 最終目標の確認

- **Kerr ブラックホール**（スピンパラメータ a/M = 0.9程度）
- **物理的降着円盤**（Doppler + 重力赤方偏移 + 温度プロファイル）
- **安定 60+ FPS** @ 1080p（RTX 3060クラスのGPU想定）

### 段階的達成目標

| マイルストーン | 達成時期 | 期待FPS |
|--------------|---------|---------|
| Phase 1完了: RK4実装 | 2-3週間後 | 65-75 |
| Phase 2完了: 降着円盤 | 6-8週間後 | 50-55 |
| Phase 3完了: Kerr | 10-14週間後 | 30-40 |
| Phase 4-1完了: マルチ解像度 | 11-15週間後 | **60-80** ✓ |
| Phase 4完了: 全最適化 | 13-18週間後 | **65-90** ✓ |

### 重要な意思決定ポイント

1. **Phase 0終了時**: Phase 1とPhase 2の実施順序を最終決定
2. **Phase 3a終了時**: Kerr-Schild座標への移行要否を判断
3. **Phase 4-3終了時**: Phase 5（ニューラル加速）実施要否を判断
4. **Phase 4完了時**: Phase 6（レポート）の詳細スコープ決定

### 技術的成功の鍵

1. **数値安定性**: RK4 + 適応ステップサイズ
2. **段階的検証**: 各フェーズで既知解（円軌道等）との比較
3. **性能プロファイリング**: 各最適化の定量評価
4. **文献活用**: 既存実装のコードを積極参照

### このロードマップの使い方

1. **各Phaseの開始時**: 「実装項目」「参考文献」「成功基準」を確認
2. **実装中**: 技術的注意点を参照し、既知の落とし穴を回避
3. **Phase完了時**: 成功基準をチェックし、未達なら対策実施
4. **全体進捗管理**: GitHubのMilestoneやProjectsで可視化

このロードマップに従えば、**3-5ヶ月で目標達成が現実的**です。健闘を祈ります。