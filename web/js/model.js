// In-browser inference for the exported abalone model (see scripts/export_web_model.py).
//
// Mirrors the Python pipeline step by step:
//   QuantileClipper -> MorphometricFeatures -> one-hot sex -> XGBoost trees on log1p(rings)
// Feature arithmetic is done in float64 exactly as pandas does it; values are then rounded to
// float32 (what XGBoost does when it builds its DMatrix) before being compared with the
// float32 split thresholds. Leaf values are accumulated in float32 like XGBoost's predictor.

const f32 = Math.fround;
const WEIGHTS = ["whole_weight", "shucked_weight", "viscera_weight", "shell_weight"];

export function createPredictor(spec) {
  if (spec.format !== "abalone-xgb-trees/1") {
    throw new Error(`Unsupported model format: ${spec.format}`);
  }
  const eps = spec.eps;
  const clipIndex = new Map(spec.clip.columns.map((c, i) => [c, i]));
  const featureIndex = new Map(spec.feature_names.map((n, i) => [n, i]));
  const nFeatures = spec.feature_names.length;
  const base = f32(spec.base_score);

  const trees = spec.trees.map((t) => ({
    f: Int32Array.from(t.f),
    t: Float32Array.from(t.t),
    l: Int32Array.from(t.l),
    r: Int32Array.from(t.r),
    d: Uint8Array.from(t.d),
  }));

  function clip(name, v) {
    const i = clipIndex.get(name);
    if (i === undefined) return v;
    const lo = spec.clip.lower[i];
    const hi = spec.clip.upper[i];
    return v < lo ? lo : v > hi ? hi : v;
  }

  /** Engineered feature values (float64), keyed by feature name. */
  function features(row) {
    const x = {};
    for (const c of spec.clip.columns) x[c] = clip(c, Number(row[c]));
    const sex = String(row.sex).trim().toUpperCase();
    for (const s of spec.sex_categories) x[`sex_${s}`] = sex === s ? 1 : 0;

    const whole = Math.max(x.whole_weight, eps);
    x.shucked_ratio = x.shucked_weight / whole;
    x.viscera_ratio = x.viscera_weight / whole;
    x.shell_ratio = x.shell_weight / whole;
    x.meat_to_shell = x.shucked_weight / Math.max(x.shell_weight, eps);
    const volume = x.length * x.diameter * Math.max(x.height, eps);
    x.volume = volume;
    x.density = x.whole_weight / Math.max(volume, eps);
    x.aspect_ratio = x.diameter / Math.max(x.length, eps);
    for (const w of WEIGHTS) x[`log_${w}`] = Math.log1p(Math.max(x[w], 0));
    return x;
  }

  function vector(row) {
    const named = features(row);
    const v = new Float32Array(nFeatures);
    for (const [name, i] of featureIndex) {
      const value = named[name];
      if (value === undefined) throw new Error(`Feature ${name} not computed`);
      v[i] = value; // Float32Array assignment rounds to float32
    }
    return { named, vector: v };
  }

  /** Raw model output: log1p(rings) margin. */
  function margin(v) {
    let s = base;
    for (const tree of trees) {
      let n = 0;
      while (tree.f[n] !== -1) {
        const x = v[tree.f[n]];
        if (Number.isNaN(x)) n = tree.d[n] ? tree.l[n] : tree.r[n];
        else n = x < tree.t[n] ? tree.l[n] : tree.r[n];
      }
      s = f32(s + tree.t[n]);
    }
    return s;
  }

  function predict(row) {
    const { named, vector: v } = vector(row);
    const m = margin(v);
    const rings = Math.max(Math.expm1(m), spec.min_rings);
    return { rings, age: rings + spec.age_offset, margin: m, features: named };
  }

  return { predict, features, nTrees: trees.length, featureNames: spec.feature_names };
}
