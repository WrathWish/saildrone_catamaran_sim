// Tarayıcıda çalışan fizik: app.py'deki body_forces / hareket denklemleri / RK4 ve kaskad PID otopilotun
// satır satır karşılığı. Kumandalı (canlı) sürüş için; Python sürümü DEĞİŞMEDİ ve referanstır.
// İşlem sırası Python/NumPy ile aynı tutuldu (np.sum'ın ikili toplaması dahil); eşitlik
// tests/test_physics.py içinde Python ile karşılaştırılarak sınanır.
const SIM = (() => {
  const KT2MS = 0.514444, RHO_AIR = 1.225, RHO_WATER = 1025.0, NU_WATER = 1.19e-6;
  const WIND_FROM_DEG = 0.0, PSI_WIND_TOWARD_DEG = 180.0;
  const D2R = Math.PI / 180.0, R2D = 180.0 / Math.PI;
  const REV_A = [0, 5, 10, 15, 20, 25, 90];                         // ters akış ΔCD (Sheldahl & Klimas 1981, Tablo 2)
  const REV_DCD = [0.0171, 0.0437, 0.1216, 0.0400, 0.0230, 0.0, 0.0];
  const SOFT_A = [8, 15, 22, 30, 45, 60, 90];
  const SOFT_CL = [0.00, 1.15, 1.40, 1.30, 1.00, 0.65, 0.0];
  const SOFT_CD = [0.10, 0.15, 0.25, 0.40, 0.75, 1.00, 1.25];

  const pymod = (a, b) => { let m = a % b; if (m !== 0 && (m < 0) !== (b < 0)) m += b; return m; };   // Python %
  const wrap180 = a => pymod(a + 180.0, 360.0) - 180.0;
  const cu = deg => { const r = deg * D2R; return [Math.sin(r), Math.cos(r)]; };          // compass_unit
  const sign = x => (x > 0 ? 1 : x < 0 ? -1 : 0);                                          // np.sign
  const copysign = (a, b) => (b < 0 || Object.is(b, -0) ? -Math.abs(a) : Math.abs(a));
  function interp(x, xp, fp) {                                                             // np.interp
    const n = xp.length;
    if (x <= xp[0]) return x < xp[0] ? fp[0] : fp[0];
    if (x >= xp[n - 1]) return fp[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xp[mid] <= x) lo = mid; else hi = mid; }
    const s = (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo]);
    return s * (x - xp[lo]) + fp[lo];
  }
  function npsum(a) {                                                                      // NumPy ikili toplama (n ≤ 128)
    const n = a.length;
    if (n < 8) { let r = 0; for (let i = 0; i < n; i++) r += a[i]; return r; }
    const r = a.slice(0, 8);
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[i + j];
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[i];
    return res;
  }
  const smoothstep = t => { t = Math.min(1.0, Math.max(0.0, t)); return t * t * (3 - 2 * t); };
  const friction = (V, p) => { const Re = Math.max(V * p.L / NU_WATER, 1e4); return 0.075 / (Math.log10(Re) - 2.0) ** 2; };
  const cardioid = h => 90.0 * (Math.cos((PSI_WIND_TOWARD_DEG - h) * D2R) + 1.0) / 2.0;

  function sailCoeffs(a_abs, sail, p) {
    let a_e, sgn;
    if (a_abs <= 90.0) { a_e = a_abs; sgn = 1.0; } else { a_e = 180.0 - a_abs; sgn = -1.0; }
    const ar = a_e * D2R;
    if (sail === "wing") {
      const CD0 = 0.012, CLa = 2 * Math.PI / (1 + 2 / p.AR_wing);
      const CL_att = CLa * ar, CD_att = CD0 + CL_att ** 2 / (Math.PI * 0.9 * p.AR_wing);
      const CL_sep = p.CN_plate * Math.sin(ar) * Math.cos(ar), CD_sep = CD0 + p.CN_plate * Math.sin(ar) ** 2;
      const t = smoothstep((a_e - p.alpha_stall) / 6.0);
      const CL = (1 - t) * CL_att + t * CL_sep;
      let CD = (1 - t) * CD_att + t * CD_sep;
      if (sgn > 0) return [CL, CD, false, 0.25 + 0.25 * t];
      CD += interp(a_e, REV_A, REV_DCD);                                // ters akış: firar kenarı önde
      return [-CL, CD, false, 0.75 - 0.25 * t];
    }
    if (a_e < p.alpha_luff) return [0.0, 0.10, true, 0.25];
    return [sgn * interp(a_e, SOFT_A, SOFT_CL), interp(a_e, SOFT_A, SOFT_CD), false, 0.25];
  }
  function sailForces(aw, delta_s, sail, p) {
    const aaw = Math.hypot(aw[0], aw[1]);
    const out = { Fx: 0.0, Fy: 0.0, alpha: 0.0, flapping: sail === "soft", backwinded: false, aaw, cp: 0.25 };
    if (aaw < 1e-9) return out;
    const w = [aw[0] / aaw, aw[1] / aaw];
    const ca = (180.0 - delta_s) * D2R, c = [Math.cos(ca), Math.sin(ca)];
    const cross = w[0] * c[1] - w[1] * c[0];
    const a_abs = Math.atan2(Math.abs(cross), w[0] * c[0] + w[1] * c[1]) * R2D;
    const sin_beta = -w[1];
    let backwinded = false;
    if (sail === "soft" && Math.abs(delta_s) > 1.0 && Math.abs(sin_beta) > Math.sin(10 * D2R))
      backwinded = sign(delta_s) === sign(sin_beta);
    const [CL, CD, flapping, cp] = sailCoeffs(a_abs, sail, p);
    const area = sail === "wing" ? p.A_wing : p.A_soft;
    const q = 0.5 * RHO_AIR * area * aaw ** 2;
    const rot = [-w[1], w[0]];
    const L = Math.abs(cross) > 1e-12 ? [-sign(cross) * rot[0], -sign(cross) * rot[1]] : rot;
    Object.assign(out, { Fx: q * (CL * L[0] + CD * w[0]), Fy: q * (CL * L[1] + CD * w[1]), alpha: a_abs, flapping, backwinded, cp });
    return out;
  }
  function sailCE(delta_s, sail, p, cp) {
    const a = (180.0 - delta_s) * D2R, c = [Math.cos(a), Math.sin(a)];
    const k = sail === "wing" ? (cp - 0.25) * p.chord_wing : p.ce_frac * p.boom_len;
    return [p.x_mast + k * c[0], 0.0 + k * c[1]];
  }
  function rudderCoeffs(a_abs, p) {
    let a_e, sgn;
    if (a_abs <= 90.0) { a_e = a_abs; sgn = 1.0; } else { a_e = 180.0 - a_abs; sgn = -1.0; }
    const ar = a_e * D2R;
    const CL_att = p.CLa_rud * ar, CD_att = 0.008 + CL_att ** 2 / (Math.PI * 0.9 * p.AR_rud);
    const CL_sep = p.CN_plate * Math.sin(ar) * Math.cos(ar), CD_sep = 0.008 + p.CN_plate * Math.sin(ar) ** 2;
    const t = smoothstep((a_e - p.rud_stall) / 6.0);
    return [sgn * ((1 - t) * CL_att + t * CL_sep), (1 - t) * CD_att + t * CD_sep];
  }

  // env: { p, tws_kt, sail: "wing"|"soft", trim: "opt"|"card"|"man", trim_manual, tbl: {betas, best} }
  function bodyForces(u, v, r, heading, d_r, env) {
    const p = env.p, tws = env.tws_kt * KT2MS, twu = cu(PSI_WIND_TOWARD_DEG), tw = [tws * twu[0], tws * twu[1]];
    const f = cu(heading), s = cu(heading + 90.0);
    const aw = [(tw[0] * f[0] + tw[1] * f[1]) - u, (tw[0] * s[0] + tw[1] * s[1]) - v];
    const beta = Math.atan2(-aw[1], -aw[0]) * R2D;
    const side = -sign(Math.sin(beta * D2R)) || -1.0;
    let delta_s;
    if (env.trim === "opt") delta_s = side * interp(Math.abs(beta), env.tbl.betas, env.tbl.best);
    else if (env.trim === "card") delta_s = side * cardioid(heading);
    else delta_s = +env.trim_manual;
    let streaming = false;
    if (env.sail === "soft" && !p.boom_rigid) {
      const dmax = Math.min(Math.abs(delta_s), 90.0), d_stream = wrap180(-beta);
      if (Math.abs(d_stream) <= dmax) { delta_s = d_stream; streaming = true; } else delta_s = side * dmax;
    }
    const sf = sailForces(aw, delta_s, env.sail, p);
    sf.streaming = streaming;
    const ce = sailCE(delta_s, env.sail, p, sf.cp);
    const N_s = ce[0] * sf.Fy - ce[1] * sf.Fx;
    const kw = 0.5 * RHO_AIR * p.CDA_windage * Math.hypot(aw[0], aw[1]);
    const Fa = [sf.Fx + kw * aw[0], sf.Fy + kw * aw[1]];

    const V = Math.hypot(u, v), qw = 0.5 * RHO_WATER * V * V;
    const n = 20, dx = p.L / n, s1 = -0.5 * RHO_WATER * (p.A_hull_lat / p.L), au = Math.abs(u);
    const dY = new Array(n), xdY = new Array(n);
    for (let k = 0; k < n; k++) {
      const x = (k + 0.5) / n * p.L - p.L / 2 + p.x_CLR, vl = v + r * x;
      dY[k] = s1 * (p.CLa_hull * au * vl + p.Cd_cross * vl * Math.abs(vl)) * dx;
      xdY[k] = x * dY[k];
    }
    const Y_h = npsum(dY), N_h = npsum(xdY);
    let X_h = 0.0, Yd = 0.0, Di = 0.0, R = 0.0;
    if (V > 1e-9) {
      const lam = Math.atan2(v, u), CL_h = p.CLa_hull * Math.sin(lam) * Math.abs(Math.cos(lam));
      Di = qw * p.A_hull_lat * CL_h ** 2 / (Math.PI * p.AR_hull);
      R = qw * ((1 + p.form_k) * friction(V, p) + p.C_R) * p.S_wet;
      X_h = -(R + Di) * u / V; Yd = -(R + Di) * v / V;
    }
    const x_r = p.x_rud, w = [-u, -(v + r * x_r)], wn = Math.hypot(w[0], w[1]);
    const c = [-Math.cos(d_r), Math.sin(d_r)];
    let Fr = [0, 0], CLr = 0.0;
    const a_flow = wn > 1e-9 ? Math.atan2(v + r * x_r, u) * R2D : 0.0;
    if (wn > 1e-9) {
      const wh = [w[0] / wn, w[1] / wn], cross = wh[0] * c[1] - wh[1] * c[0];
      const a_abs = Math.atan2(Math.abs(cross), wh[0] * c[0] + wh[1] * c[1]) * R2D;
      let CDr; [CLr, CDr] = rudderCoeffs(a_abs, p);
      const rot = [-wh[1], wh[0]], Ld = Math.abs(cross) > 1e-12 ? [-sign(cross) * rot[0], -sign(cross) * rot[1]] : rot;
      const k = 0.5 * RHO_WATER * p.A_rud * wn * wn;
      Fr = [k * (CLr * Ld[0] + CDr * wh[0]), k * (CLr * Ld[1] + CDr * wh[1])];
    }
    const N_r = x_r * Fr[1];
    const X = Fa[0] + X_h + Fr[0], Y = Fa[1] + Y_h + Yd + Fr[1];
    const N_munk = -p.k_munk * (p.m22 - p.m11) * u * v;
    const N = N_s + N_h + N_r + N_munk;
    return { X, Y, N, Fx: Fa[0], Fy: Fa[1], N_s, N_h, N_r, N_munk, Y_h: Y_h + Yd, Y_r: Fr[1], X_r: Fr[0],
             R: R + Di - Fr[0] * sign(u || 1.0), beta, delta_s, sf, a_flow, CL_r: CLr };
  }

  function deriv(st, d_r, env) {                                    // predict_path.deriv / _route_deriv
    const p = env.p, Mu = p.mass + p.m11, Mv = p.mass + p.m22, Mr = p.Iz + p.Iz_added;
    const [, , h, u, v, r] = st;
    const b = bodyForces(u, v, r, h, d_r, env);
    const f = cu(h), s = cu(h + 90.0);
    return [[u * f[0] + v * s[0], u * f[1] + v * s[1], r * R2D,
             (b.X + Mv * v * r) / Mu, (b.Y - Mu * u * r) / Mv, b.N / Mr], b];
  }
  function rk4(st, d_r, env, dt, k1) {                              // dümen adım boyunca sabit
    const a = 0.5 * dt, add = (x, k, m) => x.map((xi, i) => xi + m * k[i]);
    const k2 = deriv(add(st, k1, a), d_r, env)[0];
    const k3 = deriv(add(st, k2, a), d_r, env)[0];
    const k4 = deriv(add(st, k3, dt), d_r, env)[0];
    const m = dt / 6.0;
    return st.map((x, i) => x + m * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
  }

  // Kaskad PID (simulate_route ile aynı). rt: { A, chi, t_hat, n_hat, lim, V_ref }, ap: Autopilot alanları,
  // cs: denetleyici durumu { I_e, I_h }. Dümen KOMUTUNU ve ara değerleri döndürür (servo dışarıda).
  function autopilot(st, rt, ap, cs, dt) {
    const [x, y, h, u, v, r] = st;
    const rel = [x - rt.A[0], y - rt.A[1]];
    const e = rel[0] * rt.n_hat[0] + rel[1] * rt.n_hat[1];
    const f = cu(h), sv = cu(h + 90.0), vel = [u * f[0] + v * sv[0], u * f[1] + v * sv[1]];
    const V = Math.hypot(u, v), lam = V > 1e-3 ? Math.atan2(v, u) * R2D : 0.0;
    const e_dot = vel[0] * rt.n_hat[0] + vel[1] * rt.n_hat[1];
    const o_P = -ap.kp_e * e, o_D = -ap.kd_e * e_dot, o_I = -ap.ki_e * cs.I_e;
    const dpsi_raw = o_P + o_I + o_D, dpsi = Math.max(-ap.dpsi_max, Math.min(ap.dpsi_max, dpsi_raw));
    if (Math.abs(e) < 5.0 && dpsi === dpsi_raw && ap.ki_e > 0)
      cs.I_e = Math.max(-10.0 / ap.ki_e, Math.min(10.0 / ap.ki_e, cs.I_e + e * dt));
    const o_FF = ap.leeway_ff && V > 0.1 ? -lam : 0.0;
    let psi_star = rt.chi + dpsi + o_FF, clamp = false;
    const twa_star = wrap180(psi_star - WIND_FROM_DEG);
    if (rt.lim !== null && Math.abs(twa_star) < rt.lim + 3.0) {
      const side = Math.abs(twa_star) > 1e-6 ? copysign(1.0, twa_star) : copysign(1.0, wrap180(h - WIND_FROM_DEG) || 1.0);
      psi_star = WIND_FROM_DEG + side * (rt.lim + 3.0); clamp = true;
    }
    const eps = wrap180(psi_star - h);
    const g = ap.gain_sched ? Math.min(3.0, Math.max(0.5, (rt.V_ref / Math.max(V, 0.2)) ** 2)) : 1.0;
    const i_P = g * ap.kp_h * eps, i_D = -g * ap.kd_h * (r * R2D), i_I = g * ap.ki_h * cs.I_h;
    const cmd_raw = i_P + i_I + i_D, cmd = Math.max(-ap.rud_max, Math.min(ap.rud_max, cmd_raw));
    if (cmd === cmd_raw && ap.ki_h > 0) cs.I_h = Math.max(-10.0 / ap.ki_h, Math.min(10.0 / ap.ki_h, cs.I_h + eps * dt));
    return { cmd, e, V, lam, psi_star, dpsi, o_P, o_I, o_D, o_FF, clamp, eps, g, i_P, i_I, i_D };
  }
  const servo = (dr, cmd, rate, dt) => dr + Math.max(-rate * dt, Math.min(rate * dt, cmd - dr));

  // simulate_route'un aynısı (yalnız test için): Python'dan gelen başlangıçla, t_max süre.
  function simulateRoute(st0, rt, ap, env, t_max, dt = 0.025, every = 10) {
    let st = st0.slice(), dr = 0.0;
    const cs = { I_e: 0.0, I_h: 0.0 }, rec = { t: [], x: [], y: [], h: [], u: [], v: [], r: [], dr: [], e: [] };
    for (let k = 0; k <= Math.round(t_max / dt); k++) {
      const a = autopilot(st, rt, ap, cs, dt);
      dr = servo(dr, a.cmd, ap.rud_rate, dt);
      const [k1] = deriv(st, dr * D2R, env);
      if (k % every === 0) {
        rec.t.push(k * dt); rec.x.push(st[0]); rec.y.push(st[1]); rec.h.push(pymod(st[2], 360.0));
        rec.u.push(st[3]); rec.v.push(st[4]); rec.r.push(st[5] * R2D); rec.dr.push(dr); rec.e.push(a.e);
      }
      st = rk4(st, dr * D2R, env, dt, k1);
    }
    return rec;
  }
  return { KT2MS, D2R, R2D, WIND_FROM_DEG, wrap180, pymod, cu, bodyForces, deriv, rk4, autopilot, servo, simulateRoute };
})();
