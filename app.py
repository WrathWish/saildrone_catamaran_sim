"""
Modül 1 — Statik Kuvvet Dengesi & Rota Tahmini
Otonom Katamaran (iskele TURUNCU, sancak SİYAH, kıçta TEK merkezi dümen)

Çalıştırma:  streamlit run app.py

BİLİMSEL DAYANAK
  [M22] A. Morge, V. Pelle, J. Wan, L. Jaulin, "Experimental Studies of Autonomous
        Sailing With a Radio Controlled Sailboat", IEEE Access, vol.10, 2022.
  [J13] L. Jaulin, F. Le Bars, "An Interval Approach for Stability Analysis:
        Application to Sailboat Robotics", IEEE T-RO, vol.29, no.1, 2013.

  * Görünür rüzgar  : [M22] Eq.(1)-(2)  T_w = v·[cosφ, sinφ] + a_aw·[cos(ψ_aw+θ), sin(ψ_aw+θ)]
                      → buradan a_aw vektörü = T_w − v (aynı denklemin yeniden düzenlenmesi).
                      φ (rota/course) ≠ θ (pruva/heading): fark = drift (leeway).
  * Yelken trimi    : [J13] Eq.(11)  δs_max = (π/2)·(cos(ψ − θ̄) + 1)/2   (kardioid)
                      δs = ±δs_max (yelken rüzgar altı tarafa).
  * Dümen işareti   : [J13] Eq.(11)  δr = δr_max·sin(θ − θ̄) → +δr tekneyi sancağa döndürür.
  * Ölü açı         : KUVVET DENGESİNDEN — ileri denge (itiş = direnç) kökü yoksa tekne yol alamaz.
                      Sınır, optimum trimle köklerin başladığı TWA olarak hesaplanır.
                      [J13] Eq.(12) cos(ψ − θ̄) + cos ζ ≥ 0 (ζ = 45°) yalnız karşılaştırma çizgisidir.
  * Yönelim/rota    : 3-DOF durağan denge (X ileri, Y yanal, N dönme momenti). Yelken kuvveti
                      basınç merkezinden (CE), gövde kuvveti yanal direnç merkezinden (CLR), dümen
                      kuvveti kıçtan uygulanır; dönüş hızı r moment dengesinden ÇIKAR (uydurma kazanç yok).

  ÖNEMLİ: İki makalede de yelken/gövde kuvvet KATSAYILARI (CL, CD, su direnci,
  yanal direnç, dönüş kazancı) VERİLMEZ ([M22] bunun için ref.[13] Melin 2015'e,
  [J13] ise ref.[15] Jaulin 2015'e atıf yapar). Bu katsayılar aşağıda açıkça
  "VARSAYIM" olarak işaretlenmiştir ve arayüzden elle değiştirilebilir.
"""

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Sabitler
# ----------------------------------------------------------------------------
KT2MS = 0.514444
RHO_AIR = 1.225      # kg/m³
RHO_WATER = 1025.0   # kg/m³ (deniz suyu)
NU_WATER = 1.19e-6   # m²/s, kinematik viskozite (15 °C deniz suyu)
G = 9.81
WIND_FROM_DEG = 0.0  # Rüzgar KUZEYDEN eser (0°)
# [J13] ψ: rüzgarın ESTİĞİ yön (pupa'da cos(ψ−θ̄)=1). Kuzeyden esen rüzgar güneye eser:
PSI_WIND_TOWARD_DEG = (WIND_FROM_DEG + 180.0) % 360.0

COL_PORT = "#FF7F0E"       # iskele — TURUNCU
COL_STBD = "#111111"       # sancak — SİYAH
COL_RUDDER = "#1F4E79"
COL_FX = "#2CA02C"         # ileri itiş — YEŞİL
COL_FY = "#D62728"         # yanal kuvvet — KIRMIZI
COL_PATH = "#00CED1"       # rota — TURKUAZ
COL_WIND = "#1F77B4"
COL_YR = "#8E44AD"         # dümen yanal kuvveti — MOR
COL_YH = "#566573"         # gövde yanal kuvveti — GRİ


@dataclass(frozen=True)
class Params:
    # --- Prototip donanımı (proje protokolü) ---
    L: float = 1.80            # m, boy (≈ su hattı boyu kabul)
    mass: float = 25.0         # kg → deplasman → su çekimi, ıslak alan, doğrultucu moment
    hull_sep: float = 0.80     # m, gövde eksenleri arası
    zeta_deg: float = 45.0     # [J13] Eq.(12) close-hauled açısı
    # --- VARSAYIMLAR (makalelerde yok) ---
    hull_B: float = 0.18       # m, tek gövde su hattı genişliği
    Cb: float = 0.45           # blok katsayısı
    form_k: float = 0.20       # form faktörü (1+k)
    C_R: float = 0.004         # artık (dalga) direnç katsayısı — sabit kabul
    A_wing: float = 0.59       # m², sert kanat alanı (0.45 m kord × 1.3 m)
    AR_wing: float = 3.5       # kanat açıklık oranı
    A_soft: float = 0.75       # m², bez yelken alanı
    alpha_stall: float = 12.0  # °, NACA 0012 düşük-Re tutunma kaybı başlangıcı
    alpha_luff: float = 8.0    # °, bez yelkenin sönme (flapping) eşiği
    CN_plate: float = 1.25     # sonlu düz levha normal kuvvet katsayısı (α=90°)
    CLa_hull: float = 1.5      # 1/rad, gövde yanal kaldırma eğimi (düşük AR)
    AR_hull: float = 0.5       # gövde efektif açıklık oranı (indüklenmiş direnç)
    A_rud: float = 0.02        # m², merkezi dümen alanı
    AR_rud: float = 2.0        # dümen açıklık oranı
    rud_stall: float = 20.0    # °, dümen tutunma kaybı
    x_mast: float = 0.12       # m, direk konumu (merkezin önü +)
    x_CLR: float = 0.00        # m, gövdelerin yanal direnç merkezi (merkezin önü +)
    chord_wing: float = 0.45   # m, kanat kordu
    boom_len: float = 0.55     # m, bumba boyu
    ce_frac: float = 0.35      # bez yelken CE'sinin bumba boyunca konumu (üçgen ağırlık merkezi ≈ 1/3)
    Cd_cross: float = 1.0      # gövde çapraz akış sürükleme katsayısı (büyük drift açıları)
    max_leeway: float = 85.0   # °, sayısal sınır
    boom_rigid: bool = False   # bez yelken bumbası: False = ıskota (ip, yalnız çeker), True = rijit servo
    k_munk: float = 1.0        # Munk momenti kalibrasyon katsayısı (1 = potansiyel akış teorisi)
    CDA_windage: float = 0.16  # m², su üstü yapı sürükleme alanı CD·A (gövde+kiriş+direk+kutu)
    h_CE: float = 0.75         # m, yelken basınç merkezi ↔ yanal direnç merkezi kolu

    # --- Kütleden türetilen hidrostatik büyüklükler ---
    @property
    def draft(self):           # ∇_gövde = Cb·L·B·T
        return self.mass / RHO_WATER / 2 / (self.Cb * self.L * self.hull_B)

    @property
    def S_wet(self):           # Mumford: S = 1.7·L·T + ∇/T  (gövde başına) ×2
        T = self.draft
        return 2 * (1.7 * self.L * T + self.mass / RHO_WATER / 2 / T)

    @property
    def A_lat(self):           # iki gövde yanal izdüşümü + dümen
        return 2 * 0.85 * self.L * self.draft + self.A_rud

    @property
    def A_hull_lat(self):      # iki gövdenin yanal izdüşümü
        return 2 * 0.85 * self.L * self.draft

    @property
    def CLa_rud(self):         # sonlu kanat: 2π/(1+2/AR)
        return 2 * math.pi / (1 + 2 / self.AR_rud)

    @property
    def x_rud(self):           # dümen kıçta, simetri ekseninde
        return -self.L / 2

    @property
    def m11(self):             # ileri ek kütle (VARSAYIM: ince gövde ≈ %5 m)
        return 0.05 * self.mass

    @property
    def m22(self):             # yanal ek kütle, şerit teorisi: serbest yüzeyli düşey levha ρπT²/2 (gövde başına)
        return 2 * RHO_WATER * math.pi * self.draft ** 2 / 2 * 0.85 * self.L

    @property
    def Iz(self):              # iki ince gövde (her biri m/2, boy L, ±b/2): m(L²/12 + (b/2)²)
        return self.mass * (self.L ** 2 / 12 + (self.hull_sep / 2) ** 2)

    @property
    def Iz_added(self):        # dönme ek eylemsizliği: ∫ m22'(x)·x² dx
        return self.m22 * self.L ** 2 / 12

    @property
    def righting_moment(self):  # rüzgarüstü gövde suyu terk ettiği an: m·g·(b/2)
        return self.mass * G * self.hull_sep / 2


# ----------------------------------------------------------------------------
# Yardımcılar  (pusula açıları: 0°=Kuzey, saat yönünde pozitif; dünya: x=Doğu, y=Kuzey)
# ----------------------------------------------------------------------------
def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def compass_unit(deg):
    r = math.radians(deg)
    return np.array([math.sin(r), math.cos(r)])


def no_go(heading_deg, zeta_deg):
    """[J13] Eq.(12): rota uygulanabilir ⇔ cos(ψ − θ̄) + cos ζ ≥ 0."""
    val = math.cos(math.radians(PSI_WIND_TOWARD_DEG - heading_deg)) + math.cos(math.radians(zeta_deg))
    return val < -1e-9


def cardioid_dsmax(heading_deg):
    """[J13] Eq.(11): δs_max = (π/2)·(cos(ψ − θ̄)+1)/2  → derece."""
    return 90.0 * (math.cos(math.radians(PSI_WIND_TOWARD_DEG - heading_deg)) + 1.0) / 2.0


def point_of_sail(heading_deg, eq, limit):
    """Durum rozeti KUVVET DENGESİNDEN: ileri denge kökü yoksa tekne yol alamaz."""
    twa = abs(wrap180(heading_deg - WIND_FROM_DEG))
    if eq["V"] == 0:
        if limit is None or twa < limit:
            return "ÖLÜ AÇI (No-Go Zone)", "#C0392B", twa
        return "YOL YOK — trim uygun değil", "#8E44AD", twa
    if twa < 80:
        return "ORSA", "#E67E22", twa
    if twa < 150:
        return "APAZ", "#27AE60", twa
    return "PUPA", "#2980B9", twa


@lru_cache(maxsize=32)
def _opt_trim_table(sail_type, p):
    """Her görünür rüzgar açısı β için yelken itişini (Fx) maksimum yapan |δs| (kuvvet hesabından).
    Kuvvetler a_aw²'ye orantılı olduğundan en iyi trim yalnızca β'ya bağlıdır."""
    betas = np.arange(0.0, 180.01, 1.0)
    deltas = np.arange(0.0, 90.01, 0.5)
    best = np.zeros_like(betas)
    for i, b_ in enumerate(betas):
        br = math.radians(b_)
        aw_b = -np.array([math.cos(br), math.sin(br)])     # rüzgar sancaktan → yelken iskelede (δs<0)
        fx = [sail_forces(aw_b, -d_, sail_type, p)["Fx"] for d_ in deltas]
        best[i] = deltas[int(np.argmax(fx))]
    return betas, best


def optimal_trim_mag(beta_abs, sail_type, p):
    betas, best = _opt_trim_table(sail_type, p)
    return float(np.interp(beta_abs, betas, best))


def friction_coeff(V, p):
    """ITTC-1957 sürtünme çizgisi: Cf = 0.075/(log10 Re − 2)²."""
    Re = max(V * p.L / NU_WATER, 1e4)
    return 0.075 / (math.log10(Re) - 2.0) ** 2


# ----------------------------------------------------------------------------
# Yelken aerodinamiği  (gövde ekseni: [ileri, sancak]; açılar pruvadan saat yönünde)
# ----------------------------------------------------------------------------
_SOFT_A = [8, 15, 22, 30, 45, 60, 90]                 # ° (VARSAYIM: tipik bez yelken eğrisi şekli)
_SOFT_CL = [0.00, 1.15, 1.40, 1.30, 1.00, 0.65, 0.0]
_SOFT_CD = [0.10, 0.15, 0.25, 0.40, 0.75, 1.00, 1.25]


def _smoothstep(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def sail_coeffs(a_abs, sail_type, backwinded, p):
    """CL (işaretli), CD, flapping.  VARSAYIM: makalelerde katsayı yok. Tüm eğriler süreklidir."""
    if a_abs <= 90.0:
        a_e, sgn = a_abs, 1.0
    else:  # akış firar kenarından geliyor → ayna simetri, kaldırma yön değiştirir
        a_e, sgn = 180.0 - a_abs, -1.0
    ar = math.radians(a_e)

    if sail_type == "wing":  # NACA 0012 simetrik profil
        CD0 = 0.012
        CLa = 2 * math.pi / (1 + 2 / p.AR_wing)          # sonlu açıklık düzeltmesi
        CL_att = CLa * ar
        CD_att = CD0 + CL_att ** 2 / (math.pi * 0.9 * p.AR_wing)
        CL_sep = p.CN_plate * math.sin(ar) * math.cos(ar)  # ayrılmış akış: normal kuvvet modeli
        CD_sep = CD0 + p.CN_plate * math.sin(ar) ** 2
        t = _smoothstep((a_e - p.alpha_stall) / 6.0)       # sürekli geçiş (sıçrama yok)
        return sgn * ((1 - t) * CL_att + t * CL_sep), (1 - t) * CD_att + t * CD_sep, False, 0.25 + 0.25 * t

    # Bez yelken
    if a_e < p.alpha_luff:
        return 0.0, 0.10, True, 0.25  # LUFF / ÇIRPMA (akışa hizalı) / FLAPPING: kaldırma yok, yalnız çırpınma direnci
    CL = float(np.interp(a_e, _SOFT_A, _SOFT_CL))
    CD = float(np.interp(a_e, _SOFT_A, _SOFT_CD))
    return sgn * CL, CD, False, 0.25


def sail_forces(aw_b, delta_s, sail_type, p):
    """aw_b: görünür rüzgar AKIŞ vektörü (gövde ekseni). δs>0 → bumba sancakta."""
    aaw = float(np.hypot(*aw_b))
    out = dict(Fx=0.0, Fy=0.0, alpha=0.0, flapping=sail_type == "soft", backwinded=False, aaw=aaw, cp=0.25)
    if aaw < 1e-9:
        return out
    w = aw_b / aaw
    c_ang = math.radians(180.0 - delta_s)            # kord: ön kenar → firar kenarı
    c = np.array([math.cos(c_ang), math.sin(c_ang)])
    cross = w[0] * c[1] - w[1] * c[0]
    a_abs = math.degrees(math.atan2(abs(cross), float(w @ c)))

    # Bez yelken ters rüzgar kontrolü: bumba rüzgarüstü taraftaysa yelken söner
    sin_beta = -w[1]
    backwinded = False
    if sail_type == "soft" and abs(delta_s) > 1.0 and abs(sin_beta) > math.sin(math.radians(10)):
        backwinded = bool(np.sign(delta_s) == np.sign(sin_beta))

    CL, CD, flapping, cp = sail_coeffs(a_abs, sail_type, backwinded, p)
    area = p.A_wing if sail_type == "wing" else p.A_soft
    q = 0.5 * RHO_AIR * area * aaw ** 2
    rot = np.array([-w[1], w[0]])
    L_dir = -np.sign(cross) * rot if abs(cross) > 1e-12 else rot
    F = q * (CL * L_dir + CD * w)                    # kaldırma ⟂ akış, sürükleme ∥ akış
    out.update(Fx=float(F[0]), Fy=float(F[1]), alpha=a_abs, flapping=flapping, backwinded=backwinded, cp=cp)
    return out


# ----------------------------------------------------------------------------
# 3-DOF denge (Fossen tipi rijit gövde denklemleri, durağan hal). Dinamik rota: predict_path
#   Gövde ekseni: x ileri, y sancak; r > 0 → sancağa dönüş (saat yönü).  u = V·cosλ, v = V·sinλ
#   X:  Fx_yelken − R·cosλ − D_dümen − D_ind + m·v·r = 0
#   Y:  Fy_yelken + Y_gövde + Y_dümen − m·u·r = 0          (merkezcil)
#   N:  N_yelken  + N_gövde + N_dümen = 0                    (yönelim / dönme momenti)
#   Y_gövde = −q·A_h·CLα_h·(v/u),  N_gövde = x_h·Y_gövde − q·A_h·CLα_h·(r·L/u)·L/12   (şerit teorisi)
#   Y_dümen = −q·A_r·CLα_r·α_r,  α_r = δr + v/u + r·x_r/u,  N_dümen = x_r·Y_dümen
#   N_yelken = x_CE·Fy − y_CE·Fx  (basınç merkezi kuvvet diyagramından)
# ----------------------------------------------------------------------------
def sail_CE(delta_s, sail_type, p, cp=0.25):
    """Yelken basınç merkezi (gövde ekseni). Kanat çeyrek kordda döner → CE kayması (cp−0.25)·kord."""
    a = math.radians(180.0 - delta_s)
    c = np.array([math.cos(a), math.sin(a)])
    if sail_type == "wing":
        return np.array([p.x_mast, 0.0]) + (cp - 0.25) * p.chord_wing * c
    return np.array([p.x_mast, 0.0]) + p.ce_frac * p.boom_len * c


def _rudder_coeffs(a_abs, p):
    """Merkezi dümen: ince kanat (CLα, sonlu AR) + tutunma sonrası düz levha; ters akışta ayna simetri."""
    if a_abs <= 90.0:
        a_e, sgn = a_abs, 1.0
    else:
        a_e, sgn = 180.0 - a_abs, -1.0
    ar = math.radians(a_e)
    CL_att = p.CLa_rud * ar
    CD_att = 0.008 + CL_att ** 2 / (math.pi * 0.9 * p.AR_rud)
    CL_sep = p.CN_plate * math.sin(ar) * math.cos(ar)
    CD_sep = 0.008 + p.CN_plate * math.sin(ar) ** 2
    t = _smoothstep((a_e - p.rud_stall) / 6.0)
    return sgn * ((1 - t) * CL_att + t * CL_sep), (1 - t) * CD_att + t * CD_sep


def body_forces(u, v, r, heading, tws_kt, d_r, trim_auto, trim_manual, sail_type, p):
    """Gövde ekseninde (x ileri, y sancak) TÜM dış kuvvetler ve z-momenti. Her akış yönünde geçerli
    (ileri, geri/sternway, yan). Statik denge ve zaman entegrasyonu AYNI fonksiyonu kullanır."""
    tw = tws_kt * KT2MS * compass_unit(PSI_WIND_TOWARD_DEG)     # gerçek rüzgar akışı (ENU)
    f, s = compass_unit(heading), compass_unit(heading + 90.0)
    aw_b = np.array([tw @ f - u, tw @ s - v])                   # [M22] Eq.(2): a_aw = T_w − v
    beta = math.degrees(math.atan2(-aw_b[1], -aw_b[0]))         # görünür rüzgarın GELDİĞİ açı (+sancak)
    side = -np.sign(math.sin(math.radians(beta))) or -1.0       # yelken rüzgar altına
    if trim_auto == "opt":                                      # kuvvet hesabından maks. itiş trimi
        delta_s = side * optimal_trim_mag(abs(beta), sail_type, p)
    elif trim_auto:                                             # [J13] Eq.(11) kardioid
        delta_s = side * cardioid_dsmax(heading)
    else:
        delta_s = float(trim_manual)
    streaming = False
    if sail_type == "soft" and not p.boom_rigid:
        # Iskota (ip) yalnız ÇEKER: |δs| ≤ δs_max, bumba rüzgar altına serbest savrulur ([J13] §IV-A).
        dmax = min(abs(delta_s), 90.0)
        d_stream = wrap180(-beta)                               # bumbanın akışa hizalandığı (bayrak) açı
        if abs(d_stream) <= dmax:
            delta_s, streaming = d_stream, True                 # yelken akışa hizalı → çırpar (luff)
        else:
            delta_s = side * dmax                               # ıskota gergin, yelken dolu
    sf = dict(sail_forces(aw_b, delta_s, sail_type, p))
    sf["streaming"] = streaming
    ce = sail_CE(delta_s, sail_type, p, sf["cp"])
    N_s = ce[0] * sf["Fy"] - ce[1] * sf["Fx"]
    Wd = 0.5 * RHO_AIR * p.CDA_windage * float(np.hypot(*aw_b)) * aw_b   # su üstü yapı rüzgar direnci
    Fa = np.array([sf["Fx"] + Wd[0], sf["Fy"] + Wd[1]])

    V = math.hypot(u, v)
    qw = 0.5 * RHO_WATER * V * V
    # Gövde: şerit teorisi — yerel yanal hız v + r·x; doğrusal kaldırma (|u|·CLα) + çapraz akış (Cd·|v|v)
    n = 20
    xs = (np.arange(n) + 0.5) / n * p.L - p.L / 2 + p.x_CLR
    dx = p.L / n
    vl = v + r * xs
    dY = -0.5 * RHO_WATER * (p.A_hull_lat / p.L) * (p.CLa_hull * abs(u) * vl + p.Cd_cross * vl * np.abs(vl)) * dx
    Y_h, N_h = float(dY.sum()), float((xs * dY).sum())
    X_h = Yd = 0.0
    if V > 1e-9:
        lam = math.atan2(v, u)
        CL_h = p.CLa_hull * math.sin(lam) * abs(math.cos(lam))
        Di = qw * p.A_hull_lat * CL_h ** 2 / (math.pi * p.AR_hull)            # indüklenmiş direnç
        R = qw * ((1 + p.form_k) * friction_coeff(V, p) + p.C_R) * p.S_wet    # ITTC-57 + C_R
        X_h, Yd = -(R + Di) * u / V, -(R + Di) * v / V                        # hızın tersi yönünde
    else:
        Di = R = 0.0
    # Merkezi dümen (kıç, x_r): yerel su akışı w = −(u, v + r·x_r); kord ön kenar → firar kenarı
    x_r = p.x_rud
    w = -np.array([u, v + r * x_r])
    wn = float(np.hypot(*w))
    c = np.array([-math.cos(d_r), math.sin(d_r)])               # +δr → firar kenarı sancağa
    Fr = np.zeros(2)
    a_flow = math.degrees(math.atan2(v + r * x_r, u)) if wn > 1e-9 else 0.0
    CLr = 0.0
    if wn > 1e-9:
        wh = w / wn
        cross = wh[0] * c[1] - wh[1] * c[0]
        a_abs = math.degrees(math.atan2(abs(cross), float(wh @ c)))
        CLr, CDr = _rudder_coeffs(a_abs, p)
        rot = np.array([-wh[1], wh[0]])
        L_dir = -np.sign(cross) * rot if abs(cross) > 1e-12 else rot
        Fr = 0.5 * RHO_WATER * p.A_rud * wn * wn * (CLr * L_dir + CDr * wh)
    N_r = x_r * float(Fr[1])

    X = float(Fa[0]) + X_h + float(Fr[0])
    Y = float(Fa[1]) + Y_h + Yd + float(Fr[1])
    # Munk momenti (ek kütle Coriolis'i, Fossen 3-DOF): −(m22 − m11)·u·v — ince gövdeyi akışa dik çevirir
    N_munk = -p.k_munk * (p.m22 - p.m11) * u * v
    N = N_s + N_h + N_r + N_munk
    return dict(X=X, Y=Y, N=N, Fx=float(Fa[0]), Fy=float(Fa[1]), N_s=N_s, N_h=N_h, N_r=N_r, N_munk=N_munk,
                Y_h=Y_h + Yd, Y_r=float(Fr[1]), X_r=float(Fr[0]), R=R + Di - float(Fr[0]) * np.sign(u or 1.0),
                ce=ce, sf=sf, aw_b=aw_b, beta=beta, delta_s=delta_s, a_flow=a_flow, CL_r=CLr)


def _state(V, heading, tws_kt, delta_r, trim_auto, trim_manual, sail_type, p, lam0=0.0, r0=0.0, with_bal=False):
    """Durağan hal: ΣY − m·u·r = 0, ΣN = 0 Newton ile (λ, r); ileri artık g = ΣX + m·v·r."""
    m, L = p.mass, p.L
    d_r = math.radians(delta_r)
    qw = 0.5 * RHO_WATER * V ** 2
    dsmax = cardioid_dsmax(heading)

    def forces(lam, r, dr=None):
        dr = d_r if dr is None else dr
        u, v = V * math.cos(lam), V * math.sin(lam)
        b = body_forces(u, v, r, heading, tws_kt, dr, trim_auto, trim_manual, sail_type, p)
        return np.array([b["Y"] - (m + p.m11) * u * r, b["N"]]), b

    def newton(lam, r):
        F, d = forces(lam, r)
        scale = np.array([max(1.0, qw * p.A_hull_lat), max(1.0, qw * p.A_hull_lat * L)])
        for _ in range(60):
            if np.all(np.abs(F) < 1e-9 * scale):
                return lam, r, F, d, True
            J = np.empty((2, 2))
            for k, (dl, drr) in enumerate(((1e-7, 0.0), (0.0, 1e-7))):
                J[:, k] = (forces(lam + dl, r + drr)[0] - F) / 1e-7
            try:
                step = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                break
            t_ = 1.0
            n0 = np.linalg.norm(F / scale)
            while t_ > 1e-4:
                lam_t = max(-1.48, min(1.48, lam + t_ * step[0]))      # |λ| < 85°
                F_t, d_t = forces(lam_t, r + t_ * step[1])
                if np.linalg.norm(F_t / scale) < n0:
                    lam, r, F, d = lam_t, r + t_ * step[1], F_t, d_t
                    break
                t_ *= 0.5
            else:
                break
        return lam, r, F, d, bool(np.all(np.abs(F) < 1e-6 * scale))

    converged = True
    if V > 1e-4:
        lam, r, F, d, converged = newton(lam0, r0)
        if not converged or abs(lam) > 1.4:                       # yanlış dala kaçtıysa sıfırdan başla
            cand = newton(0.0, 0.0)
            if cand[4] or not converged:
                lam, r, F, d, converged = cand
    else:
        lam, r = 0.0, 0.0
        F, d = forces(0.0, 0.0)
    g = d["X"] + (m + p.m22) * V * math.sin(lam) * r             # ΣX + (m+m22)·v·r = 0 (durağan)
    d_bal = None
    if V > 1e-4 and with_bal:                                    # düz rota (r=0) için gereken dümen
        lb, db = lam, 0.0
        for _ in range(40):
            Fb = forces(lb, 0.0, db)[0]
            if np.all(np.abs(Fb) < 1e-9 * max(1.0, qw)):
                break
            J = np.empty((2, 2))
            J[:, 0] = (forces(lb + 1e-7, 0.0, db)[0] - Fb) / 1e-7
            J[:, 1] = (forces(lb, 0.0, db + 1e-7)[0] - Fb) / 1e-7
            try:
                stp = np.linalg.solve(J, -Fb)
            except np.linalg.LinAlgError:
                break
            lb, db = max(-1.48, min(1.48, lb + stp[0])), db + stp[1]
        ok = np.all(np.abs(forces(lb, 0.0, db)[0]) < 1e-6 * max(1.0, qw))
        d_bal = math.degrees(db) if ok and abs(db) < math.radians(80) else None
    sf = d["sf"]
    return dict(g=g, V=V, lam=lam, r=r, Fx_raw=d["Fx"], Fx=d["Fx"], Fy=d["Fy"], R=d["R"],
                beta=d["beta"], aaw=sf["aaw"], delta_s=d["delta_s"], dsmax=dsmax, alpha=sf["alpha"],
                flapping=sf["flapping"], backwinded=sf["backwinded"], aw_b=d["aw_b"],
                N_s=d["N_s"], N_h=d["N_h"], N_r=d["N_r"], N_munk=d["N_munk"], Y_h=d["Y_h"], Y_r=d["Y_r"],
                ce=d["ce"], streaming=sf["streaming"], alpha_r=delta_r + d["a_flow"], a_flow=d["a_flow"],
                rud_sat=abs(d["a_flow"] + delta_r) > p.rud_stall, d_bal=d_bal, YN_resid=F,
                converged=converged)


def solve_equilibrium(heading, tws_kt, delta_r, trim_auto, trim_manual, sail_type, p, V0=None):
    """İleri denge X(V)=0 kökü. Birden çok kök olabilir (düşük hızda yana sürüklenme dalı);
    yüksek hızdan aşağı süreklilikle taranır ve en yüksek hızlı KARARLI kök (dX/dV<0) seçilir."""
    args = (heading, tws_kt, delta_r, trim_auto, trim_manual, sail_type, p)
    tws = tws_kt * KT2MS

    def ok(s):
        return s is not None and s["converged"]

    def bisect(a, b, lam, r):
        lo, hi = a, b
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            s = _state(mid, *args, lam0=lam, r0=r)
            if ok(s):
                lam, r = s["lam"], s["r"]
            if ok(s) and s["g"] > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-6:
                break
        return _state(0.5 * (lo + hi), *args, lam0=lam, r0=r, with_bal=True)

    res = None
    if V0:                                       # sıcak başlangıç (rota boyunca): dar aralık
        sa, sb = _state(0.8 * V0, *args), _state(1.25 * V0, *args)
        if ok(sa) and ok(sb) and sa["g"] > 0 and sb["g"] < 0:
            res = bisect(0.8 * V0, 1.25 * V0, sa["lam"], sa["r"])
    if res is None:
        grid = np.geomspace(0.01, 1.5 * max(tws, 0.5), 40)[::-1]   # yüksekten düşüğe süreklilik
        prev = None
        lam = r = 0.0
        for Vg in grid:
            s = _state(float(Vg), *args, lam0=lam, r0=r)
            if not ok(s):
                s = _state(float(Vg), *args)
            if ok(s):
                lam, r = s["lam"], s["r"]
                if prev is not None and s["g"] > 0 and prev[1]["g"] < 0:
                    res = bisect(float(Vg), prev[0], s["lam"], s["r"])
                    break
                prev = (float(Vg), s)
            else:
                prev = None
    if res is None:
        res = _state(0.0, *args)
        res["no_eq"], res["crab"] = False, False
        V = 0.0
    else:
        V = res["V"]
    # İleri denge kökü yoksa (g(V) süreksiz → ikiye bölme bir sıçramaya yakınsar) tekne yol alamaz
    # Kontrollü seyir koşulu (dümen fiziği): dümene gelen yerel akış açısı dümen stall açısını aşarsa
    # tekne yön tutamaz → yana sürüklenme dalı, "yol alıyor" sayılmaz.
    crab = V > 1e-3 and abs(res["a_flow"]) > p.rud_stall
    no_eq = V > 1e-3 and (abs(res["g"]) > 1e-3 * max(1.0, res["R"]) or not res["converged"])
    moving = V > 1e-3 and not no_eq and not crab
    if not moving and V > 1e-3:
        res = _state(0.0, *args)
        V = 0.0
    res["no_eq"], res["crab"] = bool(no_eq), bool(crab)
    res["V"] = V if moving else 0.0
    res["lam"] = math.degrees(res["lam"]) if moving else None          # durmuş teknede drift tanımsız
    res["r"] = math.degrees(res["r"]) if moving else 0.0               # °/s
    res["radius"] = (V / abs(math.radians(res["r"]))) if moving and abs(res["r"]) > 1e-4 else math.inf
    res["course"] = heading + (res["lam"] or 0.0)
    res["Fr"] = res["V"] / math.sqrt(G * p.L)
    res["HM"] = abs(res["Fy"]) * p.h_CE
    res["RM"] = p.righting_moment
    res["capsize"] = res["HM"] > res["RM"]
    res["residual"] = abs(res["g"]) if moving else 0.0
    return res


@lru_cache(maxsize=64)
def nogo_limit(tws_kt, sail_type, p):
    """Fiziksel ölü açı sınırı (TWA, °): optimum trim ve dümen 0 ile KONTROLLÜ ileri dengenin
    ilk var olduğu açı. Sabit bir ζ kabulü YOK — tamamen kuvvet dengesinden (ikiye bölme, 0.1°)."""
    def moves(twa):
        return solve_equilibrium(float(twa), tws_kt, 0, "opt", 0, sail_type, p)["V"] > 0
    if moves(0.0):
        return 0.0
    hi = next((a for a in (90.0, 120.0, 150.0, 180.0) if moves(a)), None)
    if hi is None:
        return None
    lo = 0.0
    while hi - lo > 0.1:
        mid = 0.5 * (lo + hi)
        if moves(mid):
            hi = mid
        else:
            lo = mid
    return hi


def predict_path(heading, tws_kt, delta_r, trim_auto, trim_manual, sail_type, p, horizon, dt=0.025,
                 init=None):
    """DİNAMİK rota: 3-DOF hareket denklemleri RK4 ile entegre edilir (durağan denge yalnız başlangıç).
       (m+m11)·u̇ = ΣX + (m+m22)·v·r,  (m+m22)·v̇ = ΣY − (m+m11)·u·r,  (Iz+Iz_ek)·ṙ = ΣN (Munk dahil)
    Ölü açıda tekne durup donmaz: geriye (sternway) ve rüzgar altına sürüklenir, pruva düşebilir.
    Dönüş: dizi [x, y, t, V, pruva, u, v, r(°/s)], ve 'stalled' (her zaman False; tekne sürüklenmeye devam eder)."""
    m, d_r = p.mass, math.radians(delta_r)
    Mu, Mv, Mr = m + p.m11, m + p.m22, p.Iz + p.Iz_added

    def deriv(st):
        x, y, h, u, v, r = st
        b = body_forces(u, v, r, h, tws_kt, d_r, trim_auto, trim_manual, sail_type, p)
        f, s = compass_unit(h), compass_unit(h + 90.0)
        vel = u * f + v * s
        return np.array([vel[0], vel[1], math.degrees(r),
                         (b["X"] + Mv * v * r) / Mu, (b["Y"] - Mu * u * r) / Mv, b["N"] / Mr])

    if init is None:
        eq = solve_equilibrium(heading, tws_kt, delta_r, trim_auto, trim_manual, sail_type, p)
        if eq["V"] > 0:
            lam = math.radians(eq["lam"])
            init = (eq["V"] * math.cos(lam), eq["V"] * math.sin(lam), math.radians(eq["r"]))
        else:
            init = (0.0, 0.0, 0.0)
    st = np.array([0.0, 0.0, float(heading), *init], dtype=float)
    out = [(0.0, 0.0, 0.0, math.hypot(st[3], st[4]), heading % 360.0, st[3], st[4], math.degrees(st[5]))]
    n = int(round(horizon / dt))
    rec = max(1, int(round(0.25 / dt)))
    for k in range(1, n + 1):
        k1 = deriv(st)
        k2 = deriv(st + 0.5 * dt * k1)
        k3 = deriv(st + 0.5 * dt * k2)
        k4 = deriv(st + dt * k3)
        st = st + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if k % rec == 0:
            out.append((st[0], st[1], k * dt, math.hypot(st[3], st[4]), st[2] % 360.0,
                        st[3], st[4], math.degrees(st[5])))
    return np.array(out), False


# ----------------------------------------------------------------------------
# Geometri (gövde ekseni [ileri, sancak], metre)
# ----------------------------------------------------------------------------
def to_world(pts_b, heading, scale=1.0, origin=(0.0, 0.0)):
    pts_b = np.atleast_2d(np.asarray(pts_b, float))
    f, s = compass_unit(heading), compass_unit(heading + 90.0)
    return np.asarray(origin) + scale * (pts_b[:, :1] * f + pts_b[:, 1:2] * s)


def hull_polygon(p, side):
    half = p.L / 2
    b = 0.09
    yo = side * p.hull_sep / 2
    xs = np.linspace(-half, half, 30)
    width = b * np.sqrt(np.clip(1 - ((xs - (-0.15 * half)) / (1.15 * half)) ** 4, 0, 1))
    width[-1] = 0.0
    top = np.c_[xs, yo + width]
    bot = np.c_[xs[::-1], yo - width[::-1]]
    return np.vstack([top, bot, top[:1]])


def naca_polygon(chord, delta_s, mast, t=0.12, n=40):
    xc = 0.5 * (1 - np.cos(np.linspace(0, math.pi, n)))
    yt = 5 * t * chord * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2
                          + 0.2843 * xc ** 3 - 0.1015 * xc ** 4)
    a = math.radians(180.0 - delta_s)
    c = np.array([math.cos(a), math.sin(a)])
    nrm = np.array([-c[1], c[0]])
    le = np.asarray(mast) - 0.25 * chord * c
    up = le + (xc * chord)[:, None] * c + yt[:, None] * nrm
    lo = le + (xc * chord)[:, None] * c - yt[:, None] * nrm
    return np.vstack([up, lo[::-1], up[:1]])


def soft_sail_curve(boom_len, delta_s, mast, flapping, aw_b):
    a = math.radians(180.0 - delta_s)
    c = np.array([math.cos(a), math.sin(a)])
    nrm = np.array([-c[1], c[0]])
    u = np.linspace(0, 1, 40)
    if flapping:
        off = 0.035 * np.sin(u * 6 * math.pi) * (u < 0.98)
    else:
        w = aw_b / (np.hypot(*aw_b) + 1e-12)
        bulge = 1.0 if float(nrm @ w) >= 0 else -1.0     # kumaş rüzgar altına şişer
        off = bulge * 0.09 * boom_len * 4 * u * (1 - u)
    return np.asarray(mast) + (u * boom_len)[:, None] * c + off[:, None] * nrm


# ----------------------------------------------------------------------------
# Çizim
# ----------------------------------------------------------------------------
def build_figure(heading, eq, path, stalled, sail_type, delta_r, p, lim=None):
    fig = go.Figure()
    extent = float(np.ptp(path[:, :2], axis=0).max()) if len(path) > 1 else 0.0
    sc = max(1.0, extent / 8.0)        # tekne ikonu büyütme
    O = (0.0, 0.0)

    # Ölü açı sektörü — KUVVET DENGESİNDEN (fiziksel sınır); Eq.(12) ±ζ yalnız kesikli referans
    R0 = 1.6 * sc
    sec = np.zeros((1, 2))
    if lim is not None and lim > 0:
        angs = np.linspace(WIND_FROM_DEG - lim, WIND_FROM_DEG + lim, 40)
        sec = np.vstack([[0, 0], np.array([R0 * compass_unit(a_) for a_ in angs]), [0, 0]])
        fig.add_trace(go.Scatter(x=sec[:, 0], y=sec[:, 1], fill="toself", mode="lines",
                                 line=dict(width=0), fillcolor="rgba(192,57,43,0.13)",
                                 name=f"Ölü açı ±{lim:.1f}° (kuvvet dengesi)", hoverinfo="name"))
    xs, ys = [], []
    for sgn in (-1, 1):
        e = 1.15 * R0 * compass_unit(WIND_FROM_DEG + sgn * p.zeta_deg)
        xs += [0, e[0], None]
        ys += [0, e[1], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#C0392B", width=1, dash="dash"),
                             name=f"Jaulin Eq.(12) referansı ±{p.zeta_deg:.0f}°", hoverinfo="name"))

    # Rota
    if len(path) > 1:
        fig.add_trace(go.Scatter(
            x=path[:, 0], y=path[:, 1], mode="lines", name="Tahmini rota",
            line=dict(color=COL_PATH, width=3, dash="dash"),
            customdata=np.c_[path[:, 2], path[:, 3] / KT2MS, path[:, 4], path[:, 5]],
            hovertemplate="t=%{customdata[0]:.1f} s<br>V=%{customdata[1]:.2f} kt"
                          "<br>pruva=%{customdata[2]:.0f}°<br>u (ileri)=%{customdata[3]:+.2f} m/s"
                          "<extra></extra>"))
        back = path[:, 5] < -0.02                      # geri giderken (sternway) bölümler
        if back.any():
            fig.add_trace(go.Scatter(x=np.where(back, path[:, 0], None), y=np.where(back, path[:, 1], None),
                                     mode="lines", name="Geriye sürüklenme (sternway)",
                                     line=dict(color="#C0392B", width=4), hoverinfo="skip"))
        # Rota üzerinde her ~2 s'de pruva işaretleri: rota ile pruva arasındaki fark = drift (yengeç yürüyüşü)
        tick = max(0.6, 0.06 * max(1.0, float(np.ptp(path[:, :2], axis=0).max())))
        xs, ys = [], []
        for k in range(8, len(path), 8):
            d_ = compass_unit(path[k, 4]) * tick
            xs += [path[k, 0] - d_[0], path[k, 0] + d_[0], None]
            ys += [path[k, 1] - d_[1], path[k, 1] + d_[1], None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Rotadaki pruva (2 s'de bir)",
                                     line=dict(color="#34495E", width=2), hoverinfo="skip"))
    if stalled:
        fig.add_trace(go.Scatter(x=[path[-1, 0]], y=[path[-1, 1]], mode="markers+text",
                                 marker=dict(symbol="x", size=12, color="#C0392B"),
                                 text=["Yol kesildi"], textposition="top center",
                                 name="Rota sonu: ileri hareket dengesi yok"))

    # Kirişler + merkez kıç desteği
    hs = p.hull_sep / 2
    for xb in (0.30, -0.45):
        w = to_world([[xb, -hs], [xb, hs]], heading, sc, O)
        fig.add_trace(go.Scatter(x=w[:, 0], y=w[:, 1], mode="lines", showlegend=False,
                                 line=dict(color="#7F8C8D", width=6), hoverinfo="skip"))
    xr = p.x_rud
    w = to_world([[-0.45, 0], [xr, 0]], heading, sc, O)
    fig.add_trace(go.Scatter(x=w[:, 0], y=w[:, 1], mode="lines", showlegend=False,
                             line=dict(color="#7F8C8D", width=4), hoverinfo="skip"))

    # Gövdeler: iskele (sol) TURUNCU, sancak (sağ) SİYAH
    for side, col, nm in ((-1, COL_PORT, "İskele gövde (sol)"), (1, COL_STBD, "Sancak gövde (sağ)")):
        w = to_world(hull_polygon(p, side), heading, sc, O)
        fig.add_trace(go.Scatter(x=w[:, 0], y=w[:, 1], fill="toself", mode="lines", name=nm,
                                 fillcolor=col, line=dict(color=col, width=1), hoverinfo="name"))

    # TEK merkezi dümen (kıç, simetri ekseninde). +δr → firar kenarı sancağa → tekne sancağa döner
    dr = math.radians(delta_r)
    blade = [[xr, 0.0], [xr - 0.22 * math.cos(dr), 0.22 * math.sin(dr)]]
    w = to_world(blade, heading, sc, O)
    fig.add_trace(go.Scatter(x=w[:, 0], y=w[:, 1], mode="lines+markers", name=f"Merkezi dümen ({delta_r:+.0f}°)",
                             line=dict(color=COL_RUDDER, width=7), marker=dict(size=[8, 0], color=COL_RUDDER)))

    # Direk + yelken
    mast = np.array([p.x_mast, 0.0])
    if sail_type == "wing":
        poly = naca_polygon(p.chord_wing, eq["delta_s"], mast)
        w = to_world(poly, heading, sc, O)
        fig.add_trace(go.Scatter(x=w[:, 0], y=w[:, 1], fill="toself", mode="lines",
                                 name=f"Sert kanat NACA 0012 (δs={eq['delta_s']:+.0f}°)",
                                 fillcolor="rgba(236,240,241,0.95)", line=dict(color="#34495E", width=2)))
    else:
        a = math.radians(180.0 - eq["delta_s"])
        boom = np.vstack([mast, mast + p.boom_len * np.array([math.cos(a), math.sin(a)])])
        w = to_world(boom, heading, sc, O)
        fig.add_trace(go.Scatter(x=w[:, 0], y=w[:, 1], mode="lines", name="Bumba",
                                 line=dict(color="#5D4037", width=4)))
        cur = soft_sail_curve(p.boom_len, eq["delta_s"], mast, eq["flapping"], eq["aw_b"])
        w = to_world(cur, heading, sc, O)
        lbl = ("Bez yelken — ÇIRPIYOR (luff, akışa hizalı)" if eq["flapping"] else
               f"Bez yelken — TERS DOLMUŞ (backing, δs={eq['delta_s']:+.0f}°)" if eq["backwinded"] else
               f"Bez yelken (δs={eq['delta_s']:+.0f}°)")
        fig.add_trace(go.Scatter(x=w[:, 0], y=w[:, 1], mode="lines", name=lbl,
                                 line=dict(color="#C0392B" if eq["flapping"] else "#8E44AD" if eq["backwinded"] else "#2C3E50",
                          width=3)))
    wm = to_world(mast, heading, sc, O)
    fig.add_trace(go.Scatter(x=wm[:, 0], y=wm[:, 1], mode="markers", name="Ana direk",
                             marker=dict(size=11, color="#2C3E50", line=dict(color="white", width=2))))

    # Pruva çizgisi ve gerçek hareket yönü (COG): aradaki açı = drift
    Lh = 1.9 * sc
    hd = to_world([[0, 0], [Lh / sc, 0]], heading, sc, O)
    fig.add_trace(go.Scatter(x=hd[:, 0], y=hd[:, 1], mode="lines", name=f"Pruva θ = {heading:.0f}°",
                             line=dict(color="#7F8C8D", width=2, dash="dot"), hoverinfo="name"))
    annotations = []
    if eq["V"] > 0:
        cg = compass_unit(eq["course"]) * Lh
        annotations.append(dict(x=cg[0], y=cg[1], ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                                showarrow=True, arrowhead=2, arrowwidth=2.5, arrowcolor=COL_PATH, text=""))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=COL_PATH, width=3),
                                 name=f"Hareket yönü φ (COG) = {eq['course'] % 360:.1f}°"))

    # Kuvvet diyagramı: her kuvvet GERÇEK uygulama noktasından çizilir (moment kolu görünür)
    ce = eq["ce"]
    forces = [(eq["Fx"], [1, 0], ce, COL_FX, "Fx yelken itişi"),
              (eq["Fy"], [0, 1], ce, COL_FY, "Fy yelken yanal"),
              (eq["Y_r"], [0, 1], [p.x_rud - 0.1, 0.0], COL_YR, "Y dümen"),
              (eq["Y_h"], [0, 1], [p.x_CLR, 0.0], COL_YH, "Y gövde")]
    Fref = max([abs(f_[0]) for f_ in forces] + [1e-6])
    Lmax = 1.3 * sc
    for comp, ub, pt, col, nm in forces:
        if abs(comp) < 0.02 * Fref or abs(comp) < 1e-3:
            continue
        tail = to_world([pt], heading, sc, O)[0]
        vec = to_world([[ub[0] * comp / Fref * Lmax, ub[1] * comp / Fref * Lmax]], heading, 1.0, O)[0]
        tip = tail + vec
        annotations.append(dict(x=tip[0], y=tip[1], ax=tail[0], ay=tail[1], xref="x", yref="y",
                                axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.1,
                                arrowwidth=3.5, arrowcolor=col, text=f"<b>{comp:+.1f} N</b>",
                                font=dict(color=col, size=11)))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=col, width=4), name=nm))
    cw = to_world([ce], heading, sc, O)
    fig.add_trace(go.Scatter(x=cw[:, 0], y=cw[:, 1], mode="markers", name="Yelken basınç merkezi (CE)",
                             marker=dict(size=9, color=COL_FY, symbol="diamond",
                                         line=dict(color="white", width=1))))

    # Eksen aralığı
    allp = [path[:, :2], sec, to_world(hull_polygon(p, -1), heading, sc, O),
            to_world(hull_polygon(p, 1), heading, sc, O)]
    for a in annotations:
        allp.append(np.array([[a["x"], a["y"]]]))
    allp = np.vstack(allp)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    ctr, half = (lo + hi) / 2, max(hi - lo) / 2 * 1.25 + 0.5
    xr_, yr_ = [ctr[0] - half, ctr[0] + half], [ctr[1] - half, ctr[1] + half]

    # Gerçek rüzgar oku (kuzeyden)
    wx, wy0 = xr_[0] + 0.12 * half, yr_[1] - 0.06 * half
    annotations.append(dict(x=wx, y=wy0 - 0.35 * half, ax=wx, ay=wy0, xref="x", yref="y",
                            axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=3,
                            arrowcolor=COL_WIND, text="", ))
    annotations.append(dict(x=wx, y=wy0, xref="x", yref="y", showarrow=False, yshift=12,
                            text="<b>Rüzgar 0° (K)</b>", font=dict(color=COL_WIND, size=12)))

    fig.update_layout(
        height=640, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#E8F4FA", annotations=annotations,
        title=dict(text=f"Kuşbakışı — pruva {heading:.0f}° | tekne ikonu ×{sc:.1f} büyütülmüş", x=0.01,
                   font=dict(size=14)),
        legend=dict(orientation="h", yanchor="top", y=-0.06, x=0, font=dict(size=11)),
        xaxis=dict(range=xr_, title="Doğu (m)", zeroline=False, gridcolor="white"),
        yaxis=dict(range=yr_, title="Kuzey (m)", zeroline=False, gridcolor="white",
                   scaleanchor="x", scaleratio=1),
    )
    return fig


# ----------------------------------------------------------------------------
# Arayüz
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Katamaran — Modül 1", layout="wide")
    st.markdown("""
        <style>
        [data-testid="stMetric"] {background:#F7F9FB; border:1px solid #E1E5EA;
            border-radius:10px; padding:10px 14px; overflow:hidden;}
        [data-testid="stMetricValue"] {font-size:1.55rem; white-space:nowrap;}
        [data-testid="stMetricLabel"] p {white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
        </style>""", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Girdiler")
        sail_lbl = st.radio("Yelken tipi", ["Sert Kanat (NACA Wingsail)", "Bez Yelken (Soft Sail)"])
        sail_type = "wing" if sail_lbl.startswith("Sert") else "soft"
        heading = st.slider("Pruva açısı θ (°)", 0, 360, 90, 1)
        trim_mode = st.radio("Yelken trim modu", ["Optimum — kuvvet dengesi (maks. itiş)",
                                                  "Otomatik — Jaulin Eq.(11) kardioid", "Manuel"])
        trim_auto = "opt" if trim_mode.startswith("Optimum") else trim_mode.startswith("Otomatik")
        boom_lbl = st.radio("Bez yelken bumba tahriki", ["Iskota (ip — yalnız çeker)", "Rijit servo (iki yönlü)"],
                            disabled=sail_type != "soft",
                            help="Iskota: bumba rüzgar altına serbest savrulur, trim yalnız en dış açıyı sınırlar "
                                 "(Jaulin 2013). Rijit: bumba tam komut açısında tutulur; rüzgar arkadan "
                                 "vurursa yelken TERS DOLAR (backing) ve gerçek kuvvet üretir.")
        trim_manual = st.slider("Yelken açısı / trim δs (°)  (+ = bumba sancakta; ıskotada yalnız |δs| sınırdır)",
                                -90, 90, 45, 1, disabled=trim_mode != "Manuel")
        delta_r = st.slider("Merkezi dümen δr (°)  (+ = sancağa dönüş)", -35, 35, 0, 1)
        tws_kt = st.slider("Rüzgar hızı (knot) — yön 0° Kuzey", 2, 30, 10, 1)
        horizon = st.slider("Rota tahmin süresi (s)", 5, 60, 20, 1)

        with st.expander("Model parametreleri — VARSAYIM (makalelerde verilmiyor)"):
            d = Params()
            p = replace(
                d,
                L=st.number_input("Boy L (m)", 0.5, 5.0, d.L, 0.05),
                mass=st.number_input("Kütle (kg)", 1.0, 200.0, d.mass, 1.0),
                hull_sep=st.number_input("Gövde aralığı (m)", 0.2, 2.0, d.hull_sep, 0.05),
                zeta_deg=st.number_input("Eq.12 referans ζ (°) — yalnız karşılaştırma çizgisi", 20.0, 70.0, d.zeta_deg, 1.0),
                CDA_windage=st.number_input("Su üstü rüzgar direnci CD·A (m²)", 0.0, 1.0, d.CDA_windage, 0.01),
                hull_B=st.number_input("Tek gövde genişliği B (m)", 0.05, 0.6, d.hull_B, 0.01),
                Cb=st.number_input("Blok katsayısı Cb", 0.2, 0.8, d.Cb, 0.01),
                C_R=st.number_input("Artık direnç katsayısı C_R", 0.0, 0.05, d.C_R, 0.001, format="%.3f"),
                A_wing=st.number_input("Kanat alanı (m²)", 0.05, 3.0, d.A_wing, 0.01),
                A_soft=st.number_input("Bez yelken alanı (m²)", 0.05, 3.0, d.A_soft, 0.01),
                h_CE=st.number_input("Yelken basınç merkezi kolu (m)", 0.1, 3.0, d.h_CE, 0.05),
                CLa_hull=st.number_input("Gövde yanal eğim (1/rad)", 0.2, 6.0, d.CLa_hull, 0.1),
                x_mast=st.number_input("Direk konumu x (m, merkezin önü +)", -0.8, 0.8, d.x_mast, 0.01),
                x_CLR=st.number_input("Gövde yanal direnç merkezi x (m)", -0.8, 0.8, d.x_CLR, 0.01),
                A_rud=st.number_input("Dümen alanı (m²)", 0.005, 0.2, d.A_rud, 0.005, format="%.3f"),
                k_munk=st.number_input("Munk momenti katsayısı (1 = teorik)", 0.0, 1.5, d.k_munk, 0.05,
                                       help="0: ihmal, 1: potansiyel akış teorisi. Prototipte dönme testiyle kalibre edin."),
                boom_rigid=boom_lbl.startswith("Rijit"),
            )
            st.caption(f"Kütleden türetilen: su çekimi T = {p.draft * 100:.1f} cm, ıslak alan "
                       f"S = {p.S_wet:.2f} m², yanal alan = {p.A_lat:.2f} m², "
                       f"maks. doğrultucu moment = {p.righting_moment:.0f} N·m")

    eq = solve_equilibrium(heading, tws_kt, delta_r, trim_auto, trim_manual, sail_type, p)
    path, stalled = predict_path(heading, tws_kt, delta_r, trim_auto, trim_manual, sail_type, p, horizon)
    lim = nogo_limit(tws_kt, sail_type, p)              # fiziksel ölü açı sınırı (kuvvet dengesi)
    status, scol, twa = point_of_sail(heading, eq, lim)

    st.title("Modül 1 — Statik Kuvvet Dengesi & Rota Tahmini")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tahmini Hız", f"{eq['V'] / KT2MS:.2f} kt")
    c2.metric("İleri İtiş Fx", f"{eq['Fx']:+.1f} N")
    c3.metric("Yanal Fy", f"{eq['Fy']:+.1f} N",
              help="+ sancağa, − iskeleye doğru (gövde ekseni)")
    c4.metric("Drift (Sapma)", "—" if eq["lam"] is None else f"{eq['lam']:+.1f}°",
              help="Rota φ − pruva θ ([M22] Eq.2'deki φ≠θ). Tekne dururken tanımsız.")

    extra = ""
    if status.startswith("ÖLÜ"):
        extra = (f" — Kuvvet dengesi: bu pruvada itiş, direnci hiçbir trimle yenemiyor "
                 f"(fiziksel sınır ±{lim:.1f}°)." if lim is not None else " — Bu rüzgarda hiçbir pruvada yol yok.")
    elif status.startswith("YOL YOK"):
        extra = " — Optimum trimle bu pruvada yol alınabilir; mevcut trim yetersiz."
    elif eq["flapping"]:
        extra = " — Bez yelken ÇIRPIYOR (luff): rüzgar ıskota sınırından daha önden geliyor."
    elif eq["backwinded"]:
        extra = " — Bez yelken TERS DOLMUŞ (backing): kuvvet tekneyi geri/yana itiyor."
    st.markdown(
        f"<div style='display:inline-block;padding:6px 16px;border-radius:20px;background:{scol};"
        f"color:white;font-weight:700;letter-spacing:.5px'>{status}</div>"
        f"<span style='margin-left:10px;color:#555'>Gerçek rüzgar açısı TWA = {twa:.0f}°{extra}</span>",
        unsafe_allow_html=True)

    if eq["capsize"]:
        st.error(f"DEVRİLME RİSKİ: yatırma momenti {eq['HM']:.0f} N·m > doğrultucu moment "
                 f"{eq['RM']:.0f} N·m (m·g·b/2). Rüzgarüstü gövde sudan kalkar; bu durumda statik "
                 "sonuçlar geçersizdir — yelkeni boşaltın / alanı küçültün.")
    elif eq["HM"] > 0.7 * eq["RM"]:
        st.warning(f"Yatırma momenti doğrultucu momentin %{100 * eq['HM'] / eq['RM']:.0f}'i — "
                   "rüzgarüstü gövde hafiflemiş.")
    if eq["Fr"] > 0.40:
        st.warning(f"Froude sayısı Fr = {eq['Fr']:.2f} > 0.40: dalga direnci hızla artar; sabit C_R "
                   "kabulü bu bölgede direnci düşük tahmin eder → hız İYİMSERDİR.")

    if eq.get("crab"):
        st.error("Yalnızca yana sürüklenme dengesi var: tekne çok yavaş ve büyük drift açısıyla yan yan "
                 f"gidiyor; dümene gelen akış dümen stall açısını (±{p.rud_stall:.0f}°) aşıyor → yön tutulamaz, "
                 "kontrollü seyir YOK.")
    elif eq.get("no_eq"):
        st.error("İleri hareket dengesi yok: yelken itişi, bu pruvada yanal kaymanın yarattığı direnci "
                 "yenemiyor → tekne yol alamaz, rüzgar altına sürüklenir.")
    elif eq["V"] > 0 and (eq["V"] < 0.3 or abs(eq["lam"]) > 20):
        st.warning(f"Dümen hakimiyeti zayıf: V = {eq['V']:.2f} m/s, drift = {eq['lam']:+.0f}°. Dümene gelen akışı "
                   f"yana kayma belirliyor (α_dümen = {eq['alpha_r']:+.0f}°); dümen komutu beklenen yönü vermeyebilir.")

    # ---------------- Yönelim telemetrisi ----------------
    st.subheader("Yönelim telemetrisi")
    eq0 = eq if delta_r == 0 else solve_equilibrium(heading, tws_kt, 0, trim_auto, trim_manual, sail_type, p)
    rel_wind = wrap180(WIND_FROM_DEG - heading)          # + : rüzgar sancaktan
    if eq["V"] > 0 and abs(eq["r"]) > 0.05:
        turn_txt = "Sancağa ↻" if eq["r"] > 0 else "İskeleye ↺"
        turn_txt += " (orsa)" if np.sign(eq["r"]) == np.sign(rel_wind) else " (boca)"
    else:
        turn_txt = "Düz"
    end_h = path[-1, 4] if len(path) > 1 else heading
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Hareket yönü φ", "—" if eq["V"] == 0 else f"{eq['course'] % 360:.1f}°",
              help="Teknenin suya göre gerçekte gittiği yön (COG) = pruva + drift")
    t2.metric("Dönüş hızı r", f"{eq['r']:+.2f} °/s", help="+ sancağa (saat yönü). " + turn_txt)
    t3.metric("Dönüş yarıçapı", "∞ (düz)" if not math.isfinite(eq["radius"]) else f"{eq['radius']:.1f} m",
              help="R = V / r (durağan dönüş)")
    t4.metric("Düz rota dümeni", "—" if eq["d_bal"] is None else f"{eq['d_bal']:+.1f}°",
              help="r = 0 olması için gereken merkezi dümen açısı (yelken momentini dengeler)")
    t5.metric(f"{horizon} s sonra pruva", f"{end_h:.0f}°",
              help=f"Net pruva değişimi {wrap180(end_h - heading):+.0f}°")
    st.caption((f"Fiziksel ölü açı sınırı (kuvvet dengesi, optimum trim): **±{lim:.1f}°** — "
                f"Jaulin Eq.(12) referansı ±{p.zeta_deg:.0f}°.  " if lim is not None else
                "Bu rüzgarda hiçbir pruvada ileri denge yok.  ")
               + f"Dönüş yönü: **{turn_txt}**.  "
               + ("" if eq0["V"] == 0 or abs(eq0["r"]) < 0.05 else
                  f"Dümen 0°'de tekne kendiliğinden **{'sancağa' if eq0['r'] > 0 else 'iskeleye'}** "
                  f"döner ({eq0['r']:+.2f} °/s) → "
                  + ("**ORSA eğilimi** (weather helm)" if np.sign(eq0["r"]) == np.sign(rel_wind)
                     else "**BOCA eğilimi** (lee helm)")
                  + ": yelken basınç merkezi ile yanal direnç merkezi dengede değil."))

    if eq["V"] == 0 and len(path) > 1:               # ölü açıdaki teknenin zaman içindeki davranışı
        twa_t = np.abs(((path[:, 4] - WIND_FROM_DEG + 180) % 360) - 180)
        esc = np.where((twa_t > (lim or 180)) & (path[:, 5] > 0.05))[0]
        umin = path[:, 5].min()
        drift_m = math.hypot(path[-1, 0], path[-1, 1])
        msg = (f"**Ölü açıdaki davranış (dinamik):** tekne {'geriye (sternway) en fazla ' + format(-umin, '.2f') + ' m/s hızla ' if umin < -0.02 else ''}"
               f"rüzgar altına sürükleniyor; {horizon} s'de {drift_m:.0f} m yer değiştiriyor. ")
        if len(esc):
            msg += (f"Pruva **{path[esc[0], 2]:.1f} s'de düşüyor** (TWA {twa_t[esc[0]]:.0f}°) ve tekne yeniden yol alıyor.")
        else:
            msg += (f"Pruva düşmüyor: {horizon} s sonunda TWA {twa_t[-1]:.0f}° — tekne ölü açıda **kalıyor** "
                    "(yelken basınç merkezi yanal direnç merkezinin gerisinde → rüzgara dönme / weathervane). "
                    "Kurtulmak için dümen veya trim manevrası gerekir; ya da direk/CLR konumunu kontrol edin.")
        st.info(msg)

    fig = build_figure(heading, eq, path, stalled, sail_type, delta_r, p, lim)
    try:
        st.plotly_chart(fig, width="stretch")          # Streamlit ≥ 1.50
    except Exception:
        st.plotly_chart(fig, use_container_width=True)  # eski sürümler

    a, b = st.columns(2)
    with a:
        st.markdown("**Denge durumu**")
        st.write(
            f"- Görünür rüzgar: **{eq['aaw'] / KT2MS:.1f} kt**, pruvadan **{eq['beta']:+.0f}°** "
            f"(+ sancak) — [M22] Eq.(2)\n"
            f"- Kardioid δs_max: **{eq['dsmax']:.0f}°** — [J13] Eq.(11)\n"
            f"- Uygulanan yelken açısı δs: **{eq['delta_s']:+.0f}°**, hücum açısı α: **{eq['alpha']:.0f}°**\n"
            f"- İleri denge: yelken Fx **{eq['Fx']:+.1f} N** ↔ toplam direnç (gövde+dümen+indüklenmiş) **{eq['R']:.1f} N**\n"
            f"- Rota φ = θ + drift: **{eq['course'] % 360:.1f}°**, Froude Fr = **{eq['Fr']:.2f}**\n"
            f"- Yatırma / doğrultucu moment: **{eq['HM']:.0f} / {eq['RM']:.0f} N·m**")
        if eq["V"] > 0:
            mur = (p.mass + p.m11) * eq["V"] * math.cos(math.radians(eq["lam"])) * math.radians(eq["r"])
            st.markdown("**Kuvvet / moment diyagramı (dengede toplam ≈ 0)**")
            st.table({
                "Bileşen": ["Yelken + rüzgar direnci (CE)", "Gövde (şerit teorisi)", "Merkezi dümen",
                            "Munk momenti", "Merkezcil −(m+m11)·u·r", "TOPLAM"],
                "Yanal Y (N)": [f"{eq['Fy']:+.2f}", f"{eq['Y_h']:+.2f}", f"{eq['Y_r']:+.2f}", "—", f"{-mur:+.2f}",
                                f"{eq['Fy'] + eq['Y_h'] + eq['Y_r'] - mur:+.3f}"],
                "Moment N (N·m)": [f"{eq['N_s']:+.3f}", f"{eq['N_h']:+.3f}", f"{eq['N_r']:+.3f}",
                                   f"{eq['N_munk']:+.3f}", "—",
                                   f"{eq['N_s'] + eq['N_h'] + eq['N_r'] + eq['N_munk']:+.4f}"],
            })
    with b:
        st.markdown("**Kaynak & varsayımlar**")
        st.caption(
            "Makalelerden: görünür rüzgar [M22] Eq.(1)-(2); kardioid trim ve dümen işareti [J13] Eq.(11); "
            "[J13] Eq.(12) ζ=45° yalnız referans çizgisi — ölü açı kuvvet dengesinden hesaplanır. Standart gemi mühendisliği: 3-DOF rijit gövde denge "
            "denklemleri (X, Y, N; durağan hal, ek kütle ihmal), şerit teorisi ile gövde sönümü, "
            "ITTC-1957 sürtünme çizgisi, Mumford ıslak alan formülü, m·g·b/2 doğrultucu moment. "
            "VARSAYIM (makalelerde yok): CL/CD eğrileri, form faktörü, C_R, gövde yanal eğimi ve "
            "çapraz akış katsayısı, dümen alanı, DİREK ve YANAL DİRENÇ MERKEZİ konumları (yönelimi en "
            "çok bunlar belirler — prototipten ölçüp girin). Kartlar durağan dengeden; ROTA ise 3-DOF hareket "
            "denklemlerinin RK4 entegrasyonundan (kütle, Iz, ek kütle m22 ve Iz_ek şerit teorisinden; m11 ve "
            "Munk momenti ihmali varsayım). Ölü açıda tekne donmaz: geriye ve rüzgar altına sürüklenir.")


if __name__ == "__main__":
    main()
