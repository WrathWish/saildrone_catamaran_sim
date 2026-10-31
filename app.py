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

import json
import math
import os
from dataclasses import astuple, dataclass, replace
from functools import lru_cache

import numpy as np
import pandas as pd  # plotly'den önce: Streamlit soğuk başlangıçta pandas'ı başka iş parçacığında yüklerken
#                      plotly yarım yüklenmiş modülü görüp "pd.Series yok" hatası veriyordu; import burada bekler
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ----------------------------------------------------------------------------
# Yeniden çalıştırmada kalıcı önbellek (yalnız performans; hiçbir sonucu değiştirmez)
#   Streamlit her girdide bu dosyayı baştan çalıştırır; burada tanımlı @lru_cache fonksiyonları her seferinde
#   YENİDEN tanımlandığı için önbellekleri boşalıyordu (ölü açı sınırı ve trim tablosu her girdide yeniden
#   hesaplanıyordu). persistent_lru aynı lru_cache sarmalayıcısını oturumlar ve yeniden çalıştırmalar
#   arasında saklar. Anahtar dosya içeriğinin özetini içerir: kod değişirse eski sonuç kullanılmaz.
# ----------------------------------------------------------------------------
import hashlib as _hashlib

try:
    with open(__file__, "rb") as _fh:
        _APP_HASH = _hashlib.sha1(_fh.read()).hexdigest()
except OSError:                                   # dosya okunamazsa kalıcılık kapalı (davranış eskisi gibi)
    _APP_HASH = None


@st.cache_resource(show_spinner=False)
def _persistent_store():
    import threading
    return {"lock": threading.Lock(), "tables": {}}


_MISSING = object()


def _canon(a):
    """Önbellek anahtarı: dataclass (Params) her çalıştırmada yeniden tanımlandığından sınıf kimliğiyle değil
    alan DEĞERLERİYLE anahtarlanır."""
    fl = getattr(a, "__dataclass_fields__", None)
    return (type(a).__name__,) + tuple(getattr(a, f) for f in fl) if fl is not None else a


def persistent_lru(maxsize):
    """İki katman: (1) bu çalıştırmanın lru_cache'i (sıcak yol, değişmedi); (2) kaçırırsa kalıcı tablo."""
    def deco(fn):
        if _APP_HASH is None:
            return lru_cache(maxsize=maxsize)(fn)
        try:
            ps = _persistent_store()
        except Exception:                         # Streamlit dışında / önbellek kullanılamıyorsa
            return lru_cache(maxsize=maxsize)(fn)
        lock = ps["lock"]
        with lock:
            table = ps["tables"].setdefault((_APP_HASH, fn.__qualname__), {})

        def slow(*args):
            key = tuple(_canon(a) for a in args)
            val = table.get(key, _MISSING)
            if val is _MISSING:
                val = fn(*args)
                with lock:
                    while len(table) >= 4 * maxsize:
                        table.pop(next(iter(table)))
                    table[key] = val
            return val

        slow.__name__, slow.__qualname__, slow.__doc__ = fn.__name__, fn.__qualname__, fn.__doc__
        return lru_cache(maxsize=maxsize)(slow)
    return deco


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
    k_munk: float = 1.0        # Munk momenti katsayısı — 1'de SABİT (menüden kaldırıldı). Munk momenti ideal akış
    #                            etkisidir; ≠1 yapmak ek kütle Coriolis terimleriyle dengeyi bozup enerji üretir/yok
    #                            eder. Viskoz etki ayrı bir kıç kuvvetidir (MIT 2.154 ders 8) → x_CLR ile ayarlanır.
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
    # Seyir adları (yalnız etiket; fiziğe girmez). Kanonik açılar: orsa ≈45°, apaz 90°, geniş apaz 135°,
    # pupa 180° (Wikipedia "Point of sail"; Boğaziçi Yelken "Seyirler ve Manevralar"; TÜBİTAK Bilim ve Teknik).
    # Sınırlar komşu kanonik açıların orta noktaları: 55° / 80° / 110° / 155°.
    if twa < 55:
        return "ORSA", "#E67E22", twa
    if twa < 80:
        return "DAR APAZ", "#EAB308", twa
    if twa < 110:
        return "APAZ", "#27AE60", twa
    if twa < 155:
        return "GENİŞ APAZ", "#14B8A6", twa
    return "PUPA", "#2980B9", twa


@persistent_lru(maxsize=32)
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
    # YALNIZ ETİKET (kuvvetleri etkilemez; sail_coeffs bu bayrağı kullanmaz): rüzgar, bumbanın açıldığı tarafın
    # DIŞ yüzüne çarpıyorsa kumaş ters dolmuştur. Bumba rüzgarüstündeyken de, rüzgaraltında ama görünür rüzgar
    # açısından fazla açılmışken de (rijit bumba) geçerli. Iskotada bu durum çırpmaya dönüştüğü için oluşmaz.
    if flapping:
        backwinded = False                          # akışa hizalı çırpan yelken ters dolmuş sayılmaz
    elif sail_type == "soft" and abs(delta_s) > 1.0:
        backwinded = bool(cross * delta_s < 0.0)
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


@persistent_lru(maxsize=64)
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


def nogo_trap(path, tws_kt, delta_r, trim_auto, trim_manual, sail_type, p, lim, extra=60.0, hold=20.0):
    """Rota ölü açıda bitiyorsa tekne oradan kendiliğinden çıkabiliyor mu? Fizik DEĞİŞMEZ: aynı dinamik
    model rota sonundaki durumdan aynı dümen/trimle `extra` s daha entegre edilir. Son `hold` s boyunca
    hâlâ ölü açıdaysa tekne TAKILDI sayılır (ileri hız çıkmaya yetmiyor).
    Dönüş: ölü açıya son giriş zamanı (s), ya da None (rota ölü açıda bitmiyor / tekne çıkabiliyor)."""
    if lim is None or len(path) < 2:
        return None
    twa = np.abs(wrap180(path[:, 4] - WIND_FROM_DEG))
    if twa[-1] >= lim:
        return None
    _, _, _, _, h, u, v, r = path[-1]
    ext, _ = predict_path(h, tws_kt, delta_r, trim_auto, trim_manual, sail_type, p, extra,
                          init=(u, v, math.radians(r)))
    tail = ext[ext[:, 2] >= extra - hold]
    if (np.abs(wrap180(tail[:, 4] - WIND_FROM_DEG)) >= lim).any():
        return None
    out = np.where(twa >= lim)[0]
    return float(path[out[-1] + 1, 2]) if len(out) else 0.0


# ----------------------------------------------------------------------------
# A→B düz rota takibi: kaskad PID otopilot.
#   FİZİK DEĞİŞMEZ — hareket denklemleri predict_path ile birebir aynıdır (body_forces + 3-DOF, RK4,
#   dt = 0.025 s). Tek fark: dümen açısı sabit değil, her adımda otopilottan gelir (ZOH).
#   Dış döngü:  sapma e (rotanın sancağı +) → pruva düzeltmesi Δψ = −(Kp·e + Ki·∫e + Kd·ė), |Δψ| ≤ Δψmax
#               hedef pruva ψ* = χ + Δψ − λ   (χ: A→B rota açısı, λ: yan kayma; ileri besleme)
#               ψ* ölü açıya düşerse ölü açı sınırına (+3°) çekilir (tramola bu sürümde yok).
#   İç döngü:   pruva hatası ε = ψ* − ψ → dümen δr = g·(Kp·ε + Ki·∫ε − Kd·r), |δr| ≤ δr_max,
#               servo hız sınırı; g = (V_ref/V)² hıza göre kazanç (dümen etkinliği ∝ V²).
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Autopilot:
    kp_e: float = 6.0           # °/m       dış döngü P
    ki_e: float = 0.01          # °/(m·s)   dış döngü I (yalnız |e| < 5 m iken biriktirir)
    kd_e: float = 10.0          # °/(m/s)   dış döngü D (ė = yere göre hızın rotaya dik bileşeni)
    dpsi_max: float = 45.0      # °         en büyük yaklaşma açısı
    kp_h: float = 1.0           # °/°       iç döngü P
    ki_h: float = 0.05          # °/(°·s)   iç döngü I
    kd_h: float = 0.8           # °/(°/s)   iç döngü D (dönüş hızı r üzerinden)
    rud_max: float = 35.0       # °         dümen sınırı (kokpit kaydırıcısıyla aynı)
    rud_rate: float = 90.0      # °/s       dümen servosu hız sınırı
    leeway_ff: bool = True      # ileri besleme: yan kayma telafisi
    gain_sched: bool = True     # hıza göre iç döngü kazancı


def _route_deriv(st, d_r, tws_kt, trim_auto, trim_manual, sail_type, p):
    """predict_path içindeki deriv ile aynı 3-DOF hareket denklemleri; dümen açısı (rad) dışarıdan verilir."""
    m = p.mass
    Mu, Mv, Mr = m + p.m11, m + p.m22, p.Iz + p.Iz_added
    x, y, h, u, v, r = st
    b = body_forces(u, v, r, h, tws_kt, d_r, trim_auto, trim_manual, sail_type, p)
    f, s = compass_unit(h), compass_unit(h + 90.0)
    vel = u * f + v * s
    return np.array([vel[0], vel[1], math.degrees(r),
                     (b["X"] + Mv * v * r) / Mu, (b["Y"] - Mu * u * r) / Mv, b["N"] / Mr]), b


def route_start(A, B, e0, head_err, tws_kt, trim_auto, trim_manual, sail_type, p):
    """Rota geometrisi ve başlangıç: A'dan rotaya dik e0 m kaymış nokta, pruva χ + head_err, hız o pruvadaki
    durağan denge. V_ref: rota pruvasındaki durağan hız (iç döngü kazanç ölçeği)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    Lr = float(np.hypot(*(B - A)))
    chi = math.degrees(math.atan2(B[0] - A[0], B[1] - A[1])) % 360.0
    t_hat, n_hat = compass_unit(chi), compass_unit(chi + 90.0)          # n: rotanın sancak tarafı (+e)
    lim = nogo_limit(tws_kt, sail_type, p)
    h0 = (chi + head_err) % 360.0
    P0 = A + e0 * n_hat
    eq0 = solve_equilibrium(h0, tws_kt, 0, trim_auto, trim_manual, sail_type, p)
    u0 = v0 = r0 = 0.0
    if eq0["V"] > 0:
        lam0 = math.radians(eq0["lam"])
        u0, v0, r0 = eq0["V"] * math.cos(lam0), eq0["V"] * math.sin(lam0), math.radians(eq0["r"])
    eq_t = solve_equilibrium(chi, tws_kt, 0, trim_auto, trim_manual, sail_type, p)
    V_ref = eq_t["V"] if eq_t["V"] > 0 else max(eq0["V"], 0.5)
    track_nogo = lim is not None and abs(wrap180(chi - WIND_FROM_DEG)) < lim
    return dict(A=A, B=B, L=Lr, chi=chi, t_hat=t_hat, n_hat=n_hat, lim=lim, h0=h0, P0=P0, V_ref=V_ref,
                track_nogo=track_nogo, st0=(float(P0[0]), float(P0[1]), h0, u0, v0, r0))


def simulate_route(A, B, e0, head_err, tws_kt, trim_auto, trim_manual, sail_type, p, ap,
                   dt=0.025, rec_dt=0.25, t_max=None):
    """A→B doğrusunu kaskad PID ile takip eden teknenin dinamik benzetimi.
    Başlangıç: A'dan rotaya dik e0 m kaymış nokta, pruva = χ + head_err, hız = o pruvadaki durağan denge.
    Bitiş: rota boyunca B hizası geçildiğinde, tekne 30 s durursa ya da süre dolunca."""
    rs = route_start(A, B, e0, head_err, tws_kt, trim_auto, trim_manual, sail_type, p)
    A, B, Lr, chi, t_hat, n_hat = rs["A"], rs["B"], rs["L"], rs["chi"], rs["t_hat"], rs["n_hat"]
    lim, h0, P0, V_ref, track_nogo = rs["lim"], rs["h0"], rs["P0"], rs["V_ref"], rs["track_nogo"]
    if t_max is None:
        t_max = min(1200.0, 60.0 + 3.0 * (Lr + abs(e0)) / max(V_ref, 0.3))

    st = np.array(rs["st0"], dtype=float)
    I_e = I_h = 0.0
    dr = 0.0                                                           # gerçek dümen açısı (°)
    every = max(1, int(round(rec_dt / dt)))
    rec = {k: [] for k in ("t", "x", "y", "h", "V", "u", "v", "r", "e", "s", "psi_star", "o_P", "o_I", "o_D",
                           "o_FF", "dpsi", "clamp", "eps", "i_P", "i_I", "i_D", "g", "cmd", "dr", "ds", "alpha",
                           "beta", "aaw", "Fx", "Fy", "X_r", "Y_r", "Y_h", "R", "N_s", "N_h", "N_r", "N_munk",
                           "lam", "cog", "twa", "flap")}
    status, slow_t = "time", 0.0
    for k in range(int(round(t_max / dt)) + 1):
        t = k * dt
        x, y, h, u, v, r = st
        rel = np.array([x, y]) - A
        e, s_al = float(rel @ n_hat), float(rel @ t_hat)
        f, sv = compass_unit(h), compass_unit(h + 90.0)
        vel = u * f + v * sv
        V = math.hypot(u, v)
        lam = math.degrees(math.atan2(v, u)) if V > 1e-3 else 0.0

        # --- Dış döngü: sapma → hedef pruva ---
        e_dot = float(vel @ n_hat)
        o_P, o_D = -ap.kp_e * e, -ap.kd_e * e_dot
        o_I = -ap.ki_e * I_e
        dpsi_raw = o_P + o_I + o_D
        dpsi = max(-ap.dpsi_max, min(ap.dpsi_max, dpsi_raw))
        if abs(e) < 5.0 and dpsi == dpsi_raw and ap.ki_e > 0:         # anti-windup: yalnız rotaya yakınken
            I_e = max(-10.0 / ap.ki_e, min(10.0 / ap.ki_e, I_e + e * dt))
        o_FF = -lam if (ap.leeway_ff and V > 0.1) else 0.0
        psi_star = chi + dpsi + o_FF
        clamp = False
        twa_star = wrap180(psi_star - WIND_FROM_DEG)
        if lim is not None and abs(twa_star) < lim + 3.0:              # ölü açıya hedef verilmez
            side = math.copysign(1.0, twa_star) if abs(twa_star) > 1e-6 else \
                math.copysign(1.0, wrap180(h - WIND_FROM_DEG) or 1.0)
            psi_star, clamp = WIND_FROM_DEG + side * (lim + 3.0), True

        # --- İç döngü: pruva hatası → dümen ---
        eps = wrap180(psi_star - h)
        g = min(3.0, max(0.5, (V_ref / max(V, 0.2)) ** 2)) if ap.gain_sched else 1.0
        i_P, i_D = g * ap.kp_h * eps, -g * ap.kd_h * math.degrees(r)
        i_I = g * ap.ki_h * I_h
        cmd_raw = i_P + i_I + i_D
        cmd = max(-ap.rud_max, min(ap.rud_max, cmd_raw))
        if cmd == cmd_raw and ap.ki_h > 0:
            I_h = max(-10.0 / ap.ki_h, min(10.0 / ap.ki_h, I_h + eps * dt))
        dr += max(-ap.rud_rate * dt, min(ap.rud_rate * dt, cmd - dr))  # servo

        # --- Fizik (değişmedi): RK4, dümen adım boyunca sabit ---
        d_r = math.radians(dr)
        k1, b = _route_deriv(st, d_r, tws_kt, trim_auto, trim_manual, sail_type, p)
        if k % every == 0:
            sf = b["sf"]
            for key, val in (("t", t), ("x", x), ("y", y), ("h", h % 360.0), ("V", V), ("u", u), ("v", v),
                             ("r", math.degrees(r)), ("e", e), ("s", s_al), ("psi_star", psi_star % 360.0),
                             ("o_P", o_P), ("o_I", o_I), ("o_D", o_D), ("o_FF", o_FF), ("dpsi", dpsi),
                             ("clamp", clamp), ("eps", eps), ("i_P", i_P), ("i_I", i_I), ("i_D", i_D), ("g", g),
                             ("cmd", cmd), ("dr", dr), ("ds", b["delta_s"]), ("alpha", sf["alpha"]),
                             ("beta", b["beta"]), ("aaw", sf["aaw"]), ("Fx", b["Fx"]), ("Fy", b["Fy"]),
                             ("X_r", b["X_r"]), ("Y_r", b["Y_r"]), ("Y_h", b["Y_h"]), ("R", b["R"]),
                             ("N_s", b["N_s"]), ("N_h", b["N_h"]), ("N_r", b["N_r"]), ("N_munk", b["N_munk"]),
                             ("lam", lam), ("cog", (h + lam) % 360.0), ("twa", wrap180(h - WIND_FROM_DEG)),
                             ("flap", bool(sf["flapping"] or sf["streaming"]))):
                rec[key].append(val)
        if s_al >= Lr:
            status = "reached"
            break
        slow_t = slow_t + dt if V < 0.05 else 0.0
        if slow_t > 30.0:
            status = "stalled"
            break
        if track_nogo and abs(e) > abs(e0) + 40.0:                     # rota ölü açıda: tramola yok
            status = "nogo"
            break
        k2, _ = _route_deriv(st + 0.5 * dt * k1, d_r, tws_kt, trim_auto, trim_manual, sail_type, p)
        k3, _ = _route_deriv(st + 0.5 * dt * k2, d_r, tws_kt, trim_auto, trim_manual, sail_type, p)
        k4, _ = _route_deriv(st + dt * k3, d_r, tws_kt, trim_auto, trim_manual, sail_type, p)
        st = st + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    out = {k_: np.asarray(v_) for k_, v_ in rec.items()}
    out.update(A=A, B=B, P0=P0, h0=h0, chi=chi, L=Lr, lim=lim, V_ref=V_ref, status=status,
               t_end=float(out["t"][-1]), track_nogo=track_nogo, t_max=t_max)
    return out


# ----------------------------------------------------------------------------
# Rota gözlemi: evreler, özet ve dışa aktarım (yalnız kayıtlı seriden; benzetime dokunmaz)
# ----------------------------------------------------------------------------
PHASES = ("Yaklaşma", "Oturma", "Rotada")
SETTLE_BAND = 1.0          # m: |e| bundan sonra hep bu bandın içindeyse tekne rotaya oturmuştur


def route_phases(res, ap, band=SETTLE_BAND):
    """Yaklaşma: dış döngü doymuş (|Δψ| = Δψmax), tekne rotaya en büyük açıyla gidiyor.
    Oturma: Δψ küçülüyor, tekne dönüp rotaya hizalanıyor.  Rotada: |e| sonuna kadar < band."""
    t, e, dr = res["t"], res["e"], res["dr"]
    n = len(t)
    unsat = np.flatnonzero(np.abs(res["dpsi"]) < ap.dpsi_max - 1e-6)
    i_turn = int(unsat[0]) if len(unsat) else n
    out_band = np.flatnonzero(np.abs(e) >= band)
    i_set = 0 if not len(out_band) else (int(out_band[-1]) + 1 if out_band[-1] + 1 < n else None)
    if i_set is not None:
        i_turn = min(i_turn, i_set)
    phase = np.full(n, 2 if i_set is not None else 1, dtype=int)
    phase[:min(i_turn, n)] = 0
    if i_set is not None:
        phase[min(i_turn, n):i_set] = 1
    s0 = np.sign(e[0])
    i_max = int(np.argmax(np.abs(dr)))
    seg = np.hypot(np.diff(res["x"]), np.diff(res["y"]))
    path_len = float(seg.sum())
    rows = []
    for k, name in enumerate(PHASES):
        m = phase == k
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        t0, t1 = float(t[idx[0]]), float(t[min(idx[-1] + 1, n - 1)])
        rows.append(dict(name=name, t0=t0, t1=t1, dur=t1 - t0,
                         V=float(res["V"][m].mean() / KT2MS), dr_mean=float(dr[m].mean()),
                         dr_max=float(np.abs(dr[m]).max()), ds_mean=float(res["ds"][m].mean()),
                         lam_mean=float(res["lam"][m].mean()), e_max=float(np.abs(e[m]).max())))
    on = phase == 2
    return dict(
        phase=phase, i_turn=i_turn if i_turn < n else None, i_settle=i_set, band=band,
        t_turn=float(t[i_turn]) if i_turn < n else None, t_settle=None if i_set is None else float(t[i_set]),
        overshoot=float(max(0.0, (-s0 * e).max())) if s0 != 0 else float(np.abs(e).max()),
        dr_max=float(dr[i_max]), t_dr_max=float(t[i_max]),
        dr_hold=float(dr[on].mean()) if on.sum() >= 4 else None,
        dr_hold_std=float(dr[on].std()) if on.sum() >= 4 else None,
        path_len=path_len, eff=float((res["s"][-1] - res["s"][0]) / path_len) if path_len > 0 else None,
        rows=rows)


def sim_env(tws_kt, trim_auto, trim_manual, sail_type, p):
    """sim_physics.js için ortam: parametreler (türetilmişler Python'da hesaplanır) ve optimum trim tablosu."""
    pd_ = {k: getattr(p, k) for k in p.__dataclass_fields__}
    pd_.update({k: getattr(p, k) for k in ("draft", "S_wet", "A_hull_lat", "CLa_rud", "x_rud", "m11", "m22",
                                           "Iz", "Iz_added")})
    betas, best = _opt_trim_table(sail_type, p)
    trim = "opt" if trim_auto == "opt" else ("card" if trim_auto else "man")
    return dict(p=pd_, tws_kt=float(tws_kt), sail=sail_type, trim=trim, trim_manual=float(trim_manual),
                tbl=dict(betas=betas.tolist(), best=best.tolist()))


def route_js(rs):
    """route_start çıktısının sim_physics.js autopilot() girdisi."""
    return dict(A=rs["A"].tolist(), B=rs["B"].tolist(), L=rs["L"], chi=rs["chi"], t_hat=rs["t_hat"].tolist(),
                n_hat=rs["n_hat"].tolist(), lim=rs["lim"], V_ref=rs["V_ref"], st0=list(rs["st0"]),
                track_nogo=rs["track_nogo"])


ROUTE_CSV = (("t", "Zaman t (s)"), ("x", "Konum x Doğu (m)"), ("y", "Konum y Kuzey (m)"), ("phase", "Evre"),
             ("h", "Pruva ψ (°)"), ("psi_star", "Hedef pruva ψ* (°)"), ("e", "Sapma e (m)"),
             ("s", "Rota boyunca s (m)"), ("V", "Hız V (m/s)"), ("Vkt", "Hız V (kt)"), ("u", "İleri hız u (m/s)"),
             ("v", "Yanal hız v (m/s)"), ("r", "Dönüş hızı r (°/s)"), ("lam", "Yan kayma λ (°)"),
             ("cog", "Gidiş yönü COG (°)"), ("twa", "Gerçek rüzgar açısı TWA (°)"),
             ("beta", "Görünür rüzgar açısı β (°)"), ("aaw", "Görünür rüzgar hızı (m/s)"),
             ("ds", "Yelken açısı δs (°)"), ("alpha", "Yelken hücum açısı α (°)"), ("cmd", "Dümen komutu (°)"),
             ("dr", "Dümen açısı δr (°)"), ("o_P", "Dış döngü P (°)"), ("o_I", "Dış döngü I (°)"),
             ("o_D", "Dış döngü D (°)"), ("o_FF", "Dış döngü ileri besleme (°)"), ("dpsi", "Pruva düzeltmesi Δψ (°)"),
             ("clamp", "Ölü açı sınırında"), ("eps", "Pruva hatası ε (°)"), ("i_P", "İç döngü P (°)"),
             ("i_I", "İç döngü I (°)"), ("i_D", "İç döngü D (°)"), ("g", "Kazanç çarpanı g"),
             ("Fx", "Yelken ileri kuvvet Fx (N)"), ("Fy", "Yelken yanal kuvvet Fy (N)"),
             ("X_r", "Dümen boyuna kuvvet (N)"), ("Y_r", "Dümen yanal kuvvet (N)"),
             ("Y_h", "Gövde yanal kuvvet (N)"), ("R", "Su direnci R (N)"), ("N_s", "Yelken momenti (N·m)"),
             ("N_h", "Gövde momenti (N·m)"), ("N_r", "Dümen momenti (N·m)"), ("N_munk", "Munk momenti (N·m)"),
             ("flap", "Yelken çırpıyor"))


def route_csv(res, ph, excel_tr=False):
    """Tüm zaman serisi (0.25 s aralıkla). excel_tr: Türkçe Excel için ';' ayraç ve virgüllü ondalık."""
    cols = {}
    for k, name in ROUTE_CSV:
        if k == "phase":
            cols[name] = [PHASES[i] for i in ph["phase"]]
        elif k == "Vkt":
            cols[name] = np.asarray(res["V"]) / KT2MS
        elif k in ("clamp", "flap"):
            cols[name] = ["evet" if v else "hayır" for v in res[k]]
        else:
            cols[name] = np.asarray(res[k], dtype=float)
    df = pd.DataFrame(cols)
    return df.to_csv(index=False, sep=";" if excel_tr else ",", decimal="," if excel_tr else ".",
                     float_format="%.4f").encode("utf-8-sig")


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
# Kokpit görselleştirme (yalnız çizim — fizik motoruna dokunmaz)
# ----------------------------------------------------------------------------
BG = "#0F172A"          # sayfa
PANEL = "#1E293B"       # kartlar
WATER = "#0B1A2E"       # su
GRIDC = "#16304D"
TXT = "#E2E8F0"
MUTED = "#94A3B8"
GLOW = "#22D3EE"        # rota kılavuzu (turkuaz)


def _side_anchor(v):
    """Ekran yönü v için etiket bağlama noktası: etiket kutusu noktanın v tarafında, noktanın dışında kalır."""
    xa = "left" if v[0] > 0.38 else "right" if v[0] < -0.38 else "center"
    ya = "bottom" if v[1] > 0.38 else "top" if v[1] < -0.38 else "middle"
    return xa, ya


def _label(fig, at, text, color, size, away, gap=10):
    """Noktanın `away` (birim vektör) tarafına yaslı etiket; çizginin/okun üstüne binmez."""
    xa, ya = _side_anchor(away)
    fig.add_annotation(x=at[0], y=at[1], xref="x", yref="y", showarrow=False, text=f"<b>{text}</b>",
                       font=dict(color=color, size=size), bgcolor="rgba(15,23,42,0.85)", borderpad=3,
                       xanchor=xa, yanchor=ya, xshift=gap * away[0], yshift=gap * away[1])


def _arrow(fig, tail, tip, color, text, width=7, size=14, away=(0.0, 1.0)):
    fig.add_annotation(x=tip[0], y=tip[1], ax=tail[0], ay=tail[1], xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1.0, arrowwidth=width, arrowcolor=color,
                       text="", standoff=0)
    _label(fig, tip, text, color, size, away)


def build_cockpit_figure(heading, eq, path, sail_type, delta_r, p, lim, tws_kt, horizon):
    """Tekne merkezli, sabit kameralı kahraman görünüm (kuzey yukarı)."""
    fig = go.Figure()
    O = (0.0, 0.0)
    R = max(2.6, 1.5 * p.L)                       # sabit kamera yarıçapı (m)

    # Pusula halkası + ölü açı sektörü (kuvvet dengesi) + Eq.(12) referansı
    ring = np.array([R * 0.93 * compass_unit(a) for a in np.linspace(0, 360, 121)])
    fig.add_trace(go.Scatter(x=ring[:, 0], y=ring[:, 1], mode="lines", line=dict(color="#1F3B5C", width=1),
                             hoverinfo="skip", showlegend=False))
    for a, lbl, xa, ya in ((0, "K", "center", "top"), (90, "D", "right", "middle"),
                           (180, "G", "center", "bottom"), (270, "B", "left", "middle")):
        q = R * compass_unit(a)                       # kadraj kenarına içten yaslı: kesilmez
        fig.add_annotation(x=q[0], y=q[1], text=f"<b>{lbl}</b>", showarrow=False, xanchor=xa, yanchor=ya,
                           font=dict(color="#475569", size=13))
    for a in range(0, 360, 30):
        q1, q2 = R * 0.90 * compass_unit(a), R * 0.93 * compass_unit(a)
        fig.add_trace(go.Scatter(x=[q1[0], q2[0]], y=[q1[1], q2[1]], mode="lines",
                                 line=dict(color="#1F3B5C", width=1), hoverinfo="skip", showlegend=False))
    if lim is not None and lim > 0:
        angs = np.linspace(WIND_FROM_DEG - lim, WIND_FROM_DEG + lim, 40)
        sec = np.vstack([[0, 0], np.array([R * 0.93 * compass_unit(a_) for a_ in angs]), [0, 0]])
        fig.add_trace(go.Scatter(x=sec[:, 0], y=sec[:, 1], fill="toself", mode="lines", line=dict(width=0),
                                 fillcolor="rgba(239,68,68,0.10)", hoverinfo="text",
                                 text=f"Ölü açı ±{lim:.1f}° (kuvvet dengesinden)", showlegend=False))
        for sgn in (-1, 1):
            e = R * 0.93 * compass_unit(WIND_FROM_DEG + sgn * lim)
            fig.add_trace(go.Scatter(x=[0, e[0]], y=[0, e[1]], mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(color="rgba(239,68,68,0.55)", width=1.5)))
    for sgn in (-1, 1):
        e = R * 0.93 * compass_unit(WIND_FROM_DEG + sgn * p.zeta_deg)
        fig.add_trace(go.Scatter(x=[0, e[0]], y=[0, e[1]], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color="rgba(239,68,68,0.35)", width=1, dash="dot")))

    # Rota kılavuzu: tam rota, tekne ölçeğini bozmadan kadraja sığacak şekilde düzgün ölçeklenir
    k = 1.0
    stbd = compass_unit(heading + 90.0)
    path_side = 1.0                                     # rota sonu teknenin hangi bordasında (+ sancak)
    lbl_pts = []                                        # etiket noktaları (rüzgar okunun yeri için)
    if len(path) > 1:
        ext = float(np.max(np.hypot(path[:, 0], path[:, 1])))
        astern = float(np.mean(path[:, 5])) < 0           # çoğunlukla geri gidiyorsa kıçtan başlat
        bow = to_world([[-p.L / 2 - 0.25 if astern else p.L / 2, 0.0]], heading, 1.0, O)[0]
        k = min(1.0, (0.86 * R - p.L / 2) / max(ext, 1e-6))   # pruva + rota kadraja sığar
        pp = (path[:, :2] - path[0, :2]) * k + bow          # kılavuz pruvadan çıkar
        fig.add_trace(go.Scatter(x=pp[:, 0], y=pp[:, 1], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color="rgba(34,211,238,0.16)", width=10)))
        fig.add_trace(go.Scatter(
            x=pp[:, 0], y=pp[:, 1], mode="lines", showlegend=False,
            line=dict(color=GLOW, width=2, dash="dash"),
            customdata=np.c_[path[:, 2], path[:, 3] / KT2MS, path[:, 4], path[:, 5]],
            hovertemplate="t=%{customdata[0]:.1f} s · V=%{customdata[1]:.2f} kt<br>pruva %{customdata[2]:.0f}°"
                          " · u=%{customdata[3]:+.2f} m/s<extra>rota</extra>"))
        back = path[:, 5] < -0.02
        if back.any():
            fig.add_trace(go.Scatter(x=np.where(back, pp[:, 0], None), y=np.where(back, pp[:, 1], None),
                                     mode="lines", line=dict(color="#EF4444", width=4), hoverinfo="skip",
                                     showlegend=False))
        fig.add_trace(go.Scatter(x=[pp[-1, 0]], y=[pp[-1, 1]], mode="markers", showlegend=False,
                                 marker=dict(size=9, color=GLOW, line=dict(color=BG, width=2)), hoverinfo="skip"))
        # Süre etiketi rotanın dış tarafına; Fx etiketi karşı bordaya → ikisi üst üste binmez
        path_side = 1.0 if float((pp[-1] - np.asarray(O)) @ stbd) >= 0 else -1.0
        lbl_pts.append(pp[-1])
        _label(fig, pp[-1], f"{horizon:.0f} s", GLOW, 12, path_side * stbd, gap=8)

    # Tekne (gerçek boyut)
    hs = p.hull_sep / 2
    for xb in (0.30, -0.45):
        w = to_world([[xb, -hs], [xb, hs]], heading, 1.0, O)
        fig.add_trace(go.Scatter(x=w[:, 0], y=w[:, 1], mode="lines", line=dict(color="#94A3B8", width=7),
                                 hoverinfo="skip", showlegend=False))
    w = to_world([[-0.45, 0], [p.x_rud, 0]], heading, 1.0, O)
    fig.add_trace(go.Scatter(x=w[:, 0], y=w[:, 1], mode="lines", line=dict(color="#94A3B8", width=4),
                             hoverinfo="skip", showlegend=False))
    for side, col, edge, nm in ((-1, COL_PORT, "#FDBA74", "İskele gövde (turuncu)"),
                                (1, COL_STBD, "#CBD5E1", "Sancak gövde (siyah)")):
        w = to_world(hull_polygon(p, side), heading, 1.0, O)
        fig.add_trace(go.Scatter(x=w[:, 0], y=w[:, 1], fill="toself", mode="lines", fillcolor=col,
                                 line=dict(color=edge, width=1.6), hoverinfo="text", text=nm, showlegend=False))
    dr = math.radians(delta_r)
    blade = to_world([[p.x_rud, 0.0], [p.x_rud - 0.24 * math.cos(dr), 0.24 * math.sin(dr)]], heading, 1.0, O)
    fig.add_trace(go.Scatter(x=blade[:, 0], y=blade[:, 1], mode="lines+markers", showlegend=False,
                             line=dict(color="#60A5FA", width=8), marker=dict(size=[9, 0], color="#60A5FA"),
                             hoverinfo="text", text=f"Merkezi dümen {delta_r:+.0f}°"))
    mast = np.array([p.x_mast, 0.0])
    if sail_type == "wing":
        w = to_world(naca_polygon(p.chord_wing, eq["delta_s"], mast), heading, 1.0, O)
        fig.add_trace(go.Scatter(x=w[:, 0], y=w[:, 1], fill="toself", mode="lines", showlegend=False,
                                 fillcolor="rgba(226,232,240,0.92)", line=dict(color="#F8FAFC", width=1.5),
                                 hoverinfo="text", text=f"Sert kanat NACA 0012 · δs={eq['delta_s']:+.0f}°"))
    else:
        a = math.radians(180.0 - eq["delta_s"])
        boom = np.vstack([mast, mast + p.boom_len * np.array([math.cos(a), math.sin(a)])])
        w = to_world(boom, heading, 1.0, O)
        fig.add_trace(go.Scatter(x=w[:, 0], y=w[:, 1], mode="lines", line=dict(color="#A16207", width=5),
                                 hoverinfo="skip", showlegend=False))
        cur = to_world(soft_sail_curve(p.boom_len, eq["delta_s"], mast, eq["flapping"], eq["aw_b"]), heading, 1.0, O)
        col = "#F87171" if eq["flapping"] else "#C084FC" if eq["backwinded"] else "#F8FAFC"
        state = "ÇIRPIYOR" if eq["flapping"] else "TERS DOLMUŞ" if eq["backwinded"] else "dolu"
        fig.add_trace(go.Scatter(x=cur[:, 0], y=cur[:, 1], mode="lines", line=dict(color=col, width=4),
                                 hoverinfo="text", text=f"Bez yelken ({state}) · δs={eq['delta_s']:+.0f}°",
                                 showlegend=False))
    wm = to_world(mast, heading, 1.0, O)
    fig.add_trace(go.Scatter(x=wm[:, 0], y=wm[:, 1], mode="markers", hoverinfo="text", text="Ana direk",
                             marker=dict(size=11, color="#0F172A", line=dict(color="#F8FAFC", width=2.5)),
                             showlegend=False))

    # Kuvvet okları: gövdenin DIŞINDAN başlar (pruva/kıç ve borda), kalın, 14 px etiket
    Fx, Fy = eq["Fx"], eq["Fy"]
    Fref = max(abs(Fx), abs(Fy), 1e-6)
    Lmax = 0.38 * R

    def seg(F):
        return 0.0 if abs(F) < 1e-3 else max(0.22 * R * 0.5, Lmax * abs(F) / Fref)

    if abs(Fx) >= 1e-3:
        x0 = (p.L / 2 + 0.08) if Fx > 0 else -(p.L / 2 + 0.30)
        t = to_world([[x0, 0.0]], heading, 1.0, O)[0]
        e = to_world([[x0 + math.copysign(seg(Fx), Fx), 0.0]], heading, 1.0, O)[0]
        _arrow(fig, t, e, "#22C55E", f"Fx {Fx:+.1f} N", away=-path_side * stbd)
        lbl_pts.append(e)
    if abs(Fy) >= 1e-3:
        y0 = math.copysign(hs + 0.20, Fy)
        t = to_world([[0.0, y0]], heading, 1.0, O)[0]
        e = to_world([[0.0, y0 + math.copysign(seg(Fy), Fy)]], heading, 1.0, O)[0]
        _arrow(fig, t, e, "#EF4444", f"Fy {Fy:+.1f} N", away=math.copysign(1.0, Fy) * stbd)
        lbl_pts.append(e)

    # Rüzgar oku (kuzeyden) — kadrajın üst köşesi; diğer etiketlerden uzak olan köşe seçilir
    def clearance(sx):
        return min((float(np.hypot(q[0] - sx * 0.80 * R, q[1] - 0.55 * R)) for q in lbl_pts), default=R)

    wx = (-0.80 if clearance(-1) >= clearance(1) else 0.80) * R
    _arrow(fig, (wx, 0.98 * R), (wx, 0.62 * R), "#60A5FA", f"Rüzgar {tws_kt:.0f} kt", width=5, size=13,
           away=(0.0, -1.0))

    if k < 0.999:
        fig.add_annotation(x=0.99, y=0.01, xref="paper", yref="paper", xanchor="right", yanchor="bottom",
                           showarrow=False, text=f"rota kılavuzu ölçeği ×{k:.2f} · tekne gerçek boyut",
                           font=dict(color=MUTED, size=11))
    fig.update_layout(
        autosize=True, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=BG, plot_bgcolor=WATER,
        showlegend=False, dragmode="pan", hoverlabel=dict(bgcolor=PANEL, font_color=TXT),
        xaxis=dict(range=[-R, R], showgrid=True, gridcolor=GRIDC, zeroline=False, showticklabels=False,
                   fixedrange=False),
        yaxis=dict(range=[-R, R], showgrid=True, gridcolor=GRIDC, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
    )
    return fig


def build_minimap(path, heading, p, lim):
    """Gerçek ölçekli rota haritası (m)."""
    fig = go.Figure()
    if len(path) > 1:
        fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines", line=dict(color=GLOW, width=2.5),
                                 hoverinfo="skip"))
        back = path[:, 5] < -0.02
        if back.any():
            fig.add_trace(go.Scatter(x=np.where(back, path[:, 0], None), y=np.where(back, path[:, 1], None),
                                     mode="lines", line=dict(color="#EF4444", width=3), hoverinfo="skip"))
    for row, col in ((path[0], "#F97316"), (path[-1], GLOW)):
        tri = to_world([[0.9, 0], [-0.9, -0.5], [-0.9, 0.5], [0.9, 0]], row[4], 1.0, (row[0], row[1]))
        fig.add_trace(go.Scatter(x=tri[:, 0], y=tri[:, 1], fill="toself", mode="lines", fillcolor=col,
                                 line=dict(color=col, width=1), hoverinfo="skip"))
    pts = path[:, :2]
    lo, hi = pts.min(axis=0) - 3, pts.max(axis=0) + 3
    c, h = (lo + hi) / 2, max(hi - lo) / 2
    fig.update_layout(height=210, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=PANEL, plot_bgcolor=WATER,
                      showlegend=False,
                      xaxis=dict(range=[c[0] - h, c[0] + h], gridcolor=GRIDC, zeroline=False,
                                 tickfont=dict(color=MUTED, size=9), ticksuffix=" m"),
                      yaxis=dict(range=[c[1] - h, c[1] + h], gridcolor=GRIDC, zeroline=False,
                                 tickfont=dict(color=MUTED, size=9), scaleanchor="x", scaleratio=1))
    return fig


# ----------------------------------------------------------------------------
# Kokpit arayüzü
# ----------------------------------------------------------------------------
CSS = f"""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{ background:{BG}; color:{TXT}; }}
header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display:none !important; }}
.block-container {{ padding:0.55rem 1rem 0 1rem !important; max-width:100% !important; }}
[data-testid="stSidebar"] {{ background:#0B1222; border-right:1px solid #1E293B; }}
/* Sabit sol menü: küçültme düğmesi ve genişlik tutamacı yok, her koşulda açık */
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarResizeHandle"],
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] {{ display:none !important; }}
section[data-testid="stSidebar"] {{ transform:none !important; visibility:visible !important;
    min-width:300px !important; width:300px !important; max-width:300px !important; }}
section[data-testid="stSidebar"] > div {{ width:300px !important; }}
[data-testid="stSidebarHeader"] {{ height:0.4rem !important; min-height:0 !important; padding:0 !important;
    margin:0 !important; overflow:hidden; }}
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarContent"] {{ padding-top:0.6rem; }}
[data-testid="stSidebarUserContent"] {{ padding-top:0.4rem; }}
[data-testid="stVerticalBlock"] {{ gap:0.45rem; }}
.ck-row {{ display:flex; gap:10px; align-items:stretch; }}
.ck-card {{ flex:1; background:{PANEL}; border:1px solid #334155; border-radius:12px; padding:8px 14px;
           min-width:0; cursor:help; }}
.ck-lbl {{ font-size:11.5px; letter-spacing:.06em; text-transform:uppercase; color:{MUTED}; white-space:nowrap;
          overflow:hidden; text-overflow:ellipsis; }}
.ck-val {{ font-size:27px; font-weight:800; line-height:1.15; white-space:nowrap; font-variant-numeric:tabular-nums; }}
.ck-unit {{ font-size:14px; font-weight:600; color:{MUTED}; margin-left:3px; }}
.ck-sub {{ font-size:12px; font-weight:600; color:#FCA5A5; margin-left:8px; letter-spacing:0; }}
.ck-badge {{ display:flex; flex-direction:column; justify-content:center; align-items:center; min-width:150px;
            border-radius:12px; padding:6px 14px; font-weight:800; letter-spacing:.08em; color:#0F172A;
            cursor:help; text-align:center; }}
.ck-badge small {{ font-weight:600; letter-spacing:0; font-size:11.5px; opacity:.85; }}
.ck-panel {{ background:{PANEL}; border:1px solid #334155; border-radius:12px; padding:8px 12px; }}
.ck-pr {{ display:flex; justify-content:space-between; align-items:baseline; padding:3px 0;
         border-bottom:1px solid #263449; font-size:13px; cursor:help; }}
.ck-pr:last-child {{ border-bottom:none; }}
.ck-pr span:first-child {{ color:{MUTED}; }}
.ck-pr b {{ font-size:15px; font-variant-numeric:tabular-nums; }}
.ck-h {{ font-size:11.5px; letter-spacing:.08em; text-transform:uppercase; color:{MUTED}; margin:2px 0 4px; }}
.ck-chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
.ck-chip {{ font-size:12px; padding:3px 9px; border-radius:999px; border:1px solid; cursor:help; white-space:nowrap; }}
div[data-testid="stPlotlyChart"] {{ background:{BG}; border-radius:12px; overflow:hidden; }}
.st-key-hero {{ flex:0 0 calc(100vh - 104px) !important; height:calc(100vh - 104px) !important;
               min-height:420px; }}
.st-key-hero [data-testid="stElementContainer"] {{ height:100% !important; }}
.ck-row {{ margin-bottom:18px; }}
[data-testid="stSidebar"] button[kind] {{ height:46px; font-weight:700; }}
/* Kendi HTML bloklarımız: Streamlit'in markdown alt boşluk düzeltmesi (−16px) bloklarımızı üst üste bindiriyordu */
[data-testid="stMarkdownContainer"]:has(> .ck-panel, > .ck-chips, > .ck-h, > .rt-title, > .rt-tbl) {{ margin-bottom:0 !important; }}
.rt-title {{ font-size:22px; font-weight:800; margin:2px 0 4px; }}
.rt-desc {{ color:{MUTED}; font-size:13px; line-height:1.45; margin-bottom:6px; }}
.rt-sum {{ flex-wrap:wrap; margin-bottom:10px !important; }}
.rt-sum .ck-card {{ min-width:150px; }}
.rt-sum .ck-val {{ font-size:22px; }}
.rt-tbl {{ overflow-x:auto; background:{PANEL}; border:1px solid #334155; border-radius:12px; margin-bottom:8px; }}
.rt-tbl table {{ width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }}
.rt-tbl th {{ color:{MUTED}; font-size:12px; font-weight:700;
             text-align:right; padding:7px 10px; border-bottom:1px solid #334155; white-space:nowrap; }}
.rt-tbl td {{ text-align:right; padding:6px 10px; border-bottom:1px solid #263449; white-space:nowrap; }}
.rt-tbl tr:last-child td {{ border-bottom:none; }}
.rt-tbl th:first-child, .rt-tbl td:first-child {{ text-align:left; font-weight:700; }}
.rt-tbl td i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:7px; }}
/* Kaydırıcı + sayı kutusu: dar kutuda +/- düğmeleri yer kaplamasın (ok tuşları ve yazma çalışır) */
[class*="st-key-pair_"] [data-testid="stNumberInputStepUp"],
[class*="st-key-pair_"] [data-testid="stNumberInputStepDown"] {{ display:none !important; }}
[class*="st-key-pair_"] [data-testid="stNumberInputField"] {{ text-align:center; padding-left:4px; padding-right:4px; }}
</style>
"""


def _esc(s):
    return (str(s).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;"))


def _card(label, value, unit, color, tip, sub=None):
    sub_html = f'<span class="ck-sub">{sub}</span>' if sub else ""
    return (f'<div class="ck-card" title="{_esc(tip)}"><div class="ck-lbl">{label} ⓘ</div>'
            f'<div class="ck-val" style="color:{color}">{value}<span class="ck-unit">{unit}</span>{sub_html}</div></div>')


def _row(label, value, tip, color=TXT):
    return f'<div class="ck-pr" title="{_esc(tip)}"><span>{label}</span><b style="color:{color}">{value}</b></div>'


def _chip(text, color, tip):
    return (f'<span class="ck-chip" style="color:{color};border-color:{color};background:{color}1A" '
            f'title="{_esc(tip)}">{text}</span>')


def _set_sail(v):
    st.session_state["sail"] = v


def _sync(key, src):
    st.session_state[key] = st.session_state[src]


def _slider_box(label, lo, hi, default, step, key, help=None):
    """Kaydırıcı + yanında elle sayı girilebilen kutu. İkisi senkron; değer st.session_state[key]'de tutulur."""
    sk, nk = f"{key}__s", f"{key}__n"
    if key not in st.session_state:
        st.session_state[key] = default
    st.session_state[sk] = st.session_state[nk] = st.session_state[key]
    with st.container(key=f"pair_{key}"):
        c1, c2 = st.columns([2.5, 1], gap="small", vertical_alignment="bottom")
        c1.slider(label, lo, hi, step=step, key=sk, help=help, on_change=_sync, args=(key, sk))
        c2.number_input(label, lo, hi, step=step, key=nk, label_visibility="collapsed",
                        on_change=_sync, args=(key, nk))
    return st.session_state[key]


# ----------------------------------------------------------------------------
# A→B rota sayfası (benzetim Python'da bir kez hesaplanır; oynatma tarayıcıda, yeniden hesap yok)
# ----------------------------------------------------------------------------
ROUTE_FIELDS = ("t", "x", "y", "h", "V", "r", "e", "s", "psi_star", "o_P", "o_I", "o_D", "o_FF", "dpsi", "i_P",
                "i_I", "i_D", "g", "cmd", "dr", "ds", "beta", "aaw", "Fx", "Fy", "X_r", "Y_r", "Y_h", "R", "N_s",
                "N_r", "lam", "cog", "twa", "alpha", "eps", "N_h", "N_munk")
ARRIVE_TOL = 5.0           # m: B hizası geçilirken rotaya uzaklık bundan küçükse "varış" (≈3 tekne boyu; yalnız gösterim)
ROUTE_VIEW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "route_view.html")


@st.cache_data(show_spinner=False, max_entries=16)
def _route_cached(A, B, e0, head_err, tws_kt, trim_auto, trim_manual, sail_type, p_vals, ap_vals):
    return simulate_route(A, B, e0, head_err, tws_kt, trim_auto, trim_manual, sail_type,
                          Params(*p_vals), Autopilot(*ap_vals))


def _route_html(res, ph, tws_kt, sail_type, p, ap):
    data = {k: np.round(np.asarray(res[k], dtype=float), 3).tolist() for k in ROUTE_FIELDS}
    data["clamp"] = [bool(v) for v in res["clamp"]]
    data["flap"] = [bool(v) for v in res["flap"]]
    data["phase"] = ph["phase"].tolist()
    data["meta"] = dict(
        phases=dict(names=PHASES, i_turn=ph["i_turn"], i_settle=ph["i_settle"], band=ph["band"]),
        A=res["A"].tolist(), B=res["B"].tolist(), P0=res["P0"].tolist(), chi=res["chi"], L=res["L"],
        lim=res["lim"], status=res["status"], t_end=res["t_end"], t_max=res["t_max"], tws_kt=tws_kt, rec_dt=0.25,
        boat=dict(L=p.L, sep=p.hull_sep, x_mast=p.x_mast,
                  boom=p.boom_len if sail_type == "soft" else 0.75 * p.chord_wing),
        ap=dict(dpsi_max=ap.dpsi_max, rud_max=ap.rud_max, gain_sched=ap.gain_sched))
    with open(ROUTE_VIEW, encoding="utf-8") as fh:
        return fh.read().replace("__DATA__", json.dumps(data))


def _pnum(col, label, lo, hi, default, step, key, **kw):
    """Mod değişince kaybolmayan sayı kutusu: değer st.session_state[key]'de, widget anahtarı key__w."""
    wk = f"{key}__w"
    if key not in st.session_state:
        st.session_state[key] = default
    st.session_state[wk] = st.session_state[key]
    col.number_input(label, lo, hi, step=step, key=wk, on_change=_sync, args=(key, wk), **kw)
    return st.session_state[key]


def _ptoggle(col, label, default, key, **kw):
    wk = f"{key}__w"
    if key not in st.session_state:
        st.session_state[key] = default
    st.session_state[wk] = st.session_state[key]
    col.toggle(label, key=wk, on_change=_sync, args=(key, wk), **kw)
    return st.session_state[key]


def _route_controls(flat=False):
    """A→B senaryosu ve otopilot ayarları. A→B Rota ve Kumanda sayfalarında ortak; sayfa değişince korunur.
    flat: PID ayarları ayrı açılır kutusuz (zaten bir açılır kutunun içindeyken)."""
    d = Autopilot()
    c = st.columns(4)
    ax = _pnum(c[0], "A · Doğu x (m)", -5000.0, 5000.0, 0.0, 10.0, "rt_ax")
    ay = _pnum(c[1], "A · Kuzey y (m)", -5000.0, 5000.0, 0.0, 10.0, "rt_ay")
    bx = _pnum(c[2], "B · Doğu x (m)", -5000.0, 5000.0, 200.0, 10.0, "rt_bx")
    by = _pnum(c[3], "B · Kuzey y (m)", -5000.0, 5000.0, 0.0, 10.0, "rt_by")
    c = st.columns(4, vertical_alignment="bottom")
    e0 = _pnum(c[0], "Başlangıç sapması (m)", -500.0, 500.0, 30.0, 5.0, "rt_e0",
               help="Başlangıç noktasının A'dan rotaya dik uzaklığı e₀. + : rotanın sancak tarafı "
                    "(A'dan B'ye bakarken sağ), − : iskele tarafı.")
    he = _pnum(c[1], "Pruva hatası (°)", -180.0, 180.0, 0.0, 5.0, "rt_he",
               help="Başlangıç pruvası = rota açısı χ + bu değer. Tekne o pruvadaki durağan hızla başlar.")
    c = st.columns(2)
    ff = _ptoggle(c[0], "Yan kayma telafisi (ileri besleme)", d.leeway_ff, "rt_ff",
                  help="Hedef pruvadan yan kayma açısı λ çıkarılır; böylece teknenin GİDİŞ yönü (COG) rotaya oturur.")
    gs = _ptoggle(c[1], "Hıza göre kazanç", d.gain_sched, "rt_gs",
                  help="Dümen etkinliği hızın karesiyle değişir; iç döngü kazancı (V_ref/V)² ile ölçeklenir (×0.5–×3).")
    with st.container() if flat else st.expander("PID kazançları ve dümen sınırları", expanded=False):
        if flat:
            st.caption("PID kazançları ve dümen sınırları")
        c = st.columns(4, vertical_alignment="bottom")
        kp_e = _pnum(c[0], "Dış döngü P (°/m)", 0.0, 50.0, d.kp_e, 0.5, "rt_kpe")
        ki_e = _pnum(c[1], "Dış döngü I (°/(m·s))", 0.0, 2.0, d.ki_e, 0.005, "rt_kie", format="%.3f")
        kd_e = _pnum(c[2], "Dış döngü D (°/(m/s))", 0.0, 100.0, d.kd_e, 1.0, "rt_kde")
        dpm = _pnum(c[3], "En büyük yaklaşma açısı Δψmax (°)", 5.0, 90.0, d.dpsi_max, 5.0, "rt_dpm")
        c = st.columns(4, vertical_alignment="bottom")
        kp_h = _pnum(c[0], "İç döngü P (°/°)", 0.0, 10.0, d.kp_h, 0.1, "rt_kph")
        ki_h = _pnum(c[1], "İç döngü I (°/(°·s))", 0.0, 2.0, d.ki_h, 0.01, "rt_kih", format="%.3f")
        kd_h = _pnum(c[2], "İç döngü D (°/(°/s))", 0.0, 10.0, d.kd_h, 0.1, "rt_kdh")
        rmx = _pnum(c[3], "Dümen sınırı (°)", 5.0, 35.0, d.rud_max, 1.0, "rt_rmx")
        c = st.columns(4)
        rrt = _pnum(c[0], "Dümen servo hızı (°/s)", 5.0, 600.0, d.rud_rate, 5.0, "rt_rrt")
    ap = Autopilot(kp_e=kp_e, ki_e=ki_e, kd_e=kd_e, dpsi_max=dpm, kp_h=kp_h, ki_h=ki_h, kd_h=kd_h,
                   rud_max=rmx, rud_rate=rrt, leeway_ff=ff, gain_sched=gs)
    return (ax, ay), (bx, by), e0, he, ap


def _route_ok(A, B):
    if math.hypot(B[0] - A[0], B[1] - A[1]) < 5.0:
        st.warning("A ile B arası en az 5 m olmalı.")
        return False
    return True


def route_page(sail_type, tws_kt, trim_auto, trim_manual, p):
    st.markdown('<div class="rt-title">A\'dan B\'ye düz rota · kaskad PID otopilot</div>'
                '<div class="rt-desc">Tekne A\'nın dışından başlar ve A–B doğrusuna oturmaya çalışır: dış döngü '
                'doğrudan sapmayı (e) ölçüp hedef pruvayı (ψ*) verir, iç döngü pruva hatasından dümen açısını (δr) '
                'hesaplar. Fizik kokpitle <b>aynıdır</b> (görünür rüzgarda yelken kuvvetleri, 3-DOF ataletli hareket, '
                'yan kayma, dümen stall\'u); otopilot yalnız dümeni yönetir. Yelken, rüzgar ve tekne parametreleri '
                'sol menüdedir. Rüzgar kuzeyden eser.</div>', unsafe_allow_html=True)
    (ax, ay), (bx, by), e0, he, ap = _route_controls()
    if not _route_ok((ax, ay), (bx, by)):
        return
    with st.spinner("Rota benzetimi hesaplanıyor…"):
        res = _route_cached((ax, ay), (bx, by), e0, he, tws_kt, trim_auto, trim_manual, sail_type,
                            astuple(p), astuple(ap))
    twa_t = abs(wrap180(res["chi"] - WIND_FROM_DEG))
    lim = res["lim"]
    if res["track_nogo"]:
        st.warning(f"Rota ölü açıda: rota TWA {twa_t:.0f}° < ölü açı sınırı ±{lim:.1f}°. Tekne bu doğru boyunca "
                   "gidemez; otopilot hedef pruvayı ölü açı sınırında tutar. Bu sürümde tramola (zig-zag) yok.")
    end_e = abs(float(res["e"][-1]))
    # Benzetim, tekne rota boyunca B hizasını geçince durur (fizik/durma koşulu değişmedi). Yalnız GÖSTERİM:
    # B'yi ARRIVE_TOL m'den yakın geçtiyse "varış", daha uzaktan geçtiyse "B hizası geçildi" denir.
    if res["status"] == "reached" and end_e <= ARRIVE_TOL:
        arr = _card("Varış süresi", f"{res['t_end']:.0f}", "s", "#22C55E",
                    f"B'ye {end_e:.1f} m yakından varılan an (tolerans {ARRIVE_TOL:.0f} m).")
    elif res["status"] == "reached":
        arr = _card("B hizası geçildi", f"{res['t_end']:.0f}", "s", "#F59E0B",
                    f"Tekne rota boyunca B'nin hizasını {res['t_end']:.0f}. saniyede geçti ama B'ye varmadı: "
                    f"B'nin {end_e:.1f} m yanından geçti (varış toleransı {ARRIVE_TOL:.0f} m).",
                    sub=f"B'ye {end_e:.0f} m")
    else:
        why = {"nogo": "rota ölü açıda", "stalled": "tekne durdu", "time": "süre doldu"}.get(res["status"], "")
        arr = _card("Varış süresi", "—", "s", "#22C55E", f"B'ye varılamadı ({why}).", sub=why or None)
    cards = "".join([
        _card("Rota açısı χ", f"{res['chi']:.1f}", "°", "#2DD4BF", "A'dan B'ye pusula açısı (0° = Kuzey)."),
        _card("Mesafe", f"{res['L']:.0f}", "m", "#38BDF8", "A–B doğrusunun uzunluğu."),
        _card("Rota TWA", f"{twa_t:.0f}", "°", "#EF4444" if res["track_nogo"] else "#FBBF24",
              f"Rota boyunca gerçek rüzgar açısı. Ölü açı sınırı ±{lim:.1f}°." if lim else "Gerçek rüzgar açısı."),
        arr,
        _card("Son sapma |e|", f"{end_e:.2f}", "m", "#22C55E" if end_e < 1 else "#F59E0B",
              "Benzetim sonunda rota doğrusuna uzaklık."),
    ])
    st.markdown(f'<div class="ck-row">{cards}</div>', unsafe_allow_html=True)
    ph = route_phases(res, ap)
    components.html(_route_html(res, ph, tws_kt, sail_type, p, ap), height=915, scrolling=True)
    route_summary(res, ph, ap)


def route_summary(res, ph, ap):
    """Rapor için özet: ana sonuçlar, evre tablosu ve tüm zaman serisinin CSV'si."""
    st.markdown('<div class="ck-h" style="margin-top:6px">Özet</div>', unsafe_allow_html=True)
    f = lambda v, fmt, u="": "—" if v is None else f"{v:{fmt}}{u}"
    hold = "—" if ph["dr_hold"] is None else f"{ph['dr_hold']:+.1f}° ±{ph['dr_hold_std']:.1f}"
    # Rota ölü açıdaysa tekne bu doğruya oturamaz: oturma/aşım/verim başarım ölçüsü olarak anlamsızdır
    # (aşım, benzetimin |e| > |e0| + 40 m eşiğinde durmasından gelir). Değerler gösterilmez, nedeni ipucunda.
    nogo = bool(res.get("track_nogo"))
    nogo_tip = ("Rota ölü açıda: tekne bu doğruya oturamaz (tramola yok), bu değer başarım ölçüsü olarak anlamsız. "
                "Gözlenen değer: ")
    if nogo:
        st.caption("Rota ölü açıda: oturma, aşım ve yol verimi gösterilmiyor; evre tablosu yalnız gözlemdir.")
    cards = "".join([
        _card("Oturma başladı", "—" if nogo else f(ph["t_turn"], ".1f"), "s", "#C084FC",
              (nogo_tip + f(ph["t_turn"], ".1f", " s")) if nogo else
              f"Dış döngünün doymadan çıktığı an (|Δψ| < Δψmax = {ap.dpsi_max:.0f}°): tekne rotaya en büyük açıyla "
              "yaklaşmayı bırakıp dönmeye başlıyor."),
        _card("Rotaya oturdu", f(ph["t_settle"], ".1f"), "s", "#2DD4BF",
              f"Bu andan sonra sapma hep |e| < {ph['band']:.0f} m."),
        _card("Aşım", "—" if nogo else f"{ph['overshoot']:.2f}", "m", "#F59E0B",
              (nogo_tip + f"{ph['overshoot']:.2f} m") if nogo else
              "Rotanın öbür tarafına en fazla ne kadar geçtiği."),
        _card("En büyük dümen", f"{ph['dr_max']:+.1f}", "°", "#F87171",
              f"{ph['t_dr_max']:.1f}. saniyede. Sınır ±{ap.rud_max:.0f}°."),
        _card("Rotada dümen", hold, "", "#60A5FA",
              "Rotaya oturduktan sonra ortalama dümen açısı ± standart sapma: yelkenin döndürme etkisini "
              "dengeleyen açı."),
        _card("Yol verimi", "—" if nogo else f(None if ph["eff"] is None else 100 * ph["eff"], ".0f"), "%", "#22C55E",
              (nogo_tip + f(None if ph["eff"] is None else 100 * ph["eff"], ".0f", " %")) if nogo else
              f"Rota boyunca ilerleme / gidilen yol ({ph['path_len']:.0f} m). %100 = hiç sapmadan düz gitmek."),
    ])
    st.markdown(f'<div class="ck-row rt-sum">{cards}</div>', unsafe_allow_html=True)
    head = ("Evre", "Zaman aralığı (s)", "Süre (s)", "Ort. hız (kt)", "Ort. dümen δr (°)", "En büyük |δr| (°)",
            "Ort. yelken δs (°)", "Ort. yan kayma λ (°)", "En büyük |e| (m)")
    color = dict(zip(PHASES, ("#38BDF8", "#C084FC", "#2DD4BF")))
    body = "".join(
        f'<tr><td><i style="background:{color[r["name"]]}"></i>{r["name"]}</td>'
        f'<td>{r["t0"]:.1f} – {r["t1"]:.1f}</td><td>{r["dur"]:.1f}</td><td>{r["V"]:.2f}</td>'
        f'<td>{r["dr_mean"]:+.1f}</td><td>{r["dr_max"]:.1f}</td><td>{r["ds_mean"]:+.1f}</td>'
        f'<td>{r["lam_mean"]:+.1f}</td><td>{r["e_max"]:.2f}</td></tr>' for r in ph["rows"])
    st.markdown(f'<div class="rt-tbl"><table><thead><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr></thead>'
                f'<tbody>{body}</tbody></table></div>', unsafe_allow_html=True)
    c = st.columns([1, 1, 2], vertical_alignment="center")
    c[0].download_button("⬇ CSV · Excel (Türkçe)", route_csv(res, ph, excel_tr=True), "rota_benzetimi_excel.csv",
                         "text/csv", use_container_width=True, key="rt_csv_tr",
                         help="Türkçe Excel için: ';' ile ayrılmış, ondalık virgüllü.")
    c[1].download_button("⬇ CSV · standart", route_csv(res, ph), "rota_benzetimi.csv", "text/csv",
                         use_container_width=True, key="rt_csv",
                         help="',' ile ayrılmış, ondalık noktalı (Python, MATLAB, İngilizce Excel).")
    c[2].caption(f"Tüm zaman serisi, 0.25 s aralıkla {len(res['t'])} satır, {len(ROUTE_CSV)} sütun: konum, evre, "
                 "açılar, hızlar, PID katkıları, kuvvetler ve momentler.")


# ----------------------------------------------------------------------------
# Kumanda sayfası: fizik tarayıcıda (sim_physics.js = Python fiziğinin birebir karşılığı, testle doğrulanır)
# ----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DRIVE_VIEW, SIM_JS = os.path.join(_HERE, "drive_view.html"), os.path.join(_HERE, "sim_physics.js")
DRIVE_URL = "kumanda"


@st.cache_data(show_spinner=False, max_entries=16)
def _route_start_cached(A, B, e0, head_err, tws_kt, trim_auto, trim_manual, sail_type, p_vals):
    return route_js(route_start(A, B, e0, head_err, tws_kt, trim_auto, trim_manual, sail_type, Params(*p_vals)))


def _drive_html(rt, env, ap, ghost, e0, tws_kt, sail_type, p):
    data = dict(env=env, rt=rt, ap={k: getattr(ap, k) for k in ap.__dataclass_fields__}, ghost=ghost, e0=e0,
                meta=dict(tws_kt=tws_kt, lim=rt["lim"], sail=sail_type,
                          boat=dict(L=p.L, sep=p.hull_sep, x_mast=p.x_mast,
                                    boom=p.boom_len if sail_type == "soft" else 0.75 * p.chord_wing)))
    with open(DRIVE_VIEW, encoding="utf-8") as fh, open(SIM_JS, encoding="utf-8") as fs:
        return fh.read().replace("/*__SIM__*/", fs.read()).replace("__DATA__", json.dumps(data))


def drive_page(sail_type, tws_kt, trim_auto, trim_manual, p):
    st.markdown('<div class="rt-title">Kumanda ile sür · A\'dan B\'ye</div>'
                '<div class="rt-desc">Tekneyi kumandayla (Xbox / PlayStation, USB ya da Bluetooth) veya klavyeyle sen '
                'yönetirsin. Fizik kokpit ve A→B ile <b>aynıdır</b> (aynı denklemlerin tarayıcıda çalışan karşılığı, '
                'testle doğrulanır). Kumandayı bağladıktan sonra bir tuşuna bas; tarayıcı ancak o zaman görür.</div>',
                unsafe_allow_html=True)
    with st.expander("Senaryo ve otopilot ayarları (A, B, başlangıç, PID) · A→B Rota ile ortak", expanded=False):
        A, B, e0, he, ap = _route_controls(flat=True)
    if not _route_ok(A, B):
        return
    with st.spinner("Senaryo hazırlanıyor…"):
        rt = _route_start_cached(A, B, e0, he, tws_kt, trim_auto, trim_manual, sail_type, astuple(p))
        res = _route_cached(A, B, e0, he, tws_kt, trim_auto, trim_manual, sail_type, astuple(p), astuple(ap))
    ghost = dict(x=np.round(res["x"], 2).tolist(), y=np.round(res["y"], 2).tolist(),
                 t_end=res["t_end"] if res["status"] == "reached" else None)
    if rt["track_nogo"]:
        st.warning("Rota ölü açıda: tekne bu doğru boyunca doğrudan gidemez (tramola yok).")
    env = sim_env(tws_kt, trim_auto, trim_manual, sail_type, p)
    components.html(_drive_html(rt, env, ap, ghost, e0, tws_kt, sail_type, p), height=800, scrolling=True)


def main():
    st.set_page_config(page_title="Katamaran Kokpiti — Modül 1", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    if "sail" not in st.session_state:
        st.session_state["sail"] = "wing"

    # ---------------- Sayfalar: Simülasyon (Kokpit, A→B) ve ayrı Kumanda sayfası (/kumanda) ----------------
    def simulasyon():
        if cockpit:
            cockpit_page(heading, delta_r, horizon, *page_args)
        else:
            route_page(*page_args)

    def kumanda():
        drive_page(*page_args)

    page = st.navigation([st.Page(simulasyon, title="Simülasyon", icon="⛵", default=True),
                          st.Page(kumanda, title="Kumanda ile sür", icon="🎮", url_path=DRIVE_URL)])

    # ---------------- Sol menü (iki sayfada ortak; sayfa değişince değerler korunur) ----------------
    with st.sidebar:
        cockpit = False
        if page.url_path != DRIVE_URL:
            if "mode_val" not in st.session_state:
                st.session_state["mode_val"] = "Kokpit"
            st.session_state["mode"] = st.session_state["mode_val"]
            cockpit = st.radio("Mod", ["Kokpit", "A→B Rota"], horizontal=True, key="mode",
                               label_visibility="collapsed", on_change=_sync, args=("mode_val", "mode"),
                               help="Kokpit: sabit pruva/dümen. A→B Rota: otopilot tekneyi A–B doğrusuna "
                                    "oturtur.") == "Kokpit"
        st.markdown('<div class="ck-h">Yelken</div>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        b1.button("Sert Kanat", use_container_width=True, on_click=_set_sail, args=("wing",),
                  type="primary" if st.session_state["sail"] == "wing" else "secondary")
        b2.button("Bez Yelken", use_container_width=True, on_click=_set_sail, args=("soft",),
                  type="primary" if st.session_state["sail"] == "soft" else "secondary")
        sail_type = st.session_state["sail"]
        if cockpit:                                   # A→B modunda pruva ve dümeni otopilot verir
            heading = _slider_box("Pruva θ (°)", 0, 360, 90, 1, "heading", help="0° = Kuzey, saat yönünde")
            delta_r = _slider_box("Merkezi dümen δr (°)", -35, 35, 0, 1, "rudder",
                                  help="+ sancağa, − iskeleye dönüş")
        tws_kt = _slider_box("Rüzgar hızı (kt)", 2, 30, 10, 1, "tws", help="Rüzgar yönü sabit: 0° (Kuzeyden)")
        trim_on = st.toggle("Otomatik yelken trimi", value=True,
                            help="Açık: yelken açısı otomatik ayarlanır. Yöntem 'Model parametreleri → Otomatik trim "
                                 "yöntemi'nden seçilir (varsayılan: her an ileri itişi en büyük yapan açı; diğeri "
                                 "Jaulin 2013 Eq.(11) kardioid kuralı). Kapalı: yelken açısını elle ver.")
        trim_manual = 45
        if not trim_on:
            trim_manual = _slider_box("Yelken açısı δs (°)", -90, 90, 45, 1, "trim",
                                      help="+ bumba sancakta. Iskotalı bez yelkende yalnız |δs| sınırdır.")

        with st.expander("Model parametreleri (VARSAYIM)", expanded=False):
            auto_method = st.radio("Otomatik trim yöntemi", ["Optimum (maks. itiş)", "Jaulin Eq.(11) kardioid"],
                                   help="Optimum: kuvvet hesabından. Jaulin 2013 Eq.(11): kardioid kural.")
            boom_lbl = st.radio("Bez yelken bumba tahriki", ["Iskota (ip)", "Rijit servo"],
                                help="Iskota yalnız çeker: bumba rüzgar altına savrulur (Jaulin 2013). "
                                     "Rijit: komut açısında tutulur; ters rüzgarda yelken ters dolar.")
            if cockpit:
                horizon = _slider_box("Rota süresi (s)", 5, 60, 20, 1, "horizon")
            d = Params()
            p = replace(
                d,
                L=st.number_input("Boy L (m)", 0.5, 5.0, d.L, 0.05),
                mass=st.number_input("Kütle (kg)", 1.0, 200.0, d.mass, 1.0),
                hull_sep=st.number_input("Gövde aralığı (m)", 0.2, 2.0, d.hull_sep, 0.05),
                hull_B=st.number_input("Tek gövde genişliği B (m)", 0.05, 0.6, d.hull_B, 0.01),
                Cb=st.number_input("Blok katsayısı Cb", 0.2, 0.8, d.Cb, 0.01),
                C_R=st.number_input("Artık direnç C_R", 0.0, 0.05, d.C_R, 0.001, format="%.3f"),
                A_wing=st.number_input("Kanat alanı (m²)", 0.05, 3.0, d.A_wing, 0.01),
                A_soft=st.number_input("Bez yelken alanı (m²)", 0.05, 3.0, d.A_soft, 0.01),
                CDA_windage=st.number_input("Su üstü rüzgar direnci CD·A (m²)", 0.0, 1.0, d.CDA_windage, 0.01),
                h_CE=st.number_input("Yelken basınç merkezi yüksekliği (m)", 0.1, 3.0, d.h_CE, 0.05,
                                     help="Yatırma momenti kolu: yelken basınç merkezinin sudaki yanal direnç "
                                          "merkezine göre dikey yüksekliği. Yalnız devrilme uyarısında kullanılır."),
                CLa_hull=st.number_input("Gövde yanal eğim (1/rad)", 0.2, 6.0, d.CLa_hull, 0.1),
                x_mast=st.number_input("Direk konumu x (m)", -0.8, 0.8, d.x_mast, 0.01,
                                       help="Tekne ortasından ölçülür; + pruvaya doğru."),
                x_CLR=st.number_input("Yanal direnç merkezi x (m)", -0.8, 0.8, d.x_CLR, 0.01,
                                      help="Gövdelerin yanal kuvvet merkezi; tekne ortasından, + pruvaya doğru. "
                                           "Yelken basınç merkezine göre konumu orsa/boca eğilimini belirler."),
                A_rud=st.number_input("Dümen alanı (m²)", 0.005, 0.2, d.A_rud, 0.005, format="%.3f"),
                zeta_deg=st.number_input("Eq.(12) referans ζ (°)", 20.0, 70.0, d.zeta_deg, 1.0),
                boom_rigid=boom_lbl.startswith("Rijit"),
            )
            st.caption(f"T = {p.draft * 100:.1f} cm · S = {p.S_wet:.2f} m² · m22 = {p.m22:.1f} kg · "
                       f"Iz = {p.Iz:.1f} kg·m² · RM = {p.righting_moment:.0f} N·m")

    trim_auto = ("opt" if auto_method.startswith("Optimum") else True) if trim_on else False
    page_args = (sail_type, tws_kt, trim_auto, trim_manual, p)
    page.run()


@st.cache_data(show_spinner=False, max_entries=256)
def _cockpit_cached(heading, delta_r, horizon, sail_type, tws_kt, trim_auto, trim_manual, p_vals):
    """Kokpitin tüm fizik çıktıları (önbellekli; aynı girdilere dönüşte yeniden hesap yok). Fizik değişmedi."""
    p = Params(*p_vals)
    eq = solve_equilibrium(heading, tws_kt, delta_r, trim_auto, trim_manual, sail_type, p)
    path, _ = predict_path(heading, tws_kt, delta_r, trim_auto, trim_manual, sail_type, p, horizon)
    lim = nogo_limit(tws_kt, sail_type, p)
    # Yol alarak başlayan tekne rota boyunca ölü açıya girip orada TAKILIYORSA rozet bunu gösterir
    # (girip kendiliğinden çıkabiliyorsa rozet başlangıç durumunda kalır).
    trap = nogo_trap(path, tws_kt, delta_r, trim_auto, trim_manual, sail_type, p, lim) if eq["V"] > 0 else None
    return eq, path, lim, trap


def cockpit_page(heading, delta_r, horizon, sail_type, tws_kt, trim_auto, trim_manual, p):
    # ---------------- Fizik (değişmedi; önbellekli) ----------------
    eq, path, lim, trap = _cockpit_cached(heading, delta_r, horizon, sail_type, tws_kt, trim_auto, trim_manual,
                                          astuple(p))
    status, scol, twa = point_of_sail(heading, eq, lim)
    if trap is not None:
        status, scol = "ÖLÜ AÇIDA KALDI", "#C0392B"
    rel_wind = wrap180(WIND_FROM_DEG - heading)
    in_nogo = lim is None or twa < lim               # başlangıç pruvası ölü açıda mı (etiketler için)

    # Durağan ileri denge yoksa (V = 0) yönelim/hız bilgisi DİNAMİK rotadan özetlenir (fizik değişmez;
    # aynı predict_path çıktısı yalnız okunur).
    dyn = None
    if eq["V"] == 0 and len(path) > 1:
        hu = np.degrees(np.unwrap(np.radians(path[:, 4])))
        T_end = float(path[-1, 2])
        u_end, v_end = float(path[-1, 5]), float(path[-1, 6])
        moving_end = math.hypot(u_end, v_end) > 0.02
        dyn = dict(dh=float(hu[-1] - hu[0]), rate=float(hu[-1] - hu[0]) / T_end if T_end > 0 else 0.0,
                   u_mean=float(path[:, 5].mean()), u_min=float(path[:, 5].min()),
                   v_max=float(path[:, 3].max()), dist=float(math.hypot(path[-1, 0], path[-1, 1])),
                   cog_end=(path[-1, 4] + math.degrees(math.atan2(v_end, u_end))) % 360.0 if moving_end else None,
                   astern_end=moving_end and u_end < 0)
        dyn["astern"] = dyn["u_mean"] < -0.02

    # ---------------- Üst şerit: 4 kart + durum kapsülü ----------------
    badge_txt = {"ÖLÜ AÇI (No-Go Zone)": "NO-GO", "YOL YOK — trim uygun değil": "YOL YOK"}.get(status, status)
    badge_tip = (f"Gerçek rüzgar açısı TWA = {twa:.0f}°. "
                 + (f"Fiziksel ölü açı sınırı ±{lim:.1f}° (kuvvet dengesinden; Jaulin Eq.12 referansı "
                    f"±{p.zeta_deg:.0f}°). " if lim is not None else "")
                 + ("Bu pruvada itiş, direnci hiçbir trimle yenemiyor." if badge_txt == "NO-GO" else
                    "Optimum trimle yol alınabilir; mevcut trim yetersiz." if badge_txt == "YOL YOK" else ""))
    badge_sub = f"TWA {twa:.0f}°"
    if trap is not None:
        badge_tip += (f"Başlangıçta yol alıyor, ancak rota {trap:.0f}. saniyede ölü açıya giriyor ve ileri hızı "
                      f"çıkmaya yetmiyor (rota sonundan 60 s daha simüle edildi; tekne ölü açıda kaldı).")
        badge_sub = f"{trap:.0f}. s'de takıldı"
    drift_txt = "—" if eq["lam"] is None else f"{eq['lam']:+.1f}"
    v_tip = "Durağan denge hızı: yelken itişi = toplam su direnci (gövde + dümen + indüklenmiş)."
    v_sub = None
    if dyn is not None:
        if dyn["astern"]:
            v_sub = f"geri {-dyn['u_min'] / KT2MS:.2f} kt"
            v_tip += (f" Bu pruvada ileri denge yok; dinamik rotada tekne GERİ gidiyor (en çok "
                      f"{-dyn['u_min'] / KT2MS:.2f} kt).")
        elif dyn["v_max"] > 0.05:
            v_sub = f"dinamik ≤{dyn['v_max'] / KT2MS:.2f} kt"
            v_tip += (f" Bu pruvada ileri denge yok; dinamik rotada en yüksek hız "
                      f"{dyn['v_max'] / KT2MS:.2f} kt.")
    cards = "".join([
        _card("Tahmini hız", f"{eq['V'] / KT2MS:.2f}", "kt", "#38BDF8", v_tip, sub=v_sub),
        _card("İleri itiş Fx", f"{eq['Fx']:+.1f}", "N", "#22C55E",
              "Yelken + su üstü rüzgar direncinin pruva yönündeki bileşeni (gövde ekseni)."),
        _card("Yanal Fy", f"{eq['Fy']:+.1f}", "N", "#EF4444",
              "Aerodinamik yanal kuvvet. + sancağa, − iskeleye."),
        _card("Drift / sapma", drift_txt, "°", "#FBBF24",
              "Rota φ − pruva θ (Morge Eq.2'deki φ ≠ θ). Tekne dururken tanımsız."),
    ])
    badge = (f'<div class="ck-badge" style="background:{scol}" title="{_esc(badge_tip)}">'
             f'{badge_txt}<small>{badge_sub} ⓘ</small></div>')
    st.markdown(f'<div class="ck-row">{cards}{badge}</div>', unsafe_allow_html=True)

    # ---------------- Ana alan: kahraman tuval + yönelim paneli ----------------
    c_hero, c_side = st.columns([3.15, 1.2], gap="small")
    with c_hero:
        hero = st.container(key="hero")
        fig = build_cockpit_figure(heading, eq, path, sail_type, delta_r, p, lim, tws_kt, horizon)
        cfg = {"displaylogo": False, "responsive": True, "scrollZoom": True,
               "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}
        try:
            hero.plotly_chart(fig, width="stretch", height="stretch", config=cfg)
        except Exception:
            hero.plotly_chart(fig, use_container_width=True, config=cfg)

    with c_side:
        def turn_name(rate):
            if abs(rate) <= 0.05:
                return "Düz"
            return ("Sancağa ↻" if rate > 0 else "İskeleye ↺") + \
                   (" · orsa" if np.sign(rate) == np.sign(rel_wind) else " · boca")

        end_h = path[-1, 4] if len(path) > 1 else heading
        if dyn is None:                               # durağan denge var: değerler dengeden
            turn = turn_name(eq["r"]) if eq["V"] > 0 else "Düz"
            phi_row = _row("Hareket yönü φ", "—" if eq["V"] == 0 else f"{eq['course'] % 360:.1f}°",
                           "Teknenin suya göre gerçekte gittiği yön = pruva + drift", "#22D3EE")
            r_row = _row("Dönüş", f"{eq['r']:+.2f}°/s", f"{turn}. + sancağa (saat yönü).", "#E2E8F0")
            R_row = _row("Dönüş yarıçapı", "∞" if not math.isfinite(eq["radius"]) else f"{eq['radius']:.1f} m",
                         "Durağan dönüş: R = V / r")
        else:                                         # denge yok: dinamik rotanın özeti
            turn = turn_name(dyn["rate"]) + (" (dinamik)" if abs(dyn["rate"]) > 0.05 else "")
            phi_txt = "—" if dyn["cog_end"] is None else \
                f"{dyn['cog_end']:.0f}°" + (" · geri" if dyn["astern_end"] else "")
            phi_row = _row("Hareket yönü φ", phi_txt,
                           f"Durağan denge yok. Dinamik rotanın sonunda ({horizon} s) teknenin gerçekte gittiği yön."
                           + (" Tekne geri gidiyor." if dyn["astern_end"] else ""), "#22D3EE")
            r_row = _row("Dönüş (ort.)", f"{dyn['rate']:+.2f}°/s",
                         f"Durağan denge yok; dinamik rotada {horizon} s boyunca ortalama dönüş hızı "
                         f"(net {dyn['dh']:+.0f}°). {turn_name(dyn['rate'])}. + sancağa (saat yönü).", "#E2E8F0")
            R_row = _row("Dönüş yarıçapı", "—", "Durağan dönüş yok (ileri denge hızı sıfır).")
        rows = "".join([
            phi_row,
            r_row,
            R_row,
            _row("Düz rota dümeni", "—" if eq["d_bal"] is None else f"{eq['d_bal']:+.1f}°",
                 "r = 0 için gereken merkezi dümen açısı (yelken + Munk momentini dengeler)", "#60A5FA"),
            _row(f"{horizon} s sonra pruva", f"{end_h:.0f}°",
                 f"Dinamik rota sonunda net pruva değişimi {wrap180(end_h - heading):+.0f}°"),
            _row("Ölü açı sınırı", "—" if lim is None else f"±{lim:.1f}°",
                 "Kuvvet dengesinden (optimum trim, dümen 0). Eq.(12) referansı noktalı çizgide.", "#F87171"),
            _row("Yelken δs / α", f"{eq['delta_s']:+.0f}° / {eq['alpha']:.0f}°",
                 f"Uygulanan yelken açısı ve hücum açısı. Görünür rüzgar {eq['aaw'] / KT2MS:.1f} kt, "
                 f"pruvadan {eq['beta']:+.0f}° (Morge Eq.2)."),
        ])
        st.markdown(f'<div class="ck-panel"><div class="ck-h">Yönelim · {turn}</div>{rows}</div>',
                    unsafe_allow_html=True)

        # Uyarı çipleri (açıklamalar tooltip'te)
        chips = []
        if eq["capsize"]:
            chips.append(_chip("⛔ Devrilme riski", "#EF4444",
                               f"Yatırma momenti {eq['HM']:.0f} N·m > doğrultucu moment {eq['RM']:.0f} N·m "
                               "(m·g·b/2). Rüzgarüstü gövde kalkar; statik sonuç geçersiz."))
        elif eq["HM"] > 0.7 * eq["RM"]:
            chips.append(_chip(f"⚠ Yatırma %{100 * eq['HM'] / eq['RM']:.0f}", "#F59E0B",
                               "Yatırma momenti doğrultucu momente yaklaşıyor; rüzgarüstü gövde hafif."))
        if eq["Fr"] > 0.40:
            chips.append(_chip(f"⚠ Fr {eq['Fr']:.2f}", "#F59E0B",
                               "Froude > 0.40: dalga direnci hızla artar; sabit C_R kabulüyle hız iyimser."))
        if eq.get("crab"):
            chips.append(_chip("↯ Yana sürüklenme", "#EF4444",
                               "Yalnız büyük driftli yavaş denge var; dümen akışı stall'da → kontrollü seyir yok."))
        elif eq["V"] > 0 and (eq["V"] < 0.3 or abs(eq["lam"]) > 20):
            chips.append(_chip("⚠ Dümen zayıf", "#F59E0B",
                               f"V={eq['V']:.2f} m/s, drift {eq['lam']:+.0f}°: dümene gelen akışı yana kayma "
                               "belirliyor."))
        if eq["flapping"]:
            chips.append(_chip("〰 Yelken çırpıyor", "#F87171",
                               "Rüzgar ıskota sınırından daha önden geliyor; yelken akışa hizalandı (luff)."))
        elif eq["backwinded"]:
            chips.append(_chip("⇄ Ters dolmuş", "#C084FC",
                               "Rijit bumba rüzgarüstünde: kamburu ters dönen yelken tekneyi geri/yana itiyor."))
        if dyn is not None:
            twa_t = np.abs(((path[:, 4] - WIND_FROM_DEG + 180) % 360) - 180)
            esc = np.where((twa_t > (lim or 180)) & (path[:, 5] > 0.05))[0]
            tip = (f"Dinamik ({horizon} s): " + (f"geriye en çok {-dyn['u_min']:.2f} m/s, " if dyn["u_min"] < -0.02
                                                 else "") + f"başlangıçtan {dyn['dist']:.0f} m uzaklaşma. ")
            if in_nogo:                               # başlangıç ölü açıda
                if len(esc):
                    chips.append(_chip(f"↘ Pruva {path[esc[0], 2]:.0f} s'de düşüyor", "#22C55E",
                                       tip + "Pruva ölü açı sınırını geçince tekne yeniden yol alıyor."))
                else:
                    chips.append(_chip("⟲ Ölü açıda kalıyor", "#EF4444",
                                       tip + f"Pruva {horizon} s boyunca ölü açı sınırını (±{lim:.1f}°) geçemedi. "
                                             "Kurtulma: dümen ya da ters yelken manevrası." if lim is not None else
                                       tip + "Bu rüzgarda hiçbir pruvada ileri denge yok."))
            elif len(esc):                            # ölü açıda değil, trim uygunsuz; sonradan yol alıyor
                chips.append(_chip(f"↗ {path[esc[0], 2]:.0f} s'de yol alıyor", "#22C55E",
                                   tip + "Seçili trimle durağan ileri denge yok; pruva döndükçe tekne ileri "
                                         "yol almaya başlıyor."))
            elif dyn["astern"]:                       # ölü açıda değil; yelken ters tarafta → geri seyir
                chips.append(_chip("⇦ Geri seyir", "#F59E0B",
                                   tip + "Tekne ölü açıda DEĞİL; seçili yelken açısında itiş geriye. "
                                         "Optimum trimle bu pruvada ileri yol alınır."))
            else:
                chips.append(_chip("⊘ Yol alamıyor", "#F59E0B",
                                   tip + "Tekne ölü açıda değil; seçili yelken açısı ileri itiş üretmiyor."))
        if trap is not None:
            chips.append(_chip("⟲ Ölü açıda kalıyor", "#EF4444",
                               f"Dinamik: {trap:.0f}. s'de ölü açıya (±{lim:.1f}°) giriyor; ileri hız dümenle "
                               "çıkmaya yetmiyor. Kurtulma: yelkeni tamamen boşalt, pruva düşünce yeniden trim et."))
        if not chips:
            chips.append(_chip("✓ Uyarı yok", "#22C55E", "Devrilme, Froude, dümen ve yelken durumu normal."))
        st.markdown(f'<div class="ck-chips">{"".join(chips)}</div>', unsafe_allow_html=True)

        st.markdown('<div class="ck-h" style="margin-top:4px">Rota haritası (gerçek ölçek)</div>',
                    unsafe_allow_html=True)
        try:
            st.plotly_chart(build_minimap(path, heading, p, lim), width="stretch",
                            config={"displayModeBar": False, "staticPlot": True})
        except Exception:
            st.plotly_chart(build_minimap(path, heading, p, lim), use_container_width=True,
                            config={"displayModeBar": False, "staticPlot": True})

        pc1, pc2 = st.columns(2)
        with pc1.popover("Kuvvet dengesi", use_container_width=True):
            if eq["V"] > 0:
                mur = (p.mass + p.m11) * eq["V"] * math.cos(math.radians(eq["lam"])) * math.radians(eq["r"])
                st.table({
                    "Bileşen": ["Yelken + rüzgar (CE)", "Gövde", "Dümen", "Munk", "Merkezcil", "TOPLAM"],
                    "Y (N)": [f"{eq['Fy']:+.2f}", f"{eq['Y_h']:+.2f}", f"{eq['Y_r']:+.2f}", "—", f"{-mur:+.2f}",
                              f"{eq['Fy'] + eq['Y_h'] + eq['Y_r'] - mur:+.3f}"],
                    "N (N·m)": [f"{eq['N_s']:+.3f}", f"{eq['N_h']:+.3f}", f"{eq['N_r']:+.3f}",
                                f"{eq['N_munk']:+.3f}", "—",
                                f"{eq['N_s'] + eq['N_h'] + eq['N_r'] + eq['N_munk']:+.4f}"],
                })
                st.caption(f"İleri: Fx {eq['Fx']:+.1f} N ↔ direnç {eq['R']:.1f} N · Fr {eq['Fr']:.2f} · "
                           f"yatırma {eq['HM']:.0f}/{eq['RM']:.0f} N·m")
            else:
                st.caption("Tekne yol almıyor: bu pruvada durağan ileri denge yok "
                           + ("(ölü açı)." if in_nogo else "(seçili trimle itiş direnci yenemiyor).")
                           + f" Yelken kuvvetleri: Fx {eq['Fx']:+.1f} N, Fy {eq['Fy']:+.1f} N.")
        with pc2.popover("Kaynaklar", use_container_width=True):
            st.caption(
                "Morge et al. 2022 Eq.(1)-(2): görünür rüzgar. Jaulin & Le Bars 2013 Eq.(11): kardioid trim, "
                "dümen işareti; Eq.(12) ζ yalnız referans. Ölü açı kuvvet dengesinden. Fossen 3-DOF (ek kütle, "
                "Munk), şerit teorisi, ITTC-1957, Mumford ıslak alan, m·g·b/2 doğrultucu moment. VARSAYIM: "
                "CL/CD eğrileri, C_R, CD·A, gövde eğimi, direk/CLR konumu, dümen alanı. Munk momenti teorik değerinde (k = 1).")


if __name__ == "__main__":
    main()
