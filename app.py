import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Wedge

# --- SAYFA VE ARAYÜZ YAPILANDIRMASI ---
st.set_page_config(
    page_title="Yelkenli Katamaran ve Saildrone Simülatörü",
    page_icon="⛵",
    layout="wide"
)

# Profesyonel denizcilik ve kokpit teması CSS'i
st.markdown("""
<style>
    .block-container {
        padding-top: 0.7rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
        max-width: 100% !important;
    }
    h2, h3 {
        margin-top: 0rem !important;
        margin-bottom: 0.35rem !important;
        font-weight: 600 !important;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 8px 10px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .metric-val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0284c7;
    }
    .metric-lbl {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 0.4px;
    }
    .badge-full {
        background-color: #dcfce7;
        color: #15803d;
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.76rem;
    }
    .badge-warn {
        background-color: #fef9c3;
        color: #a16207;
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.76rem;
    }
    .badge-dead {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.76rem;
    }
    .polar-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
        margin-top: 6px;
    }
    .polar-table th, .polar-table td {
        border: 1px solid #e2e8f0;
        padding: 5px 8px;
        text-align: center;
    }
    .polar-table th {
        background-color: #f1f5f9;
        font-weight: 600;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# --- SENKRON GİRDİ BİLEŞENİ (SLIDER + KLAVYEDEN GİRİŞ) ---
def sync_input(label, key_prefix, min_val, max_val, default_val, step=1, help_text=""):
    slider_key = f"{key_prefix}_slider"
    num_key = f"{key_prefix}_num"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = default_val
    if num_key not in st.session_state:
        st.session_state[num_key] = default_val

    def on_slider():
        st.session_state[num_key] = st.session_state[slider_key]

    def on_num():
        val = int(st.session_state[num_key])
        val = max(min_val, min(max_val, val))
        st.session_state[num_key] = val
        st.session_state[slider_key] = val

    st.markdown(f"<span style='font-weight: 600; font-size: 0.86rem;'>{label}</span>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1.2], gap="small")
    with c1:
        st.slider(
            label=label, min_value=min_val, max_value=max_val, step=step,
            key=slider_key, on_change=on_slider, label_visibility="collapsed", help=help_text
        )
    with c2:
        st.number_input(
            label=label, min_value=min_val, max_value=max_val, step=step,
            key=num_key, on_change=on_num, label_visibility="collapsed"
        )
    return st.session_state[slider_key]

# --- 360 DERECE PUSULA VE AERODİNAMİK SEYİR POLARLARI ---
def analyze_points_of_sail_360(heading_deg, profile_name):
    """
    360 derecelik pusula seyir modeli ve aerodinamik polar analizi:
    - 330° - 030°: ÖLÜ AÇI (No-Go Zone, 60 derecelik yasak sektör).
    - 030° - 050°: Sancak Orsa (Close-Hauled, TWA 30°-50°)
    - 050° - 080°: Sancak Dar Apaz (Close Reach, TWA 50°-80°)
    - 080° - 120°: Sancak Tam Apaz (Beam Reach - En Hızlı Seyir, TWA 80°-120°)
    - 120° - 150°: Sancak Geniş Apaz (Broad Reach, TWA 120°-150°)
    - 150° - 210°: Pupa Seyri (Running, TWA 150°-180°)
    - 210° - 240°: İskele Geniş Apaz (Broad Reach)
    - 240° - 280°: İskele Tam Apaz (Beam Reach)
    - 280° - 310°: İskele Dar Apaz (Close Reach)
    - 310° - 330°: İskele Orsa (Close-Hauled)

    NACA 0012, NACA 0015 ve Kumaş Yelken gerçekçi aerodinamik polar farkları:
    * NACA 0012: İnce simetrik kanat (%12). Düşük profil sürtünmesi (C_D0 ≈ 0.008), yüksek L/D.
                 Orsa (30°-50°) ve Tam Apazda (80°-120°) en yüksek hız (1.30x).
    * NACA 0015: Kalın simetrik kanat (%15). Yüksek C_Lmax, yumuşak/gecikmeli stall (16° AoA).
                 Geniş Apazda (120°-150°) en yüksek verim ve hız (1.05x).
    * Kumaş Yelken: Esnek torlu klasik yelken. Pupa seyrinde (150°-210°) rüzgarı cepleyerek en yüksek itki (0.74x).
    """
    h = heading_deg % 360
    is_12 = "0012" in profile_name
    is_15 = "0015" in profile_name
    is_cloth = not (is_12 or is_15)

    # ÖLÜ AÇI: 330° ile 030° arası (Tam 60 derecelik sektör)
    if h <= 30 or h >= 330:
        regime = "Ölü Açı (No-Go Zone)"
        opt_sail = 0.0
        speed_ratio = 0.0
        tack = "Kör Nokta"
        wind_side = "Pruvadan (Kafadan)"
        leeward_sign = 0.0
        desc = "Rüzgar doğrudan pruvadan geliyor (330°-030° arası 60° ölü açı). Yelken rüzgar tutamaz, yapraklar. İtki sıfırdır."
    elif 30 < h <= 50:
        regime = "Sancak Orsa (Close-Hauled)"
        tack = "Sancak Kontra"
        wind_side = "İskele (Sol)"
        leeward_sign = 1.0  # Yelken sancağa açılır
        if is_12:
            opt_sail = 12.0
            speed_ratio = 0.78
            desc = "NACA 0012 rijit kanat orsası. Düşük profil sürtünmesi (C_D0 ≈ 0.008) sayesinde 30°'ye kadar rüzgar üstüne dik tırmanır."
        elif is_15:
            opt_sail = 14.0
            speed_ratio = 0.74
            desc = "NACA 0015 kalın kanat orsası. Kalın profil (%15) sağlam taşıma üretir ancak sürtünmesi 0012'den hafif yüksektir."
        else:
            opt_sail = 16.0
            speed_ratio = 0.62
            desc = "Klasik kumaş yelken orsası. Iskotalar sonuna kadar kasılı. Kumaş esnemesi nedeniyle 30° sınırında itki düşer."
    elif 50 < h <= 80:
        regime = "Sancak Dar Apaz (Close Reach)"
        tack = "Sancak Kontra"
        wind_side = "İskele (Sol)"
        leeward_sign = 1.0
        if is_12:
            opt_sail = 28.0
            speed_ratio = 0.98
            desc = "Rüzgar sol bordadan önden gelir. NACA 0012 akıcı hava akışıyla yüksek itki üretir."
        elif is_15:
            opt_sail = 30.0
            speed_ratio = 0.94
            desc = "NACA 0015 dar apazda yüksek taşıma katsayısıyla (C_L) kararlı seyir sağlar."
        else:
            opt_sail = 32.0
            speed_ratio = 0.85
            desc = "Kumaş yelken orsaya göre biraz daha boşlanır, tor alanı genişler ve hız artar."
    elif 80 < h <= 120:
        regime = "Sancak Tam Apaz (Beam Reach)"
        tack = "Sancak Kontra"
        wind_side = "İskele (Sol)"
        leeward_sign = 1.0
        if is_12:
            opt_sail = 42.0
            speed_ratio = 1.30
            desc = "TAM APAZ (BEAM REACH): Rüzgar 90° bordadan gelir. NACA 0012 minimum sürtünmesiyle rekor hız oranına (1.30x) ulaşır!"
        elif is_15:
            opt_sail = 45.0
            speed_ratio = 1.25
            desc = "TAM APAZ (BEAM REACH): NACA 0015 güçlü taşıma üretir, rüzgar hızını %25 aşar (1.25x)."
        else:
            opt_sail = 46.0
            speed_ratio = 1.15
            desc = "TAM APAZ (BEAM REACH): Kumaş yelken tam dolu ve torludur. Rüzgar hızını %15 aşar (1.15x)."
    elif 120 < h <= 150:
        regime = "Sancak Geniş Apaz (Broad Reach)"
        tack = "Sancak Kontra"
        wind_side = "İskele (Sol)"
        leeward_sign = 1.0
        if is_15:
            opt_sail = 64.0
            speed_ratio = 1.05
            desc = "GENİŞ APAZ: Rüzgar kıç omuzluktan gelir. NACA 0015 yüksek stall açısı (16°) ve yüksek C_Lmax ile geniş apazda zirve yapar!"
        elif is_12:
            opt_sail = 62.0
            speed_ratio = 0.98
            desc = "GENİŞ APAZ: NACA 0012 rüzgar kıç omuzluktan gelirken kanat dışarı boşlanır."
        else:
            opt_sail = 65.0
            speed_ratio = 0.92
            desc = "GENİŞ APAZ: Kumaş yelken iyice boşlanır, rüzgar cebi oluşarak itki sağlar."
    elif 150 < h <= 210:
        regime = "Pupa Seyri (Running)"
        tack = "Pupa"
        wind_side = "Kıçtan (Arkadan)"
        leeward_sign = 1.0 if h <= 180 else -1.0
        if is_cloth:
            opt_sail = 82.0
            speed_ratio = 0.74
            desc = "PUPA SEYRİ: Rüzgar doğrudan arkadan gelir. Kumaş yelken rüzgarı paraşüt gibi doldurup torlayarak en yüksek itme sürüklemesini üretir."
        elif is_15:
            opt_sail = 80.0
            speed_ratio = 0.65
            desc = "PUPA SEYRİ: NACA 0015 kalın kesiti (%15) sayesinde arkadan gelen rüzgara 0012'den daha fazla yüzey direnci gösterir."
        else:
            opt_sail = 78.0
            speed_ratio = 0.58
            desc = "PUPA SEYRİ: NACA 0012 ince simetrik yapısı (%12) nedeniyle havayı yarmaya meyillidir, pupa itkisi düşüktür."
    elif 210 < h <= 240:
        regime = "İskele Geniş Apaz (Broad Reach)"
        tack = "İskele Kontra"
        wind_side = "Sancak (Sağ)"
        leeward_sign = -1.0  # Yelken iskeleye açılır
        if is_15:
            opt_sail = 64.0
            speed_ratio = 1.05
            desc = "İSKELE GENİŞ APAZ: NACA 0015 yüksek taşıma ile iskelede de zirve performans gösterir."
        elif is_12:
            opt_sail = 62.0
            speed_ratio = 0.98
            desc = "İSKELE GENİŞ APAZ: NACA 0012 iskeleye boşlanır."
        else:
            opt_sail = 65.0
            speed_ratio = 0.92
            desc = "İSKELE GENİŞ APAZ: Kumaş yelken iskeleye iyice boşlanır."
    elif 240 < h <= 280:
        regime = "İskele Tam Apaz (Beam Reach)"
        tack = "İskele Kontra"
        wind_side = "Sancak (Sağ)"
        leeward_sign = -1.0
        if is_12:
            opt_sail = 42.0
            speed_ratio = 1.30
            desc = "İSKELE TAM APAZ: NACA 0012 rekor sürat oranına (1.30x) ulaşır."
        elif is_15:
            opt_sail = 45.0
            speed_ratio = 1.25
            desc = "İSKELE TAM APAZ: NACA 0015 yüksek itki ile 1.25x hız oranına ulaşır."
        else:
            opt_sail = 46.0
            speed_ratio = 1.15
            desc = "İSKELE TAM APAZ: Kumaş yelken rüzgarı sağdan 90° bordadan alır (1.15x)."
    elif 280 < h <= 310:
        regime = "İskele Dar Apaz (Close Reach)"
        tack = "İskele Kontra"
        wind_side = "Sancak (Sağ)"
        leeward_sign = -1.0
        if is_12:
            opt_sail = 28.0
            speed_ratio = 0.98
            desc = "İSKELE DAR APAZ: NACA 0012 akıcı itki üretir."
        elif is_15:
            opt_sail = 30.0
            speed_ratio = 0.94
            desc = "İSKELE DAR APAZ: NACA 0015 dengeli tırmanış sağlar."
        else:
            opt_sail = 32.0
            speed_ratio = 0.85
            desc = "İSKELE DAR APAZ: Kumaş yelken iskeleye boşlanır."
    else:  # 310 < h < 330
        regime = "İskele Orsa (Close-Hauled)"
        tack = "İskele Kontra"
        wind_side = "Sancak (Sağ)"
        leeward_sign = -1.0
        if is_12:
            opt_sail = 12.0
            speed_ratio = 0.78
            desc = "İSKELE ORSA: NACA 0012 330° sınırına kadar dik ve hızlı tırmanır."
        elif is_15:
            opt_sail = 14.0
            speed_ratio = 0.74
            desc = "İSKELE ORSA: NACA 0015 güçlü taşıma üretir."
        else:
            opt_sail = 16.0
            speed_ratio = 0.62
            desc = "İSKELE ORSA: Kumaş yelken kasılı, 330° sınırında seyreder."

    return regime, opt_sail, speed_ratio, tack, wind_side, leeward_sign, desc

# --- SİMÜLASYON VE ÇİZİM ÇEKİRDEĞİ ---
def simulate_catamaran(heading_deg, sail_input_deg, rudder_deg, profile_name, wind_speed_ms, sim_time_s, auto_trim=False):
    """
    330°-030° Ölü Açı sektörü, NACA 0012 / NACA 0015 / Kumaş Yelken fiziksel polarları,
    ters işleyen dümen kinematiği ve merkezi katamaran görselleştirmesi.
    """
    is_12 = "0012" in profile_name
    is_15 = "0015" in profile_name
    is_cloth = not (is_12 or is_15)

    # 1. BAŞLANGIÇ SEYİR DURUMU (360 Derece Analiz):
    regime, opt_sail, max_speed_ratio, tack, wind_side, leeward_sign, desc = analyze_points_of_sail_360(heading_deg, profile_name)
    is_in_irons = (regime == "Ölü Açı (No-Go Zone)")

    # Otomatik trim seçeneği veya kullanıcının ayarı
    if auto_trim and not is_in_irons:
        sail_boom_angle = opt_sail
    else:
        sail_boom_angle = abs(sail_input_deg)

    # 2. YELKEN / KANAT TRİM VE AERODİNAMİK VERİM DEĞERLENDİRMESİ:
    if is_in_irons:
        trim_efficiency = 0.0
        trim_status = "Yelken Dolmuyor (Yapraklıyor)"
        camber_amount = 0.0
        badge_type = "badge-dead"
    else:
        angle_diff = abs(sail_boom_angle - opt_sail)
        if is_12:
            # NACA 0012: İnce simetrik profil, keskin ve dar stall aralığı (±4°)
            if angle_diff <= 4.0:
                trim_efficiency = 1.0 - (angle_diff / 4.0) * 0.04
                trim_status = "Tam Trim (NACA 0012 Optimum)"
                badge_type = "badge-full"
            elif sail_boom_angle < opt_sail:
                trim_efficiency = max(0.08, 1.0 - ((opt_sail - sail_boom_angle) / 20.0) ** 1.9)
                trim_status = "Fazla Kasılı (Erken Stall)"
                badge_type = "badge-warn"
            else:
                trim_efficiency = max(0.06, 1.0 - ((sail_boom_angle - opt_sail) / 22.0) ** 1.9)
                trim_status = "Fazla Açık (Kanat Nötr)"
                badge_type = "badge-warn"
            camber_amount = 0.0
        elif is_15:
            # NACA 0015: Kalın profil (%15), daha geniş hücum toleransı (±6°) ve yumuşak stall
            if angle_diff <= 6.0:
                trim_efficiency = 1.0 - (angle_diff / 6.0) * 0.04
                trim_status = "Tam Trim (NACA 0015 Optimum)"
                badge_type = "badge-full"
            elif sail_boom_angle < opt_sail:
                trim_efficiency = max(0.12, 1.0 - ((opt_sail - sail_boom_angle) / 24.0) ** 1.7)
                trim_status = "Fazla Kasılı (Stall)"
                badge_type = "badge-warn"
            else:
                trim_efficiency = max(0.10, 1.0 - ((sail_boom_angle - opt_sail) / 26.0) ** 1.7)
                trim_status = "Fazla Açık (Kanat Nötr)"
                badge_type = "badge-warn"
            camber_amount = 0.0
        else:
            # Kumaş Yelken: Esnek torlu klasik yelken formu
            if angle_diff <= 6.0:
                trim_efficiency = 1.0 - (angle_diff / 6.0) * 0.05
                trim_status = "Tam Trim (Yelken Dolu / Torlu)"
                camber_amount = 0.45
                badge_type = "badge-full"
            elif sail_boom_angle < opt_sail:
                trim_efficiency = max(0.15, 1.0 - ((opt_sail - sail_boom_angle) / 25.0) ** 1.6)
                trim_status = "Fazla Kasılı (Düz / Stall)"
                camber_amount = 0.18
                badge_type = "badge-warn"
            else:
                trim_efficiency = max(0.10, 1.0 - ((sail_boom_angle - opt_sail) / 30.0) ** 1.6)
                trim_status = "Fazla Açık (Yapraklama)"
                camber_amount = 0.22
                badge_type = "badge-warn"

    # 3. DÜMEN VE FRENLEME ETKİSİ:
    rudder_clamped = np.clip(rudder_deg, -35.0, 35.0)
    rudder_penalty = 1.0 - ((abs(rudder_clamped) / 35.0) ** 2) * 0.22

    # Anlık başlangıç hızı
    base_speed_ms = wind_speed_ms * max_speed_ratio * trim_efficiency * rudder_penalty
    base_speed_kts = base_speed_ms * 1.94384

    # 4. SİMÜLASYON YOLU VE DİNAMİK YELKEN-DÜMEN BAĞLANTISI:
    L_hull = 4.6
    dt = 0.25
    steps = int(sim_time_s / dt)
    traj_x = [0.0]
    traj_y = [0.0]
    cur_psi_deg = float(heading_deg)
    distances = []

    for _ in range(steps):
        s_reg, s_opt, s_ratio, _, _, s_leeward, _ = analyze_points_of_sail_360(cur_psi_deg, profile_name)
        if s_reg == "Ölü Açı (No-Go Zone)":
            step_v = 0.0
        else:
            s_diff = abs(sail_boom_angle - s_opt)
            if is_12:
                s_eff = max(0.08, 1.0 - (s_diff / 20.0) ** 1.9) if s_diff <= 20.0 else 0.06
            elif is_15:
                s_eff = max(0.12, 1.0 - (s_diff / 24.0) ** 1.7) if s_diff <= 24.0 else 0.08
            else:
                s_eff = max(0.10, 1.0 - (s_diff / 28.0) ** 1.8) if s_diff <= 28.0 else 0.08
            step_v = wind_speed_ms * s_ratio * s_eff * rudder_penalty

        if step_v > 0.2:
            yaw_rate_dps = (step_v / L_hull) * np.sin(np.radians(rudder_clamped)) * (180.0 / np.pi)
            if s_leeward != 0.0 and abs(rudder_clamped) < 2.0:
                weather_helm = -s_leeward * 0.35
                yaw_rate_dps += weather_helm
        else:
            yaw_rate_dps = 0.0

        cur_psi_deg = (cur_psi_deg + yaw_rate_dps * dt) % 360.0
        cur_psi_rad = np.radians(cur_psi_deg)
        dx = step_v * np.sin(cur_psi_rad) * dt
        dy = step_v * np.cos(cur_psi_rad) * dt
        traj_x.append(traj_x[-1] + dx)
        traj_y.append(traj_y[-1] + dy)
        distances.append(step_v * dt)

    total_distance_m = sum(distances)
    final_cog_deg = cur_psi_deg

    # --- MATPLOTLIB ÇİZİMİ ---
    fig, ax = plt.subplots(figsize=(7.8, 7.8), dpi=110)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')

    # 1. ÖLÜ AÇI (NO-GO ZONE) 330° - 030° SEKTÖRÜ (60 DERECE):
    # Matematiksel açılar: 030° pusula = 60° mat, 330° pusula = 120° mat
    no_go_wedge = Wedge((0, 0), 8.5, 60, 120, facecolor='#fee2e2', edgecolor='#ef4444', 
                        alpha=0.32, linestyle='--', lw=1.2, zorder=1)
    ax.add_patch(no_go_wedge)
    ax.text(0, 5.5, "ÖLÜ AÇI (NO-GO ZONE)\n330° - 030° (Rüzgara Karşı)", 
            ha='center', va='center', color='#b91c1c', fontweight='bold', fontsize=8.2, zorder=2)

    # 2. Dairesel Pusula Kılavuzları (2m, 4m, 6m, 8m)
    for r in [2, 4, 6, 8]:
        circle = plt.Circle((0, 0), r, color='#e2e8f0', fill=False, linestyle=':', linewidth=0.8, zorder=1)
        ax.add_patch(circle)

    # 3. Pusula Yön İmleri
    ax.text(0, 8.4, "000° KUZEY", ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#1e293b')
    ax.text(8.4, 0, "090° DOĞU", ha='left', va='center', fontsize=8.8, fontweight='bold', color='#1e293b')
    ax.text(0, -8.4, "180° GÜNEY", ha='center', va='top', fontsize=8.8, fontweight='bold', color='#1e293b')
    ax.text(-8.4, 0, "270° BATI", ha='right', va='center', fontsize=8.8, fontweight='bold', color='#1e293b')

    # Ölü açı sınırları (030° ve 330°)
    ax.text(4.3, 7.3, "030°", ha='left', va='bottom', fontsize=8.2, fontweight='bold', color='#dc2626')
    ax.text(-4.3, 7.3, "330°", ha='right', va='bottom', fontsize=8.2, fontweight='bold', color='#dc2626')
    ax.text(6.0, -6.0, "135°", ha='center', va='center', fontsize=7.8, color='#64748b')
    ax.text(-6.0, -6.0, "225°", ha='center', va='center', fontsize=7.8, color='#64748b')

    # 4. SABİT RÜZGAR AKIŞ OKLARI (Yukarıdan Aşağıya ↓↓↓)
    for wx in np.linspace(-6.8, 6.8, 7):
        ax.annotate(
            '', xy=(wx, 6.2), xytext=(wx, 8.2),
            arrowprops=dict(arrowstyle="->", color='#0284c7', lw=1.8, alpha=0.75),
            zorder=3
        )
    ax.text(0, 7.4, f"SABİT RÜZGAR: {wind_speed_ms:.1f} m/s ({wind_speed_ms*1.944:.1f} kts) ↓↓↓", 
            ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0369a1',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#e0f2fe', edgecolor='#38bdf8', alpha=0.95),
            zorder=4)

    # 5. ROTASYON VE KATAMARAN GÖVDESİ:
    init_psi_rad = np.radians(heading_deg)
    cos_p = np.cos(init_psi_rad)
    sin_p = np.sin(init_psi_rad)

    def rot(lx, ly):
        xw = lx * cos_p + ly * sin_p
        yw = -lx * sin_p + ly * cos_p
        return xw, yw

    hull_half_beam = 1.35
    def create_hull(cx):
        pts_local = [
            (cx, 2.2),             # Pruva sivri burun
            (cx + 0.22, 1.3),      # Omuzluk dış
            (cx + 0.22, -1.5),     # Bordası dış
            (cx + 0.12, -2.1),     # Kıç dış
            (cx - 0.12, -2.1),     # Kıç iç
            (cx - 0.22, -1.5),     # Bordası iç
            (cx - 0.22, 1.3),      # Omuzluk iç
        ]
        pts_w = [rot(px, py) for px, py in pts_local]
        return patches.Polygon(pts_w, closed=True, facecolor='#1e293b', edgecolor='#0f172a', lw=1.5, zorder=6)

    ax.add_patch(create_hull(-hull_half_beam))
    ax.add_patch(create_hull(hull_half_beam))

    # Kirişler ve Trambolin
    for beam_y in [1.3, -1.4]:
        p1 = rot(-hull_half_beam, beam_y)
        p2 = rot(hull_half_beam, beam_y)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#475569', lw=3.8, solid_capstyle='round', zorder=5)

    tramp_local = [
        (-hull_half_beam + 0.25, 1.25),
        (hull_half_beam - 0.25, 1.25),
        (hull_half_beam - 0.25, -1.35),
        (-hull_half_beam + 0.25, -1.35)
    ]
    ax.add_patch(patches.Polygon([rot(px, py) for px, py in tramp_local], closed=True, 
                                 facecolor='#cbd5e1', edgecolor='#94a3b8', linestyle='--', lw=1, alpha=0.5, zorder=4))

    # Direk (Mast)
    mast_w = rot(0, 0.2)
    ax.plot(mast_w[0], mast_w[1], 'o', color='#f59e0b', markersize=8, markeredgecolor='#b45309', zorder=10)

    # 6. DÜMENLER (TERS İŞLEYEN DENİZCİ DÜMENİ):
    # Kullanıcı +rudder girdiğinde dümenci yekeyi sola basar, pala sağa açılır (+), tekne sağa döner!
    rudder_len = 0.95
    rudder_rad_local = np.radians(rudder_clamped)

    for hull_lx in [-hull_half_beam, hull_half_beam]:
        hinge_w = rot(hull_lx, -2.1)
        tip_lx = hull_lx + rudder_len * np.sin(rudder_rad_local)
        tip_ly = -2.1 - rudder_len * np.cos(rudder_rad_local)
        tip_w = rot(tip_lx, tip_ly)
        ax.plot([hinge_w[0], tip_w[0]], [hinge_w[1], tip_w[1]], color='#dc2626', lw=3.8, solid_capstyle='round', zorder=8)
        ax.plot(hinge_w[0], hinge_w[1], 'o', color='#991b1b', markersize=4, zorder=9)

    # 7. YELKEN / KANAT GEOMETRİSİ:
    boom_len = 2.85
    actual_sail_rot = leeward_sign * np.radians(sail_boom_angle)
    u_chord = np.array([np.sin(actual_sail_rot), -np.cos(actual_sail_rot)])
    u_norm = np.array([np.cos(actual_sail_rot), np.sin(actual_sail_rot)])

    tip_lx = boom_len * u_chord[0]
    tip_ly = 0.2 + boom_len * u_chord[1]
    tip_w = rot(tip_lx, tip_ly)

    if is_in_irons:
        if is_cloth:
            num_w = 20
            wt = np.linspace(0, 1, num_w)
            wave_lx = wt * tip_lx + 0.15 * np.sin(4 * np.pi * wt)
            wave_ly = 0.2 + wt * (tip_ly - 0.2)
            wave_pts = [rot(lx, ly) for lx, ly in zip(wave_lx, wave_ly)]
            ax.plot([p[0] for p in wave_pts], [p[1] for p in wave_pts], color='#ef4444', linestyle='--', lw=2.2, zorder=9)
            ax.text(tip_w[0], tip_w[1] - 0.3, "Yapraklıyor\n(Dolmuyor)", fontsize=7.8, color='#b91c1c', 
                    fontweight='bold', ha='center', zorder=11)
        else:
            # Kanat rüzgar içine firar eder (feathers / nötr kanat)
            ax.plot([mast_w[0], tip_w[0]], [mast_w[1], tip_w[1]], color='#ef4444', linestyle='--', lw=2.2, zorder=9)
            ax.text(tip_w[0], tip_w[1] - 0.3, "Kanat Nötr\n(Taşıma Yok)", fontsize=7.8, color='#b91c1c', 
                    fontweight='bold', ha='center', zorder=11)
    else:
        if is_cloth:
            # Kumaş yelken: Torlu eğri
            s_t = np.linspace(0, 1, 25)
            b_dx = tip_lx - 0.0
            b_dy = tip_ly - 0.2
            b_len = np.hypot(b_dx, b_dy)
            norm_x = -b_dy / b_len * leeward_sign
            norm_y =  b_dx / b_len * leeward_sign

            camber_disp = camber_amount * np.sin(np.pi * s_t)
            curve_lx = s_t * b_dx + camber_disp * norm_x
            curve_ly = 0.2 + s_t * b_dy + camber_disp * norm_y

            curve_w = [rot(lx, ly) for lx, ly in zip(curve_lx, curve_ly)]
            cx_w = [p[0] for p in curve_w]
            cy_w = [p[1] for p in curve_w]

            ax.plot([mast_w[0], tip_w[0]], [mast_w[1], tip_w[1]], color='#475569', lw=2.2, zorder=8)
            ax.plot(cx_w, cy_w, color='#2563eb', lw=3.8, solid_capstyle='round', zorder=9)
            fill_poly_w = [mast_w] + curve_w + [tip_w]
            ax.add_patch(patches.Polygon(fill_poly_w, closed=True, facecolor='#2563eb', alpha=0.22, zorder=7))
        else:
            # NACA 0012 veya NACA 0015 Rijit Kanat Profili
            t_ratio = 0.12 if is_12 else 0.15
            n_pts = 40
            x_norm = np.linspace(0, 1, n_pts)
            yt = 5.0 * t_ratio * (
                0.2969 * np.sqrt(x_norm) - 0.1260 * x_norm - 0.3516 * (x_norm**2) + 0.2843 * (x_norm**3) - 0.1036 * (x_norm**4)
            ) * boom_len

            upper_lx = [x_i * boom_len * u_chord[0] + y_i * u_norm[0] for x_i, y_i in zip(x_norm, yt)]
            upper_ly = [0.2 + x_i * boom_len * u_chord[1] + y_i * u_norm[1] for x_i, y_i in zip(x_norm, yt)]
            lower_lx = [x_i * boom_len * u_chord[0] - y_i * u_norm[0] for x_i, y_i in zip(x_norm[::-1], yt[::-1])]
            lower_ly = [0.2 + x_i * boom_len * u_chord[1] - y_i * u_norm[1] for x_i, y_i in zip(x_norm[::-1], yt[::-1])]

            poly_lx = upper_lx + lower_lx
            poly_ly = upper_ly + lower_ly
            poly_w = [rot(lx, ly) for lx, ly in zip(poly_lx, poly_ly)]

            foil_color = '#0284c7' if is_12 else '#4338ca'
            foil_edge = '#0369a1' if is_12 else '#312e81'

            ax.add_patch(patches.Polygon(poly_w, closed=True, facecolor=foil_color, edgecolor=foil_edge, lw=2.0, alpha=0.6, zorder=9))
            ax.plot([mast_w[0], tip_w[0]], [mast_w[1], tip_w[1]], color='#0f172a', lw=1.8, linestyle='-', zorder=10)

            # Aerodinamik kaburga / istasyon çizgileri (Ribs)
            for frac in [0.25, 0.50, 0.75]:
                idx = int(frac * (n_pts - 1))
                p_up = rot(upper_lx[idx], upper_ly[idx])
                p_dn = rot(lower_lx[-(idx+1)], lower_ly[-(idx+1)])
                ax.plot([p_up[0], p_dn[0]], [p_up[1], p_dn[1]], color='#0f172a', lw=1.0, alpha=0.7, zorder=10)

    # 8. TEKNENİN GİTTİĞİ DOĞRULTU VE SİMÜLASYON YOLU:
    if total_distance_m > 1.0 and not is_in_irons:
        max_dist_to_show = max(total_distance_m, 1.0)
        scale_factor = min(1.0, 7.2 / max_dist_to_show)
        scaled_tx = [x * scale_factor for x in traj_x]
        scaled_ty = [y * scale_factor for y in traj_y]

        # Rota eğrisi / çizgisi
        ax.plot(scaled_tx, scaled_ty, color='#10b981', lw=3.8, linestyle='-', zorder=12)
        
        # 5 saniyelik zaman işaretçileri
        step_5s = max(1, int(5.0 / dt))
        for idx in range(step_5s, len(scaled_tx), step_5s):
            ax.plot(scaled_tx[idx], scaled_ty[idx], 'o', color='#059669', markersize=3.8, zorder=13)

        # Varış noktası oku ve etiketi
        fx, fy = scaled_tx[-1], scaled_ty[-1]
        ax.annotate(
            '', xy=(fx, fy), xytext=(scaled_tx[-2], scaled_ty[-2]),
            arrowprops=dict(arrowstyle="-|>", color='#10b981', lw=3.8, mutation_scale=22),
            zorder=14
        )
        label_x = fx * 1.15
        label_y = fy * 1.15
        ax.text(label_x, label_y, 
                f"GİDİLEN YOL: {total_distance_m:.1f} m\nRota: {final_cog_deg:.0f}° | {base_speed_kts:.1f} kts\n({sim_time_s} sn)",
                fontsize=8.0, fontweight='bold', color='#065f46', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#d1fae5', edgecolor='#10b981', alpha=0.95),
                zorder=15)
    else:
        ax.text(0, 0, "[!] ÖLÜ AÇI / İTKİ YOK\nYelken Dolmuyor (Kör Nokta)", color='#b91c1c', fontsize=8.5, fontweight='bold',
                ha='center', va='center', bbox=dict(boxstyle='round,pad=0.35', facecolor='#fee2e2', edgecolor='#ef4444'),
                zorder=15)

    # Pruva Referans Çizgisi (Kırmızı pruva hattı)
    bow_ref = rot(0, 4.4)
    ax.plot([0, bow_ref[0]], [0, bow_ref[1]], color='#ef4444', linestyle='-', lw=1.5, zorder=5, label='Pruva Hattı')

    ax.set_xlim(-9, 9)
    ax.set_ylim(-9, 9)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
    ax.set_xlabel("Doğu / Batı Mesafesi (m)", fontsize=8.5, color='#475569')
    ax.set_ylabel("Kuzey / Güney Mesafesi (m)", fontsize=8.5, color='#475569')
    ax.tick_params(colors='#64748b', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')

    plt.tight_layout()

    # Dümen yönü açıklaması
    if rudder_clamped > 0:
        rudder_info = f"Sağa Dönüş (+{rudder_clamped:.0f}° Sancak)"
    elif rudder_clamped < 0:
        rudder_info = f"Sola Dönüş ({rudder_clamped:.0f}° İskele)"
    else:
        rudder_info = "Düz Rota (0°)"

    telemetry = {
        "speed_kts": base_speed_kts,
        "speed_ms": base_speed_ms,
        "distance_m": total_distance_m,
        "regime": regime,
        "regime_desc": desc,
        "opt_sail": opt_sail,
        "actual_sail": sail_boom_angle,
        "heading_deg": heading_deg,
        "tack": tack,
        "wind_side": wind_side,
        "trim_status": trim_status,
        "trim_eff": trim_efficiency * 100.0,
        "badge_type": badge_type,
        "final_cog_deg": final_cog_deg,
        "rudder_info": rudder_info,
        "rudder_deg": rudder_clamped,
        "profile_name": profile_name
    }

    return fig, telemetry

# --- STREAMLIT ARAYÜZÜ ---
st.markdown("## ⛵ Otonom Yelkenli Katamaran ve Saildrone Simülatörü")
st.caption("360° Pusula Seyirleri • 330°-030° Ölü Açı • NACA 0012 / 0015 / Kumaş Yelken Aerodinamik Polarları • Realist Dümen")

col_ctrl, col_graph = st.columns([1.1, 2.0], gap="large")

with col_ctrl:
    st.markdown("### ⚙️ Kontrol Paneli")
    
    # 1. Yelken Profili
    naca_profile = st.selectbox(
        "Yelken / Kanat Profili",
        ["NACA 0012 (Sert Kanat / İnce Simetrik)", "NACA 0015 (Sert Kanat / Kalın Profil)", "Kumaş Yelken (Cambered / Klasik)"],
        index=0,
        help="NACA 0012: İnce simetrik (%12), orsa ve tam apazda en hızlı (1.30x). NACA 0015: Kalın profil (%15), geniş apazda zirve (1.05x). Kumaş Yelken: Esnek torlu, pupa seyrinde en yüksek itki (0.74x)."
    )

    # 2. Tekne Yönü (0 - 360°)
    boat_heading_val = sync_input(
        label="🧭 Tekne Yönü (Pruva Açısı: 0° - 360°)",
        key_prefix="heading",
        min_val=0,
        max_val=360,
        default_val=90,
        step=1,
        help_text="330°-030°: Ölü Açı (60° No-Go), 030°-050°: Orsa, 050°-080°: Dar Apaz, 090°: Tam Apaz (En Hızlı), 135°: Geniş Apaz, 180°: Pupa, 270°: İskele Tam Apaz."
    )

    # 3. Yelken Açısı ve Otomatik Trim
    col_s1, col_s2 = st.columns([2.5, 1.5])
    with col_s1:
        sail_angle_val = sync_input(
            label="📐 Yelken Bom Açısı (0° - 90°)",
            key_prefix="sail",
            min_val=0,
            max_val=90,
            default_val=42 if "0012" in naca_profile else (45 if "0015" in naca_profile else 46),
            step=1,
            help_text="Iskota/Kanat trimi: Bomun orta hattan açısı (0°: Kasılı, 12°-16°: Orsa, 42°-46°: Tam Apaz, 62°-65°: Geniş Apaz, 78°-82°: Pupa)."
        )
    with col_s2:
        st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
        auto_trim_active = st.checkbox("🤖 Auto-Trim", value=False, help="Saildrone otomatik trim modu: Seçili kanat profiline ve anlık rüzgar açısına göre en ideal trim açısına otomatik ayarlar.")

    # 4. Dümen Açısı (Realist ±35°)
    rudder_angle_val = sync_input(
        label="🕹️ Dümen Açısı (-35° İskele / +35° Sancak)",
        key_prefix="rudder",
        min_val=-35,
        max_val=35,
        default_val=0,
        step=1,
        help_text="Pozitif (+): Yeke sola basılır, pala sağa açılır, tekne sağa (sancağa) döner. Negatif (-): Pala sola açılır, tekne sola döner."
    )

    # 5. Rüzgar Hızı
    wind_speed_val = sync_input(
        label="💨 Sabit Rüzgar Hızı (m/s)",
        key_prefix="wind",
        min_val=1,
        max_val=25,
        default_val=8,
        step=1,
        help_text="Sabit kuzey rüzgarı (8 m/s ≈ 15.5 knot)."
    )

    # 6. Simülasyon Süresi
    sim_time_val = sync_input(
        label="⏱️ Simülasyon Süresi (sn)",
        key_prefix="sim_time",
        min_val=5,
        max_val=60,
        default_val=30,
        step=1,
        help_text="Teknenin denizde seyir ve dönüş yapacağı toplam süre."
    )

    st.markdown("---")
    st.info("💡 **Girdi Kolaylığı:** Değerleri ister kaydırıcıyla sürükleyin, ister yanındaki kutucuğa tıklayıp klavyeden doğrudan yazın.")

    with st.expander("📚 Profil Karşılaştırma & Aerodinamik Tablosu"):
        st.markdown("""
        <table class="polar-table">
            <thead>
                <tr>
                    <th>Profil</th>
                    <th>Orsa (35°)</th>
                    <th>Tam Apaz (90°)</th>
                    <th>Geniş Apaz (135°)</th>
                    <th>Pupa (180°)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><b>NACA 0012</b></td>
                    <td style="color:#16a34a;"><b>0.78x (Zirve)</b></td>
                    <td style="color:#16a34a;"><b>1.30x (Zirve)</b></td>
                    <td>0.98x</td>
                    <td>0.58x</td>
                </tr>
                <tr>
                    <td><b>NACA 0015</b></td>
                    <td>0.74x</td>
                    <td>1.25x</td>
                    <td style="color:#16a34a;"><b>1.05x (Zirve)</b></td>
                    <td>0.65x</td>
                </tr>
                <tr>
                    <td><b>Kumaş Yelken</b></td>
                    <td>0.62x</td>
                    <td>1.15x</td>
                    <td>0.92x</td>
                    <td style="color:#16a34a;"><b>0.74x (Zirve)</b></td>
                </tr>
            </tbody>
        </table>
        <div style="font-size:0.75rem; color:#64748b; margin-top:5px;">
        * Ölü Açı: 330° - 030° (60° No-Go Zone). Tüm hızlar rüzgar hızının katı (V_b / V_w) olarak hesaplanır.
        </div>
        """, unsafe_allow_html=True)

with col_graph:
    st.markdown("### 📊 Katamaran Görselleştirmesi ve Gidilen Yol")
    
    fig, tel = simulate_catamaran(
        heading_deg=boat_heading_val,
        sail_input_deg=sail_angle_val,
        rudder_deg=rudder_angle_val,
        profile_name=naca_profile,
        wind_speed_ms=float(wind_speed_val),
        sim_time_s=float(sim_time_val),
        auto_trim=auto_trim_active
    )

    st.pyplot(fig, width='stretch')

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">TEKNE HIZI</div>
            <div class="metric-val">{tel['speed_kts']:.1f} kts</div>
            <div style="font-size:0.75rem; color:#64748b;">{tel['speed_ms']:.1f} m/s</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">GİDİLEN MESAFE</div>
            <div class="metric-val">{tel['distance_m']:.1f} m</div>
            <div style="font-size:0.75rem; color:#64748b;">{sim_time_val} sn sürede</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">360° SEYİR TİPİ</div>
            <div style="font-size:0.92rem; font-weight:700; color:#0f172a; margin-top:3px;">{tel['regime']}</div>
            <div style="font-size:0.72rem; color:#64748b;">Pruva: {tel['heading_deg']}° | {tel['wind_side']}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        badge_html = f'<span class="{tel["badge_type"]}">{tel["trim_status"]}</span>'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">TRİM VE DÜMEN DURUMU</div>
            <div style="margin-top:2px;">{badge_html}</div>
            <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">{tel['rudder_info']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"**ℹ️ Seyir & Aerodinamik Durumu:** {tel['regime_desc']}")
