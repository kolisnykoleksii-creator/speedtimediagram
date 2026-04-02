import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import FuncFormatter, MultipleLocator
import io

# Налаштування сторінки
st.set_page_config(page_title="Тягові розрахунки", layout="wide")

# Ініціалізація стану для збереження результатів
if 'calculated' not in st.session_state:
    st.session_state.calculated = False

# ==========================================
# 1. БАЗИ ДАНИХ ЛОКОМОТИВІВ
# ==========================================
# Для пасажирських розширено шкалу швидкості до 160 км/год
LOCOMOTIVES = {
    "ТГ16 (За замовчуванням)": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120],
        "f": [400, 380, 300, 240, 190, 150, 120, 100, 80, 60, 40, 20],
        "mass": 120.0
    },
    "ВЛ80с": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120],
        "f": [650, 650, 630, 580, 480, 380, 310, 260, 220, 190, 160, 110],
        "mass": 192.0
    },
    "ВЛ80т": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120],
        "f": [640, 640, 620, 570, 475, 375, 305, 255, 215, 185, 155, 105],
        "mass": 192.0
    },
    "ВЛ10 (Вантажний, пост. струм)": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120],
        "f": [650, 650, 600, 500, 390, 310, 240, 190, 150, 120, 100, 70],
        "mass": 184.0
    },
    "ВЛ8 (Вантажний, пост. струм)": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120],
        "f": [640, 640, 600, 480, 380, 300, 230, 180, 140, 110, 90, 60],
        "mass": 184.0
    },
    "ВЛ11 (Вантажний, пост. струм)": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120],
        "f": [650, 650, 610, 510, 400, 320, 250, 195, 155, 125, 100, 70],
        "mass": 184.0
    },
    "ВЛ11м5 (Вантажний, пост. струм)": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120],
        "f": [670, 670, 630, 530, 420, 340, 270, 210, 165, 135, 110, 80],
        "mass": 184.0
    },
    "ДЕ1 (Вантажний, пост. струм)": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120],
        "f": [700, 700, 680, 550, 450, 350, 280, 220, 170, 130, 100, 75],
        "mass": 184.0
    },
    "ДС3 (Пасажирський, змін. струм)": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160],
        "f": [320, 320, 310, 300, 280, 240, 200, 170, 145, 125, 110, 85, 65, 50],
        "mass": 84.0
    },
    "2ЕЛ5 (Вантажний, змін. струм)": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120],
        "f": [700, 700, 680, 600, 500, 400, 330, 270, 220, 180, 150, 110],
        "mass": 192.0
    },
    "2ТЕ116": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120],
        "f": [500, 480, 420, 320, 240, 190, 150, 120, 100, 85, 70, 50],
        "mass": 276.0
    },
    "2ТЕ10М": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120],
        "f": [520, 500, 430, 330, 250, 200, 160, 130, 105, 90, 75, 55],
        "mass": 276.0
    },
    "ТЕП70 (Пасажирський)": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160],
        "f": [320, 300, 260, 220, 180, 150, 125, 105, 90, 75, 65, 50, 40, 30],
        "mass": 135.0
    },
    "ЧС4 (Пасажирський, змін. струм)": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160],
        "f": [300, 290, 270, 240, 200, 170, 140, 120, 100, 85, 75, 60, 45, 35],
        "mass": 123.0
    },
    "ЧС2 (Пасажирський, пост. струм)": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160],
        "f": [320, 320, 310, 270, 220, 180, 140, 110, 90, 75, 60, 45, 30, 20],
        "mass": 120.0
    },
    "ЧС7 (Пасажирський, пост. струм)": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160],
        "f": [450, 450, 420, 360, 290, 240, 200, 160, 130, 110, 95, 75, 55, 40],
        "mass": 172.0
    }
}

# ==========================================
# 2. ФІЗИЧНА МОДЕЛЬ (ФУНКЦІЇ)
# ==========================================
def specific_resistance(v_kmh):
    return 1.5 + 0.01 * v_kmh + 0.0002 * (v_kmh ** 2)

def specific_braking(v_kmh, theta):
    phi_k = 0.27 * (v_kmh + 100) / (5 * v_kmh + 100)
    return 1000 * theta * phi_k

def get_gradient(s, profile):
    current_s = 0
    for _, row in profile.iterrows():
        current_s += row['Довжина, м']
        if s <= current_s:
            return row['Ухил, ‰']
    return 0

def force_tick_formatter(x, pos):
    if x < 0: return f"+{abs(x):g}"
    elif x > 0: return f"-{x:g}"
    else: return "0"

# ==========================================
# 3. ІНТЕРФЕЙС - БІЧНА ПАНЕЛЬ (ВВІД ДАНИХ)
# ==========================================
st.sidebar.header("⚙️ Вхідні параметри")

loco_name = st.sidebar.selectbox("Оберіть локомотив:", list(LOCOMOTIVES.keys()))
loco_data = LOCOMOTIVES[loco_name]

m_loco = st.sidebar.number_input("Маса локомотива, т", value=loco_data["mass"], step=10.0)
m_train = st.sidebar.number_input("Маса состава, т", value=3000.0, step=100.0)
theta = st.sidebar.number_input("Гальмівний коефіцієнт (θ)", value=0.33, step=0.01)
l_p = st.sidebar.number_input("Розрахункова довжина поїзда, м", value=600.0, step=50.0)
v_max_section = st.sidebar.number_input("Обмеження по перегону, км/год", value=80.0, step=5.0)
v_p = st.sidebar.number_input("Розрахункова швидкість, км/год", value=25.0, step=5.0)

st.sidebar.subheader("Профіль колії")
default_profile = pd.DataFrame({
    "Довжина, м": pd.Series([], dtype=float),
    "Ухил, ‰": pd.Series([], dtype=float)
})
edited_profile = st.sidebar.data_editor(default_profile, num_rows="dynamic")

with st.sidebar.expander("⚡ Струмові характеристики (I від v)"):
    default_current = pd.DataFrame({
        "v, км/год": loco_data["v"],
        "I, А": [0.0] * len(loco_data["v"])
    })
    edited_current = st.data_editor(default_current, num_rows="dynamic")

calc_button = st.sidebar.button("🚀 Розрахувати", use_container_width=True)

if calc_button:
    st.session_state.calculated = True

# ==========================================
# 4. ГОЛОВНЕ ВІКНО - РЕЗУЛЬТАТИ ТА ГРАФІКИ
# ==========================================
st.title("🚂 Навчальний симулятор тягових розрахунків")

tab1, tab2 = st.tabs(["Крок 1: Діаграма питомих сил", "Крок 2: Криві швидкості та часу"])

m_total = m_loco + m_train
g, gamma = 9.81, 0.06

# Тягова характеристика
v_arr = np.array(loco_data["v"])
f_arr = np.array(loco_data["f"])
traction_force = interp1d(v_arr, f_arr, bounds_error=False, fill_value=0)

if st.session_state.calculated:
    
    edited_profile['Довжина, м'] = pd.to_numeric(edited_profile['Довжина, м'], errors='coerce')
    edited_profile['Ухил, ‰'] = pd.to_numeric(edited_profile['Ухил, ‰'], errors='coerce')
    edited_profile = edited_profile.dropna()

    edited_current["v, км/год"] = pd.to_numeric(edited_current["v, км/год"], errors='coerce')
    edited_current["I, А"] = pd.to_numeric(edited_current["I, А"], errors='coerce')
    edited_current = edited_current.dropna()
    current_interp = interp1d(edited_current["v, км/год"], edited_current["I, А"], bounds_error=False, fill_value=0)
    
    # ------------------------------------------
    # ДИНАМІЧНИЙ РОЗРАХУНОК МЕЖ ГРАФІКІВ
    # ------------------------------------------
    # Максимальна швидкість для осі Y (враховує і перегін, і можливості локомотива)
    v_plot_max = max(100, int(v_max_section + 10), int(np.max(v_arr)))
    v_plot_max = int(np.ceil(v_plot_max / 10) * 10)  # Округлення до десятків
    v_range = np.linspace(0, v_plot_max, int(v_plot_max) * 2)

    f_k_spec = (traction_force(v_range) * 1000) / (m_total * g) 
    w_o_spec = specific_resistance(v_range)
    b_t_spec = specific_braking(v_range, theta)

    f_p = f_k_spec - w_o_spec          
    w_x = -w_o_spec                    
    b_p = -b_t_spec - w_o_spec         

    # Динамічні межі для осі X (щоб влазили будь-які пікові значення сил)
    force_min_x = np.floor(np.min(-f_p) / 10) * 10 - 10
    if force_min_x > -20: force_min_x = -20
    
    force_max_x = np.ceil(np.max(np.abs(b_p)) / 10) * 10 + 10
    if force_max_x < 100: force_max_x = 100

    # ------------------------------------------
    # КРОК 1: ДІАГРАМА ПИТОМИХ СИЛ
    # ------------------------------------------
    with tab1:
        st.subheader(f"Діаграма прискорюючих і сповільнюючих сил ({loco_name})")
        
        # Перераховуємо фізичний розмір (6 мм = 1 Н/кН, 1 мм = 1 км/год)
        fig1_w = ((force_max_x - force_min_x) * 6) / 25.4 + 1.0  
        fig1_h = (v_plot_max * 1) / 25.4 + 1.0  

        fig1, ax_forces = plt.subplots(figsize=(fig1_w, fig1_h))
        ax_forces.plot(-f_p, v_range, 'g', lw=2, label=r'Тяга: $f_p$')
        ax_forces.plot(np.abs(w_x), v_range, 'b', lw=2, label=r'Вибіг: $\omega_{ox}$')
        ax_forces.plot(np.abs(b_p), v_range, 'r', lw=2, label=r'Гальмування: $b_t + \omega_{ox}$')
        
        ax_forces.axvline(0, color='black', lw=1.5)
        ax_forces.set_ylabel('Швидкість v, км/год', fontsize=12)
        ax_forces.set_xlabel('Питомі сили, Н/кН', fontsize=12)
        ax_forces.grid(True, linestyle='--')
        ax_forces.legend(fontsize=11)
        
        # Використовуємо динамічні межі
        ax_forces.set_xlim(force_min_x, force_max_x)
        ax_forces.set_ylim(0, v_plot_max)
        ax_forces.set_yticks(np.arange(0, v_plot_max + 1, 10))
        
        ax_forces.xaxis.set_major_locator(MultipleLocator(10 if (force_max_x - force_min_x) > 200 else 5))
        ax_forces.xaxis.set_major_formatter(FuncFormatter(force_tick_formatter))
        ax_forces.set_aspect(1/6)
        
        st.pyplot(fig1, use_container_width=False)
        
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png", dpi=300, bbox_inches='tight')
        st.download_button(
            label="💾 Завантажити діаграму у високій якості",
            data=buf1.getvalue(),
            file_name="diagrama_syl.png",
            mime="image/png"
        )

    # ------------------------------------------
    # КРОК 2: ІНТЕГРУВАННЯ ТА ГРАФІК РУХУ
    # ------------------------------------------
    with tab2:
        st.subheader("Побудова кривих швидкості, часу, струму та профілю колії")
        
        dt = 1.0 
        t_max = 7200 
        total_distance = edited_profile['Довжина, м'].sum()
        S_target_limit = total_distance - edited_profile.iloc[-1]['Довжина, м'] - (l_p / 2)
        v_target_limit = 50 / 3.6 

        time_log, distance_log, velocity_log, mode_log, current_log = [0], [0], [0], ['т'], [current_interp(0)]
        v_ms, s, t = 0.0, 0.0, 0.0
        is_braking_to_stop = False

        progress_bar = st.progress(0)

        max_iters = int(t_max / dt)
        for i in range(max_iters):
            if s >= total_distance: break
            
            v_kmh = v_ms * 3.6
            
            if not is_braking_to_stop:
                curr_b = specific_braking(v_kmh, theta)
                curr_w = specific_resistance(v_kmh)
                a_brake_est = (curr_b + curr_w) * g / (1000 * (1 + gamma))
                s_req_stop = (v_ms ** 2) / (2 * max(a_brake_est, 0.05))
                if total_distance - s <= s_req_stop:
                    is_braking_to_stop = True

            is_braking_for_limit = False
            if not is_braking_to_stop and s < S_target_limit and v_ms > v_target_limit:
                curr_b = specific_braking(v_kmh, theta)
                curr_w = specific_resistance(v_kmh)
                a_brake_est = (curr_b + curr_w) * g / (1000 * (1 + gamma))
                s_req_limit = (v_ms**2 - v_target_limit**2) / (2 * max(a_brake_est, 0.05))
                if S_target_limit - s <= s_req_limit:
                    is_braking_for_limit = True

            if is_braking_to_stop or is_braking_for_limit:
                mode = 'г'
                F_k = 0
                B_t = specific_braking(v_kmh, theta) * m_total * g
            else:
                if s >= S_target_limit and v_kmh > 50:
                    F_k = 0
                    if v_kmh > 52:
                        mode = 'г'
                        B_t = specific_braking(v_kmh, theta) * m_total * g
                    else:
                        mode = 'хх'
                        B_t = 0
                else:
                    if v_kmh >= v_max_section:
                        F_k = 0
                        if v_kmh > v_max_section + 2:
                            mode = 'г'
                            B_t = specific_braking(v_kmh, theta) * m_total * g
                        else:
                            mode = 'хх'
                            B_t = 0
                    else:
                        mode = 'т'
                        F_k = traction_force(v_kmh) * 1000
                        B_t = 0
            
            W_o = specific_resistance(v_kmh) * m_total * g
            W_i = m_total * g * get_gradient(s, edited_profile)
            
            m_eq = m_total * 1000 * (1 + gamma) 
            a = (F_k - W_o - W_i - B_t) / m_eq
            
            v_ms += a * dt
            if v_ms < 0: v_ms = 0
            s += v_ms * dt
            t += dt
            
            time_log.append(t / 60)
            distance_log.append(s / 1000)
            velocity_log.append(v_ms * 3.6)
            mode_log.append(mode)
            current_log.append(float(current_interp(v_ms * 3.6)) if mode == 'т' else 0.0)
            
            if i % 100 == 0: progress_bar.progress(min(s / total_distance, 1.0))
            if is_braking_to_stop and v_ms == 0: break

        progress_bar.progress(1.0)
        
        # --- ПОБУДОВА ГРАФІКА РУХУ ---
        scale_factor = 40 / 6 
        
        # Зміщуємо вісь відстані на позначку 5 (відповідає -5 Н/кН), як було спочатку
        start_x_offset = 5
        x_dist_mapped = start_x_offset + np.array(distance_log) * scale_factor

        max_x = max(force_max_x, x_dist_mapped[-1] + 5)
        
        fig_w = ((max_x - force_min_x) * 6) / 25.4 + 1.0
        fig_h_main = v_plot_max / 25.4
        fig_h_total = fig_h_main * 1.25 + 2.0  

        fig2, (ax_main, ax_prof) = plt.subplots(2, 1, figsize=(fig_w, fig_h_total), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)

        ax_main.plot(-f_p, v_range, 'g', lw=2, label=r'Тяга: $f_p$')
        ax_main.plot(np.abs(w_x), v_range, 'b', lw=2, label=r'Вибіг: $\omega_{ox}$')
        ax_main.plot(np.abs(b_p), v_range, 'r', lw=2, label=r'Гальмування: $b_t + \omega_{ox}$')
        ax_main.axvline(0, color='black', lw=1.5)

        ax_main.plot(x_dist_mapped, velocity_log, color='blue', lw=2, label='Швидкість v')
        
        limit_x_mapped = start_x_offset + (S_target_limit / 1000) * scale_factor
        ax_main.hlines(50, limit_x_mapped, x_dist_mapped[-1], colors='orange', linestyles='dashdot', lw=2, label='Обмеження ≤ 50 км/год')
        ax_main.vlines(limit_x_mapped, 0, 50, colors='orange', linestyles='dashdot', lw=2)
        ax_main.hlines(v_max_section, start_x_offset, x_dist_mapped[-1], colors='purple', linestyles='dashed', lw=1.5, label='Обмеження (перегін)')
        ax_main.hlines(v_p, start_x_offset, x_dist_mapped[-1], colors='green', linestyles='dotted', lw=1.5, label='Розрахункова швидкість')

        ax_main.text(x_dist_mapped[0], velocity_log[0] + 3, mode_log[0], fontsize=10, color='black', fontweight='bold')
        last_label_dist = distance_log[0]
        
        for i in range(1, len(mode_log)):
            if mode_log[i] != mode_log[i-1]:
                if distance_log[i] - last_label_dist > 0.2:
                    ax_main.text(x_dist_mapped[i], velocity_log[i] + 3, mode_log[i], fontsize=10, color='black', fontweight='bold')
                    last_label_dist = distance_log[i]

        ax_main.set_ylabel('Швидкість v, км/год', fontsize=12)
        ax_main.set_ylim(0, v_plot_max)
        ax_main.set_yticks(np.arange(0, v_plot_max + 1, 10))
        ax_main.set_aspect(1/6)
        ax_main.set_xlim(force_min_x, max_x)
        ax_main.xaxis.set_major_locator(MultipleLocator(10 if (max_x - force_min_x) > 200 else 5))
        ax_main.xaxis.set_major_formatter(FuncFormatter(force_tick_formatter))
        ax_main.set_xlabel('Питомі сили, Н/кН', fontsize=12)
        ax_main.tick_params(axis='x', labelbottom=True)

        x_time_plot, y_time_plot = [], []
        current_period = 0
        
        time_scale_factor = v_plot_max / 10.0

        for i in range(len(time_log)):
            t_curr, x = time_log[i], x_dist_mapped[i]
            if t_curr // 10 > current_period and i > 0:
                t_target = (current_period + 1) * 10
                t_prev, x_prev = time_log[i-1], x_dist_mapped[i-1]
                ratio = (t_target - t_prev) / (t_curr - t_prev) if (t_curr - t_prev) != 0 else 0
                x_target = x_prev + ratio * (x - x_prev)
                x_time_plot.extend([x_target, x_target])
                y_time_plot.extend([float(v_plot_max), 0.0])
                current_period += 1
            x_time_plot.append(x)
            y_time_plot.append((t_curr - current_period * 10) * time_scale_factor)

        ax_main.plot(x_time_plot, y_time_plot, color='red', linestyle='--', lw=2, label='Час t')
        secax_time = ax_main.secondary_yaxis('right', functions=(lambda y: y / time_scale_factor, lambda y: y * time_scale_factor))
        secax_time.set_ylabel('Час t, хв', color='red', fontsize=12)
        secax_time.set_yticks(np.arange(0, int(np.ceil(time_log[-1])) + 2, 1))

        # --- КРИВА СТРУМУ ---
        ax_curr = ax_main.twinx()
        ax_curr.spines['right'].set_position(('outward', 55)) # Зміщуємо вісь праворуч, щоб не накладалася на вісь часу
        ax_curr.plot(x_dist_mapped, current_log, color='magenta', lw=1.5, label='Струм I')
        ax_curr.set_ylabel('Струм I, А', color='magenta', fontsize=12)
        ax_curr.tick_params(axis='y', labelcolor='magenta')
        max_curr = max(current_log) if max(current_log) > 0 else 100
        ax_curr.set_ylim(0, max_curr * 1.2)

        # Об'єднана легенда
        lines_1, labels_1 = ax_main.get_legend_handles_labels()
        lines_2, labels_2 = ax_curr.get_legend_handles_labels()
        ax_main.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        ax_main.grid(True, linestyle='--')
        ax_main.set_title("Графік тягових розрахунків на ділянці", fontsize=14)

        # Профіль
        profile_x, profile_y = [0], [0]
        current_x, current_elev = 0, 0
        for _, row in edited_profile.iterrows():
            length = row['Довжина, м']
            grad = row['Ухил, ‰']
            current_x += length / 1000 
            current_elev += length * (grad / 1000)
            profile_x.append(current_x)
            profile_y.append(current_elev)

        profile_x_mapped = start_x_offset + np.array(profile_x) * scale_factor
        ax_prof.plot(profile_x_mapped, profile_y, color='black', lw=2)

        min_y = min(profile_y)
        ax_prof.set_ylim(min_y - 20, max(profile_y) + 10)
        ax_prof.fill_between(profile_x_mapped, profile_y, min_y - 5, color='gray', alpha=0.3)
        ax_prof.set_ylabel('Висота, м', fontsize=12)
        ax_prof.grid(True, linestyle='--')

        km_ticks = start_x_offset + np.arange(0, np.ceil(distance_log[-1]) + 1) * scale_factor
        ax_prof_km = ax_prof.twiny()
        ax_prof_km.set_xlim(ax_main.get_xlim())
        ax_prof_km.set_xticks(km_ticks)
        ax_prof_km.set_xticklabels([f"{km:g}" for km in np.arange(0, np.ceil(distance_log[-1]) + 1)])
        ax_prof_km.xaxis.set_ticks_position('bottom')
        ax_prof_km.xaxis.set_label_position('bottom')
        ax_prof_km.spines['bottom'].set_position(('outward', 0))
        ax_prof_km.set_xlabel(f'Відстань S, км (початок відліку: -{start_x_offset} Н/кН)', color='blue', fontsize=12)
        ax_prof.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        curr_x = 0
        for _, row in edited_profile.iterrows():
            length = row['Довжина, м']
            grad = row['Ухил, ‰']
            mid_km = curr_x + (length / 1000) / 2
            mid_mapped = start_x_offset + mid_km * scale_factor
            ax_prof.text(mid_mapped, min_y - 12, f'{grad} ‰ | {length} м', ha='center', va='center', fontsize=10, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
            curr_x += length / 1000

        fig2.tight_layout()
        
        st.pyplot(fig2, use_container_width=False)
        
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", dpi=300, bbox_inches='tight')
        st.download_button(
            label="💾 Завантажити графік руху у високій якості",
            data=buf2.getvalue(),
            file_name="grafik_ruhu.png",
            mime="image/png"
        )
        
        st.success(f"✅ Розрахунок завершено! Час ходу: {time_log[-1]:.2f} хв. Пройдено: {distance_log[-1]:.2f} км.")
