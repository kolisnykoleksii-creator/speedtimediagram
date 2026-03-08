import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Налаштування сторінки
st.set_page_config(page_title="Тягові розрахунки", layout="wide")

# Ініціалізація стану для збереження результатів
if 'calculated' not in st.session_state:
    st.session_state.calculated = False

# ==========================================
# 1. БАЗИ ДАНИХ ЛОКОМОТИВІВ
# ==========================================
LOCOMOTIVES = {
    "ТГ16 (За замовчуванням)": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "f": [400, 380, 300, 240, 190, 150, 120, 100, 80, 60, 40],
        "mass": 120.0
    },
    "ВЛ80с": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "f": [650, 650, 630, 580, 480, 380, 310, 260, 220, 190, 160],
        "mass": 192.0
    },
    "2ТЕ116": {
        "v": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "f": [500, 480, 420, 320, 240, 190, 150, 120, 100, 85, 70],
        "mass": 276.0
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
    "Довжина, м": [1500, 2000, 1000, 2500, 3000],
    "Ухил, ‰": [0, 8, 12, -5, 2]
})
edited_profile = st.sidebar.data_editor(default_profile, num_rows="dynamic")

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
    
    # ------------------------------------------
    # КРОК 1: ДІАГРАМА ПИТОМИХ СИЛ
    # ------------------------------------------
    with tab1:
        st.subheader(f"Діаграма прискорюючих і сповільнюючих сил ({loco_name})")
        
        v_range = np.linspace(0, 100, 100)
        f_k_spec = (traction_force(v_range) * 1000) / (m_total * g) 
        w_o_spec = specific_resistance(v_range)
        b_t_spec = specific_braking(v_range, theta)

        f_p = f_k_spec - w_o_spec          
        w_x = -w_o_spec                    
        b_p = -b_t_spec - w_o_spec         

        # Динамічний точний розрахунок розміру для уникнення дрібного тексту
        fig1_w = (120 * 6) / 25.4 + 1.0  # від -20 до 100 (120 од.) по 6 мм + відступи
        fig1_h = (100 * 1) / 25.4 + 1.0  # 100 од. по 1 мм + відступи

        fig1, ax_forces = plt.subplots(figsize=(fig1_w, fig1_h))
        ax_forces.plot(-f_p, v_range, 'g', lw=2, label=r'Тяга: $f_p$')
        ax_forces.plot(np.abs(w_x), v_range, 'b', lw=2, label=r'Вибіг: $\omega_{ox}$')
        ax_forces.plot(np.abs(b_p), v_range, 'r', lw=2, label=r'Гальмування: $b_t + \omega_{ox}$')
        
        ax_forces.axvline(0, color='black', lw=1.5)
        ax_forces.set_ylabel('Швидкість v, км/год', fontsize=12)
        ax_forces.set_xlabel('Питомі сили, Н/кН', fontsize=12)
        ax_forces.grid(True, linestyle='--')
        ax_forces.legend(fontsize=11)
        ax_forces.set_xlim(-20, 100)
        ax_forces.set_ylim(0, 100)
        ax_forces.set_yticks(np.arange(0, 101, 10))
        ax_forces.xaxis.set_major_locator(MultipleLocator(5))
        ax_forces.xaxis.set_major_formatter(FuncFormatter(force_tick_formatter))
        ax_forces.set_aspect(1/6)
        
        # ВИВОДИМО БЕЗ СТИСНЕННЯ (з'явиться горизонтальний скрол)
        st.pyplot(fig1, use_container_width=False)

    # ------------------------------------------
    # КРОК 2: ІНТЕГРУВАННЯ ТА ГРАФІК РУХУ
    # ------------------------------------------
    with tab2:
        st.subheader("Побудова кривих швидкості, часу та профілю колії")
        
        # ДОДАНО: Інтерактивні панелі для наближення графіка
        col1, col2 = st.columns(2)
        with col1:
            stretch_x = st.slider("↔️ Розтягнути дистанцію (щоб розгледіти літери):", min_value=1.0, max_value=5.0, value=1.0, step=0.5)
        with col2:
            zoom_all = st.slider("🔍 Збільшити весь графік (Зум):", min_value=1.0, max_value=3.0, value=1.0, step=0.5)
        
        dt = 1.0 
        t_max = 7200 
        total_distance = edited_profile['Довжина, м'].sum()
        S_target_limit = total_distance - edited_profile.iloc[-1]['Довжина, м'] - (l_p / 2)
        v_target_limit = 50 / 3.6 

        time_log, distance_log, velocity_log, mode_log = [0], [0], [0], ['т']
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
            
            if i % 100 == 0: progress_bar.progress(min(s / total_distance, 1.0))
            if is_braking_to_stop and v_ms == 0: break

        progress_bar.progress(1.0)
        
        # --- ПОБУДОВА ГРАФІКА РУХУ ---
        # Враховуємо коефіцієнт розтягування дистанції
        scale_factor = (40 * stretch_x) / 6 
        start_x_offset = 5
        x_dist_mapped = start_x_offset + np.array(distance_log) * scale_factor

        max_x = max(100, x_dist_mapped[-1] + 5)
        
        # Точний розрахунок розмірів полотна для Кроку 2
        fig_w = ((max_x + 20) * 6 * zoom_all) / 25.4 + 1.0
        fig_h_main = (100 * zoom_all) / 25.4
        fig_h_total = fig_h_main * 1.25 + 2.0  # Висота головного + профілю (1/4) + відступи

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

        # Виводимо літери кожного разу, коли режим змінився
        for i in range(1, len(mode_log)):
            if mode_log[i] != mode_log[i-1]:
                ax_main.text(x_dist_mapped[i], velocity_log[i] + 3, mode_log[i], fontsize=10*zoom_all, color='black', fontweight='bold')

        ax_main.set_ylabel('Швидкість v, км/год', fontsize=12)
        ax_main.set_ylim(0, 100)
        ax_main.set_yticks(np.arange(0, 101, 10))
        ax_main.set_aspect(1/6)
        ax_main.set_xlim(-20, max_x)
        ax_main.xaxis.set_major_locator(MultipleLocator(5))
        ax_main.xaxis.set_major_formatter(FuncFormatter(force_tick_formatter))
        ax_main.set_xlabel('Питомі сили, Н/кН', fontsize=12)

        x_time_plot, y_time_plot = [], []
        current_period = 0
        for i in range(len(time_log)):
            t_curr, x = time_log[i], x_dist_mapped[i]
            if t_curr // 10 > current_period and i > 0:
                t_target = (current_period + 1) * 10
                t_prev, x_prev = time_log[i-1], x_dist_mapped[i-1]
                ratio = (t_target - t_prev) / (t_curr - t_prev) if (t_curr - t_prev) != 0 else 0
                x_target = x_prev + ratio * (x - x_prev)
                x_time_plot.extend([x_target, x_target])
                y_time_plot.extend([100.0, 0.0])
                current_period += 1
            x_time_plot.append(x)
            y_time_plot.append((t_curr - current_period * 10) * 10)

        ax_main.plot(x_time_plot, y_time_plot, color='red', linestyle='--', lw=2, label='Час t')
        secax_time = ax_main.secondary_yaxis('right', functions=(lambda y: y / 10, lambda y: y * 10))
        secax_time.set_ylabel('Час t, хв', color='red', fontsize=12)
        secax_time.set_yticks(np.arange(0, 11, 1))

        ax_main.legend(loc='upper right')
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
            ax_prof.text(mid_mapped, min_y - 12, f'{grad} ‰ | {length} м', ha='center', va='center', fontsize=10*zoom_all, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
            curr_x += length / 1000

        fig2.tight_layout()
        
        st.pyplot(fig2, use_container_width=False)
        
        st.success(f"✅ Розрахунок завершено! Час ходу: {time_log[-1]:.2f} хв. Пройдено: {distance_log[-1]:.2f} км.")
