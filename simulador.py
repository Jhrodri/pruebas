import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import base64

st.set_page_config(page_title="HORTITRANS Invernadero", layout="wide")

# --- Constantes y Coeficientes del Modelo ---
LAMBDA_J_KG = 2.45e6
GAMMA = 66
RHO_AIR = 1.2
CPA = 1012
C1, C2, C3, C4, C5, C6 = 0.154, 1.10, 1.13, 1.65, 0.56, 13.0
SIGMA = 5.67e-8          # W·m⁻²·K⁻⁴  (Stefan-Boltzmann)

# --- Funciones de Cálculo del Modelo ---
def calculate_saturation_vapor_pressure(T_celsius):
    return 611.2 * np.exp((17.67 * T_celsius) / (T_celsius + 243.5))

def hortitrans_model(params):
    A_suelo, A_vent_total, tipo_cubierta, estanqueidad, _ = params['gh']
    LAI = params['crop']['LAI']
    T_o, RH_o, R_sol_ext, w = params['weather']
    T_i, angulo_vent, E_ad_kgs = params['control']

    A_cubierta = A_suelo * 1.5
    e_si = calculate_saturation_vapor_pressure(T_i)
    e_o = calculate_saturation_vapor_pressure(T_o) * (RH_o / 100.0)
    _tau = 0.5 if (params.get('lw', {}).get('cielo') == 'Cielo nublado' and R_sol_ext > 0) else 0.7
    R_i = R_sol_ext * _tau

    _f_luz = 1.0 if R_i >= 10.0 else 0.0
    h_t = C4 * LAI * (1 - C5 * np.exp(-R_i / C6)) * _f_luz
    f_w = 0.22
    f_T = 0.16

    if estanqueidad == 'Totalmente Estanco':
        A_infiltracion = 0.0
        A_ventilacion_efectiva = 0.0
    else:
        f_c = {'Muy Estanco': 2.5e-4, 'Normal': 10e-4, 'Con Fugas': 20e-4}[estanqueidad]
        A_infiltracion = A_cubierta * f_c
        Cd = 0.65
        A_ventilacion_efectiva = Cd * A_vent_total * (angulo_vent / 100.0)

    A_total_flujo = A_infiltracion + A_ventilacion_efectiva
    q_natural = A_total_flujo * np.sqrt(f_w**2 * w**2 + f_T**2 * abs(T_i - T_o))
    q_min     = A_ventilacion_efectiva * 0.05   # mínimo difusión turbulenta con ventanas abiertas (m³/s)
    q         = max(q_natural, q_min)
    h_v = (RHO_AIR * CPA * q) / A_suelo if A_suelo > 0 else 0
    hc = 5.25
    U_value = 6.0
    delta_slope = (calculate_saturation_vapor_pressure(T_i + 0.5) - calculate_saturation_vapor_pressure(T_i - 0.5))
    den_cci = (U_value + hc * delta_slope / GAMMA)
    cci = (hc/GAMMA) / den_cci if den_cci > 0 else 0
    hci = hc * cci * delta_slope
    T_x_celsius = T_i - (T_i - T_o) * (U_value / (hc + U_value)) if (hc + U_value) > 0 else T_i
    esx = calculate_saturation_vapor_pressure(T_x_celsius)
    hd = 0

    a_transp = C1 * np.log(1 + C2 * LAI**C3) if LAI > 0 else 0
    E_ad_W_m2 = E_ad_kgs * LAMBDA_J_KG

    numerador_eq15 = (E_ad_W_m2 * GAMMA) + (GAMMA * a_transp * R_i) + ((h_t + hci) * e_si) + (hc * esx) + (h_v * e_o)
    denominador_eq15 = h_t + hci + hc + h_v + hd
    if denominador_eq15 <= 1e-9: return {}
    e_i = numerador_eq15 / denominador_eq15

    RH_i = (e_i / e_si) * 100
    VPD_i_kPa = (e_si - e_i) / 1000
    RH_i = min(100, max(0, RH_i))
    VPD_i_kPa = max(0, VPD_i_kPa)

    E_t_kgs = max(0, (a_transp * R_i * GAMMA + h_t * (e_si - e_i))) / (LAMBDA_J_KG * GAMMA)
    E_c_kgs = max(0, (hc * (e_i - esx) - hci * (e_si - e_i))) / (LAMBDA_J_KG * GAMMA)
    E_v_kgs = max(0, h_v * (e_i - e_o)) / (LAMBDA_J_KG * GAMMA)

    H_INV   = 4.0                                          # altura media invernadero (m)
    V_inv   = A_suelo * H_INV                              # volumen interior (m³)
    ACH     = q * 3600.0 / V_inv if V_inv > 0 else 0.0    # renovaciones·h⁻¹

    return {
        'RH_i': RH_i, 'VPD_i_kPa': VPD_i_kPa,
        'E_t_kgs': E_t_kgs, 'E_c_kgs': E_c_kgs, 'E_v_kgs': E_v_kgs,
        'q_m3s': q, 'ACH': ACH,
    }

def solve_equilibrium_temperature(params, max_iter=60, tol=0.05, alpha=0.4):
    """
    Calcula iterativamente (punto fijo amortiguado) la temperatura interior
    de equilibrio libre (sin calefacción/refrigeración activa) y la carga
    de climatización necesaria para mantener la temperatura de consigna.

    Balance energético (radiación OL linealizada como conductancia adicional de cubierta):
      Q_solar + Q_hvac = ρ·Cp·q·(T_i−T_o) + U_r·A_cub·(T_i−T_o) + Q_off + E_t·λ·A_suelo
    donde:
      h_rad = 4·ε·σ·T_mean³             (conductancia radiativa, ε=0.90 PE térmico)
      U_r   = hc_i·(h_ext+h_rad)/(hc_i+h_ext+h_rad)   (≈ 3.5 W·m⁻²·K⁻¹)
      Q_off = hc_i·h_rad/(hc_i+h_ext+h_rad)·A_cub·ΔT_sky  (offset por cielo frío)
    La radiación y la conducción se suman EN la cubierta (no son independientes en el
    balance interior), por lo que Q_LW NO se resta por separado sobre U_eff original.
    """
    T_o          = params['weather'][0]
    R_sol_ext    = params['weather'][2]
    w            = params['weather'][3]
    A_suelo      = params['gh'][0]
    A_vent_tot   = params['gh'][1]
    estanqueidad = params['gh'][3]
    angulo_vent  = params['control'][1]
    E_ad_kgs     = params['control'][2]
    T_consigna   = params['control'][0]
    delta_T_sky  = params.get('lw', {}).get('delta_T_sky', 10)   # °C
    Q_suelo      = 35.0 * A_suelo                                  # W (25 W/m² fijo)

    EPS_CUB = 0.90   # emisividad PE térmico
    H_EXT   = 6.0    # conductancia exterior cubierta (= U_value en hortitrans)
    HC_INT  = 5.25   # convección interior

    A_cubierta = A_suelo * 1.5
    _tau       = 0.5 if (delta_T_sky == 10 and R_sol_ext > 0) else 0.7
    R_i        = R_sol_ext * _tau
    Q_solar    = R_i * A_suelo                    # W

    def _q_flow(T_i):
        if estanqueidad == 'Totalmente Estanco':
            return 0.0
        f_c  = {'Muy Estanco': 2.5e-4, 'Normal': 10e-4, 'Con Fugas': 20e-4}[estanqueidad]
        A_if = A_cubierta * f_c
        A_vf = 0.65 * A_vent_tot * (angulo_vent / 100.0)
        q_nat = (A_if + A_vf) * np.sqrt(0.22**2 * w**2 + 0.16**2 * abs(T_i - T_o))
        q_min = A_vf * 0.05    # mínimo difusión turbulenta con ventanas abiertas
        return max(q_nat, q_min)

    def _cov_terms(T_i):
        """U_eff con radiación OL linealizada y offset por depresión del cielo (W)."""
        T_cub  = T_i - (T_i - T_o) * H_EXT / (HC_INT + H_EXT)  # T cubierta aprox.
        T_sky  = T_o - delta_T_sky
        T_m_K  = (T_cub + T_sky) / 2.0 + 273.15                 # temperatura media K
        h_rad  = 4.0 * EPS_CUB * SIGMA * T_m_K**3               # W·m⁻²·K⁻¹  (~4–5)
        denom  = HC_INT + H_EXT + h_rad
        U_r    = HC_INT * (H_EXT + h_rad) / denom                # ≈ 3.5 W·m⁻²·K⁻¹
        Q_off  = HC_INT * h_rad / denom * A_cubierta * delta_T_sky  # W  (>0 = pérdida)
        return U_r, Q_off

    T_i = float(T_consigna)
    converged = False
    r_eq = {}

    for n in range(max_iter):
        p_it = {**params, 'control': (T_i, angulo_vent, E_ad_kgs)}
        r_eq = hortitrans_model(p_it)
        if not r_eq:
            break
        Q_lat      = r_eq['E_t_kgs'] * LAMBDA_J_KG * A_suelo
        U_r, Q_off = _cov_terms(T_i)
        K_loss     = RHO_AIR * CPA * _q_flow(T_i) + U_r * A_cubierta
        T_new      = float(np.clip(
            T_o + (Q_solar + Q_suelo - Q_lat - Q_off) / K_loss if K_loss > 1e-3 else T_o + 40.0,
            -15.0, 80.0))
        T_next = alpha * T_new + (1 - alpha) * T_i
        if abs(T_next - T_i) < tol:
            T_i = T_next
            converged = True
            break
        T_i = T_next

    # Carga HVAC a T_consigna
    r_c            = hortitrans_model({**params, 'control': (T_consigna, angulo_vent, E_ad_kgs)})
    U_r_c, Q_off_c = _cov_terms(T_consigna)
    K_c            = RHO_AIR * CPA * _q_flow(T_consigna) + U_r_c * A_cubierta
    Q_lat_c        = r_c.get('E_t_kgs', 0) * LAMBDA_J_KG * A_suelo if r_c else 0.0
    Q_hvac         = K_c * (T_consigna - T_o) + Q_lat_c + Q_off_c - Q_solar  # W; >0 calefacción

    return {
        'T_eq':        round(T_i, 1),
        'T_consigna':  T_consigna,
        'Q_hvac_W':    Q_hvac,
        'Q_hvac_kW':   Q_hvac / 1000,
        'Q_hvac_Wm2':  Q_hvac / A_suelo,
        'iters':       n + 1,
        'converged':   converged,
        'results_eq':  r_eq,
    }

# --- Interfaz de Usuario ---
st.title("💧 Simulador de humedad y transpiración en Invernaderos. VERSIÓN DE PRUEBAS EN DESARROLLO")
st.info("ESTA ES UNA PÁGINA DE PRUEBAS, LOS RESULTADOS PUEDEN SER ERRÓNEOS**.")

with st.expander("ℹ️ ¿Qué se va calcular?"):
    st.markdown("""
    - Este simulador está adaptado del Modelo HORTITRANS de O. Jolliet
    - El simulador calcula la transpiración del cultivo y modifica la higrometría del invernadero.
    - Además de los valores higrométricos realiza recomendaciones de control.
    - Como consigna se ha establecido el rango 0,4-1,6 kPa.
    - Cuando cambies un valor procura que el resto de valores sea coherente.
    """)

# --- Session State: valores por defecto ---
_defaults = {
    'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Normal',
    'ss_LAI': 3.0, 'ss_T_o': 25.0, 'ss_RH_o': 80, 'ss_R_sol': 400, 'ss_viento': 2.0,
    'ss_T_i': 30.0, 'ss_apertura': 50, 'ss_humidif': 0.0,
    'ss_cielo': 'Cielo nublado',
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Aplica escenario pendiente ANTES de instanciar los widgets
if '_pending_scenario' in st.session_state:
    for k, v in st.session_state['_pending_scenario'].items():
        st.session_state[k] = v
    del st.session_state['_pending_scenario']

def load_scenario(params: dict, scenario_id: str = ''):
    st.session_state['_pending_scenario'] = params
    st.session_state['_last_loaded'] = scenario_id
    st.rerun()

def show_scenario_info(scenario_ids):
    """Muestra las condiciones del escenario activo si coincide con el expander actual."""
    if st.session_state.get('_last_loaded') not in (
        [scenario_ids] if isinstance(scenario_ids, str) else scenario_ids
    ):
        return
    ss = st.session_state
    st.info(
        f"**📋 Escenario cargado** — "
        f"T ext: **{ss['ss_T_o']} °C** · "
        f"HR ext: **{ss['ss_RH_o']} %** · "
        f"Rad: **{ss['ss_R_sol']} W/m²** · "
        f"Viento: **{ss['ss_viento']} m/s** · "
        f"LAI: **{ss['ss_LAI']}** · "
        f"T int: **{ss['ss_T_i']} °C** · "
        f"Apertura: **{ss['ss_apertura']} %** · "
        f"Estanqueidad: **{ss['ss_estanqueidad']}** · "
        f"Humidif: **{ss['ss_humidif']} L·m⁻²·h⁻¹**"
    )

# --- PANEL DE ENTRADAS EN LA BARRA LATERAL ---
_logo_b64 = base64.b64encode((Path(__file__).parent / "logos.png").read_bytes()).decode()
st.sidebar.markdown(
    f'<img src="data:image/png;base64,{_logo_b64}" style="width:100%; display:block;">',
    unsafe_allow_html=True,
)

_ESTANQUEIDAD_OPTS = ('Normal', 'Muy Estanco', 'Con Fugas', 'Totalmente Estanco')

with st.sidebar:
    st.header("Parámetros de Entrada")

    with st.expander("1. Parámetros del Invernadero", expanded=True):
        p_A_suelo = st.number_input(
            "Área del suelo (m²)", 10, 10000, step=10, key="ss_area_suelo"
        )
        p_A_vent_total = st.number_input(
            "Área total de ventanas (m²)", 0, 2000, step=5, key="ss_area_ventanas",
            help="Valor independiente del área de suelo."
        )
        p_estanqueidad = st.selectbox(
            "Estanqueidad de la cubierta",
            _ESTANQUEIDAD_OPTS,
            index=_ESTANQUEIDAD_OPTS.index(st.session_state['ss_estanqueidad']),
            key="ss_estanqueidad",
            help="'Totalmente Estanco' anula toda la ventilación."
        )

    with st.expander("2. Parámetros del Cultivo", expanded=True):
        p_LAI = st.slider("Índice de Área Foliar (LAI)", 0.0, 6.0, step=0.1, key="ss_LAI")

    with st.expander("3. Condiciones Climáticas Externas", expanded=True):
        p_T_o = st.slider("Temperatura exterior (°C)", -10.0, 45.0, step=0.5, key="ss_T_o")
        p_RH_o = st.slider("Humedad relativa exterior (%)", 10, 100, step=1, key="ss_RH_o")
        p_R_sol_ext = st.slider("Radiación solar exterior (W/m²)", 0, 1000, step=10, key="ss_R_sol")
        p_w = st.slider("Velocidad del viento (m/s)", 0.0, 15.0, step=0.5, key="ss_viento")
        _CIELO_OPTS = ['Cielo nublado', 'Cielo despejado']
        p_cielo = st.selectbox(
            "Estado del cielo", _CIELO_OPTS,
            index=_CIELO_OPTS.index(st.session_state['ss_cielo']),
            key="ss_cielo",
            help="Cielo nublado: ΔT_cielo = 10 °C · Cielo despejado: ΔT_cielo = 20 °C"
        )
        p_delta_T_sky = 10 if p_cielo == 'Cielo nublado' else 20

    with st.expander("4. Consignas de Control Interno", expanded=True):
        p_T_i = st.slider("Temperatura interior (°C)", 0.0, 45.0, step=0.5, key="ss_T_i")
        p_angulo_vent = st.slider("Apertura de ventanas (%)", 0, 100, step=1, key="ss_apertura")
        p_L_m2_h = st.number_input(
            "Humid./Deshumid.(Lm⁻²h⁻¹)",
            min_value=-1.0, max_value=1.0, step=0.05, key="ss_humidif",
            help="Introduce un valor en Litros por metro cuadrado a la hora."
        )

# --- CONVERSIÓN DE UNIDADES Y EJECUCIÓN DEL MODELO ---
p_tipo_cubierta = 'Simple'
p_E_ad_kgs = p_L_m2_h / 3600.0
params_dict = {
    'gh': (p_A_suelo, p_A_vent_total, p_tipo_cubierta, p_estanqueidad, 'Enrollable (lineal)'),
    'crop': {'LAI': p_LAI},
    'weather': (p_T_o, p_RH_o, p_R_sol_ext, p_w),
    'control': (p_T_i, p_angulo_vent, p_E_ad_kgs),
    'lw': {'delta_T_sky': p_delta_T_sky, 'cielo': p_cielo},
}
results = hortitrans_model(params_dict)
if results:
    st.session_state['_sim_results'] = results

# --- TABS ---
tab_sim, tab_act = st.tabs(["📊 Simulación", "📚 Actividades Guiadas"])

# ===== TAB 1: SIMULACIÓN =====
with tab_sim:
    if not results:
        st.error("Error en los cálculos. Revisa los parámetros de entrada.")
    else:
        st.subheader("Condiciones Interiores y Flujos de Agua")

        E_t_gmh = results['E_t_kgs'] * 1000 * 3600
        E_c_gmh = results['E_c_kgs'] * 1000 * 3600
        E_v_gmh = results['E_v_kgs'] * 1000 * 3600

        col1, col2 = st.columns(2)
        with col1:
            fig_rh = go.Figure(go.Indicator(
                mode="gauge+number",
                value=results['RH_i'],
                title={'text': "Humedad Relativa (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0,  70], 'color': "lightgreen"},
                        {'range': [70, 85], 'color': "yellow"},
                        {'range': [85,100], 'color': "lightcoral"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
                }
            ))
            fig_rh.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_rh, use_container_width=True, key="chart_rh")

        with col2:
            vpd_val = results['VPD_i_kPa']
            if vpd_val < 0.4:
                bar_color = "#d62728"
            elif vpd_val > 1.6:
                bar_color = "#ff7f0e"
            else:
                bar_color = "#2ca02c"
            fig_vpd = go.Figure(go.Indicator(
                mode="gauge+number",
                value=vpd_val,
                number={'suffix': " kPa", 'valueformat': ".2f"},
                title={'text': "Déficit de Presión de Vapor — DPV (kPa)"},
                gauge={
                    'axis': {'range': [0, 3.0], 'tickformat': ".1f"},
                    'bar': {'color': bar_color},
                    'steps': [
                        {'range': [0.0, 0.4], 'color': "#ffd6d6"},
                        {'range': [0.4, 1.6], 'color': "#d4edda"},
                        {'range': [1.6, 3.0], 'color': "#ffecd2"},
                    ],
                    'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': 1.6}
                }
            ))
            fig_vpd.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_vpd, use_container_width=True, key="chart_vpd")

        st.metric(label="Transpiración Cultivo (g m⁻² h⁻¹)", value=f"{E_t_gmh:.2f}")
        st.markdown("---")

        col_flujo1, col_flujo2, col_flujo3 = st.columns(3)
        with col_flujo1:
            st.metric(label="Ventilación (g m⁻² h⁻¹)", value=f"{E_v_gmh:.2f}", help="Vapor de agua que sale por la ventilación.")
        with col_flujo2:
            st.metric(label="Condensación (g m⁻² h⁻¹)", value=f"{E_c_gmh:.2f}", help="Vapor de agua que se condensa en la cubierta.")
        with col_flujo3:
            st.metric(label="Renovaciones (h⁻¹)", value=f"{results.get('ACH', 0):.1f}",
                      help="Tasa de renovación del aire interior. Calculada para una altura media de invernadero de 4 m.")

        st.markdown("---")
        st.subheader("Recomendaciones de Control")

        if E_c_gmh > 1.6:
            st.error("⚠️ **Hay condensación significativa en la cubierta.** Riesgo de goteo y enfermedades fúngicas.")

        vpd_actual = results['VPD_i_kPa']
        if p_LAI == 0:
            st.info("ℹ️ LAI = 0: no hay cultivo activo, las recomendaciones de DPV no son aplicables.")
        elif vpd_actual < 0.4:
            st.warning("🚨 **ALERTA — DPV MUY BAJO (< 0,4 kPa):** Humedad excesiva. Considera aumentar la ventilación, deshumidificar o incrementar la temperatura interior del invernadero.")
        elif vpd_actual > 1.6:
            st.warning("🌵 **ALERTA — DPV MUY ALTO (> 1,6 kPa):** Condiciones muy estresantes para el cultivo. Considera humidificar o bajar la temperatura del invernadero.")
        else:
            st.success(f"✅ DPV dentro del rango óptimo (0,4 – 1,6 kPa).")

        # --- Temperatura de equilibrio ---
        st.markdown("---")
        eq   = solve_equilibrium_temperature(params_dict)
        st.session_state['_eq_results'] = eq
        dT   = eq['T_eq'] - eq['T_consigna']

        with st.expander("🌡️ Condiciones de equilibrio pasivo", expanded=False):
            st.markdown(
                "Condiciones interiores si el invernadero operara **sin calefacción ni refrigeración activa**, "
                "únicamente con la ventilación y la radiación solar configuradas."
            )
            if eq['T_eq'] - p_T_o > 20:
                st.warning("Comprueba los datos introducidos o el grado de apertura de las ventanas. Los valores no son coherentes para los invernaderos mediterráneos y/o provocarían la muerte del cultivo.")
            else:
                r_eq = eq.get('results_eq', {})
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.metric("T equilibrio pasivo",
                              f"{eq['T_eq']:.1f} °C",
                              delta=f"{dT:+.1f} °C vs consigna",
                              delta_color="off")
                with c2:
                    st.metric("HR", f"{r_eq.get('RH_i', 0):.1f} %")
                with c3:
                    st.metric("DPV", f"{r_eq.get('VPD_i_kPa', 0):.2f} kPa")
                with c4:
                    st.metric("Transpiración",
                              f"{r_eq.get('E_t_kgs', 0)*1000*3600:.2f} g·m⁻²·h⁻¹")
                with c5:
                    st.metric("Renovaciones",
                              f"{r_eq.get('ACH', 0):.1f} h⁻¹",
                              help="Tasa de renovación del aire a T de equilibrio (altura 4 m).")

                if r_eq.get('E_c_kgs', 0) > 0:
                    st.warning("⚠️ Riesgo de condensación en cubierta en condiciones de equilibrio pasivo.")

                conv_txt = (f"✓ {eq['iters']} iteraciones"
                            if eq['converged']
                            else f"⚠️ No convergió ({eq['iters']} iteraciones)")
                st.caption(conv_txt)

# --- Generador de informe HTML ---
def _df_to_html(df):
    if df is None:
        return "<p><em>Sin datos registrados.</em></p>"
    return df.to_html(index=False, border=0, classes="data-table",
                      na_rep="—", float_format=lambda x: f"{x:.2f}")

def generate_html_report():
    from datetime import date as _date
    ss = st.session_state
    r  = ss.get('_sim_results', {})

    E_t = r.get('E_t_kgs', 0) * 1000 * 3600
    E_c = r.get('E_c_kgs', 0) * 1000 * 3600
    E_v = r.get('E_v_kgs', 0) * 1000 * 3600

    eq     = ss.get('_eq_results', {})
    r_eq_h = eq.get('results_eq', {})
    if eq:
        _T_eq = f"{eq['T_eq']:.1f} °C"
        _T_c  = f"{eq['T_consigna']:.1f} °C"
    else:
        _T_eq = _T_c = "—"
    if r_eq_h:
        _RH_eq  = f"{r_eq_h['RH_i']:.1f} %"
        _DPV_eq = f"{r_eq_h['VPD_i_kPa']:.2f} kPa"
        _Et_eq  = f"{r_eq_h['E_t_kgs']*1000*3600:.2f} g·m⁻²·h⁻¹"
    else:
        _RH_eq = _DPV_eq = _Et_eq = "—"

    tables_html = ""
    _table_defs = [
        ("A1 — Efecto de la ventilación",             "tbl_a1_data"),
        ("A2 — Viento (Prueba 1)",                    "tbl_a2_v_data"),
        ("A2 — Efecto chimenea (Prueba 2)",            "tbl_a2_t_data"),
        ("A3 — Grado de crecimiento del cultivo",     "tbl_a3_data"),
        ("A4 — Ciclo día-noche",                      "tbl_a4_data"),
        ("B1 — Gestión del riesgo de Botrytis",       "tbl_b1_data"),
        ("B2 — Estrés hídrico en verano",             "tbl_b2_data"),
        ("B3 — Comparativa de estanqueidades",        "tbl_b3_data"),
        ("C1 — Dimensionado del sistema de humidificación", "tbl_c1_data"),
        ("C2 — Protocolo de manejo estacional",       "tbl_c2_data"),
        ("C3 — Optimización del ratio de ventilación","tbl_c3_data"),
    ]
    for title, key in _table_defs:
        df = ss.get(key)
        tables_html += f"<h3>{title}</h3>{_df_to_html(df)}"

    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8">
<title>Informe HORTITRANS</title>
<style>
  body{{font-family:Arial,sans-serif;margin:35px;color:#222;}}
  h1{{color:#1a5276;border-bottom:3px solid #1a5276;padding-bottom:6px;}}
  h2{{color:#2874a6;margin-top:28px;border-left:4px solid #2874a6;padding-left:8px;}}
  h3{{color:#555;margin-top:18px;}}
  .params{{background:#eaf4fb;padding:12px 16px;border-radius:6px;}}
  .results{{background:#eafaf1;padding:12px 16px;border-radius:6px;}}
  .data-table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:0.9em;}}
  .data-table th{{background:#2874a6;color:#fff;padding:7px 10px;text-align:left;}}
  .data-table td{{border:1px solid #ccc;padding:6px 10px;}}
  .data-table tr:nth-child(even){{background:#f4f9fd;}}
  @media print{{body{{margin:15px;}}}}
</style></head><body>
<h1>📋 Informe de sesión — Simulador HORTITRANS</h1>
<p><strong>Fecha:</strong> {_date.today().strftime("%d/%m/%Y")}</p>

<h2>Parámetros del escenario activo</h2>
<div class="params">
<table class="data-table">
<tr><th>Parámetro</th><th>Valor</th></tr>
<tr><td>Temperatura exterior</td><td>{ss['ss_T_o']} °C</td></tr>
<tr><td>Humedad relativa exterior</td><td>{ss['ss_RH_o']} %</td></tr>
<tr><td>Radiación solar</td><td>{ss['ss_R_sol']} W/m²</td></tr>
<tr><td>Velocidad del viento</td><td>{ss['ss_viento']} m/s</td></tr>
<tr><td>LAI (Índice de Área Foliar)</td><td>{ss['ss_LAI']}</td></tr>
<tr><td>Temperatura interior</td><td>{ss['ss_T_i']} °C</td></tr>
<tr><td>Apertura de ventanas</td><td>{ss['ss_apertura']} %</td></tr>
<tr><td>Estanqueidad</td><td>{ss['ss_estanqueidad']}</td></tr>
<tr><td>Humidificación/Deshumidificación</td><td>{ss['ss_humidif']} L·m⁻²·h⁻¹</td></tr>
</table></div>

<h2>Resultados de la simulación</h2>
<div class="results">
<table class="data-table">
<tr><th>Variable</th><th>Valor</th></tr>
<tr><td>Humedad relativa interior (HR)</td><td>{r.get('RH_i', '—'):.1f} %</td></tr>
<tr><td>Déficit de presión de vapor (DPV)</td><td>{r.get('VPD_i_kPa', '—'):.2f} kPa</td></tr>
<tr><td>Transpiración del cultivo</td><td>{E_t:.2f} g·m⁻²·h⁻¹</td></tr>
<tr><td>Ventilación</td><td>{E_v:.2f} g·m⁻²·h⁻¹</td></tr>
<tr><td>Condensación en cubierta</td><td>{E_c:.2f} g·m⁻²·h⁻¹</td></tr>
</table></div>

<h2>Temperatura de equilibrio y carga de climatización</h2>
<div class="results">
<table class="data-table">
<tr><th>Variable</th><th>Valor</th></tr>
<tr><td>T equilibrio pasivo</td><td>{_T_eq}</td></tr>
<tr><td>T consigna</td><td>{_T_c}</td></tr>
<tr><td>HR en equilibrio</td><td>{_RH_eq}</td></tr>
<tr><td>DPV en equilibrio</td><td>{_DPV_eq}</td></tr>
<tr><td>Transpiración en equilibrio</td><td>{_Et_eq}</td></tr>
</table></div>

<h2>Tablas de registro de actividades</h2>
{tables_html}
</body></html>"""

# --- Helper para botones Load + Reset ---
def _nan(n): return [float('nan')] * n
_col_num = st.column_config.NumberColumn
_col_txt = st.column_config.TextColumn

def activity_buttons(label, key_load, key_reset, params, scenario_id):
    col_l, col_r = st.columns([3, 2])
    with col_l:
        if st.button(label, key=key_load):
            load_scenario(params, scenario_id)
    with col_r:
        if st.button("🔄 Restablecer", key=key_reset,
                     help="Recarga los parámetros iniciales del escenario"):
            load_scenario(params, scenario_id)
    show_scenario_info(scenario_id)

# ===== TAB 2: ACTIVIDADES =====
with tab_act:
    st.header("Actividades Guiadas — Nivel 1: Exploración")
    st.markdown(
        "Pulsa **▶ Cargar escenario** en cada actividad para precargar los parámetros en la barra lateral. "
        "Luego modifica los valores indicados en la tarea y anota los resultados en la pestaña **Simulación**."
    )
    st.download_button(
        label="📥 Descargar ficha de sesión (HTML)",
        data=generate_html_report(),
        file_name="informe_hortitrans.html",
        mime="text/html",
        help="Descarga un informe HTML con los parámetros actuales, los resultados de la simulación y todas las tablas rellenadas.",
    )
    st.markdown("---")

    # --- A1 ---
    with st.expander("📘 A1 — El efecto de la ventilación", expanded=True):
        st.markdown("""
**Objetivo:** Comprender cómo la apertura de ventanas controla la humedad interior.

**Escenario inicial:** Día soleado con cultivo en producción. Las ventanas están **cerradas**.

**Tarea:**
1. Carga el escenario y anota HR y DPV con apertura = 0 %.
2. Ve aumentando la apertura: 10 % → 25 % → 50 % → 75 % → 100 %. Registra HR y DPV en cada paso.
3. Identifica a partir de qué apertura el DPV entra en el rango óptimo (0,4–1,6 kPa).

**Reflexión:** ¿Qué ocurre con la Transpiración del cultivo al aumentar la ventilación? ¿Por qué?
        """)
        _p_a1 = {'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Normal',
                 'ss_LAI': 3.0, 'ss_T_o': 25.0, 'ss_RH_o': 70, 'ss_R_sol': 400, 'ss_viento': 2.0,
                 'ss_T_i': 28.0, 'ss_apertura': 0, 'ss_humidif': 0.0}
        activity_buttons("▶ Cargar escenario A1", "btn_a1", "rst_a1", _p_a1, 'a1')
        st.markdown("**📝 Tabla de registro:**")
        st.session_state['tbl_a1_data'] = st.data_editor(pd.DataFrame({
            'Apertura (%)':              [0, 10, 25, 50, 75, 100],
            'HR (%)':                    _nan(6),
            'DPV (kPa)':                 _nan(6),
            'Transpiración (g·m⁻²·h⁻¹)':_nan(6),
        }), key='tbl_a1', use_container_width=True, hide_index=True,
        column_config={
            'Apertura (%)':              _col_num(disabled=True),
            'HR (%)':                    _col_num(format="%.1f"),
            'DPV (kPa)':                 _col_num(format="%.2f"),
            'Transpiración (g·m⁻²·h⁻¹)':_col_num(format="%.2f"),
        })

    # --- A2 ---
    with st.expander("📘 A2 — ¿Qué pesa más, el viento o la diferencia de temperatura?"):
        st.markdown("""
**Objetivo:** Identificar cuál de los dos motores de la ventilación natural (viento o efecto chimenea) tiene mayor influencia.

**Escenario inicial:** Apertura al 50 %, diferencia T_interior − T_exterior = 13 °C, viento en calma.

**Tarea:**
1. Carga el escenario y anota el DPV inicial.
2. **Prueba viento:** Mantén T_interior = 28 °C y aumenta el viento de 0 a 10 m/s en pasos. Anota el DPV.
3. **Prueba chimenea:** Vuelve al escenario inicial (viento = 0). Ahora sube T_interior de 15 °C a 35 °C. Anota el DPV.
4. Compara los cambios de DPV en ambas pruebas.

**Reflexión:** ¿Cuál de los dos factores ventila más? ¿En qué época del año sería más útil cada uno?
        """)
        _p_a2 = {'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Normal',
                 'ss_LAI': 3.0, 'ss_T_o': 15.0, 'ss_RH_o': 70, 'ss_R_sol': 400, 'ss_viento': 0.0,
                 'ss_T_i': 28.0, 'ss_apertura': 50, 'ss_humidif': 0.0}
        activity_buttons("▶ Cargar escenario A2", "btn_a2", "rst_a2", _p_a2, 'a2')
        st.markdown("**📝 Prueba 1 — Efecto del viento** (T_int = 28 °C fija):")
        st.session_state['tbl_a2_v_data'] = st.data_editor(pd.DataFrame({
            'Viento (m/s)': [0, 2, 4, 6, 8, 10],
            'DPV (kPa)':    _nan(6),
        }), key='tbl_a2_v', use_container_width=True, hide_index=True,
        column_config={
            'Viento (m/s)': _col_num(disabled=True),
            'DPV (kPa)':    _col_num(format="%.2f"),
        })
        st.markdown("**📝 Prueba 2 — Efecto chimenea** (viento = 0 m/s fijo):")
        st.session_state['tbl_a2_t_data'] = st.data_editor(pd.DataFrame({
            'T interior (°C)': [15, 20, 25, 30, 35],
            'DPV (kPa)':       _nan(5),
        }), key='tbl_a2_t', use_container_width=True, hide_index=True,
        column_config={
            'T interior (°C)': _col_num(disabled=True),
            'DPV (kPa)':       _col_num(format="%.2f"),
        })

    # --- A3 ---
    with st.expander("📘 A3 — El grado de crecimiento del cultivo"):
        st.markdown("""
**Objetivo:** Entender cómo el desarrollo del cultivo (LAI) influye en la humedad del invernadero.

**Escenario inicial:** Condiciones de primavera con el invernadero recién trasplantado (LAI = 0, suelo desnudo).

**Tarea:**
1. Carga el escenario y anota HR, DPV y Transpiración con LAI = 0.
2. Aumenta el LAI paso a paso: 0 → 1 → 2 → 3 → 4 → 5 → 6. Registra los tres valores en cada paso.
3. Identifica a partir de qué LAI el DPV sale del rango óptimo.

**Reflexión:** ¿Por qué a medida que el cultivo crece hay que ajustar el manejo? ¿Qué parámetros cambiarías para mantener el DPV en rango con LAI = 6?
        """)
        _p_a3 = {'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Normal',
                 'ss_LAI': 0.0, 'ss_T_o': 22.0, 'ss_RH_o': 65, 'ss_R_sol': 500, 'ss_viento': 3.0,
                 'ss_T_i': 26.0, 'ss_apertura': 40, 'ss_humidif': 0.0}
        activity_buttons("▶ Cargar escenario A3", "btn_a3", "rst_a3", _p_a3, 'a3')
        st.markdown("**📝 Tabla de registro:**")
        st.session_state['tbl_a3_data'] = st.data_editor(pd.DataFrame({
            'LAI':                       [0, 1, 2, 3, 4, 5, 6],
            'HR (%)':                    _nan(7),
            'DPV (kPa)':                 _nan(7),
            'Transpiración (g·m⁻²·h⁻¹)':_nan(7),
        }), key='tbl_a3', use_container_width=True, hide_index=True,
        column_config={
            'LAI':                       _col_num(disabled=True),
            'HR (%)':                    _col_num(format="%.1f"),
            'DPV (kPa)':                 _col_num(format="%.2f"),
            'Transpiración (g·m⁻²·h⁻¹)':_col_num(format="%.2f"),
        })

    # --- A4 ---
    with st.expander("📘 A4 — El ciclo día-noche"):
        st.markdown("""
**Objetivo:** Analizar cómo cambian las condiciones higrotérmicas entre el día y la noche, e identificar los riesgos de cada período.

**Tarea:**
1. Carga el **escenario de día**: alta radiación, temperatura elevada, ventanas muy abiertas. Anota HR, DPV y condensación.
2. Carga el **escenario de noche**: radiación = 0, temperatura exterior baja, ventanas casi cerradas. Anota los mismos valores.
3. Compara ambos escenarios.

**Reflexión:** ¿En qué período hay mayor riesgo de condensación y enfermedades fúngicas? ¿Qué harías si no puedes abrir ventanas de madrugada por lluvia o frío?
        """)
        _p_a4d = {'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Normal',
                  'ss_LAI': 3.0, 'ss_T_o': 28.0, 'ss_RH_o': 55, 'ss_R_sol': 750, 'ss_viento': 4.0,
                  'ss_T_i': 32.0, 'ss_apertura': 70, 'ss_humidif': 0.0}
        _p_a4n = {'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Normal',
                  'ss_LAI': 3.0, 'ss_T_o': 14.0, 'ss_RH_o': 88, 'ss_R_sol': 0, 'ss_viento': 1.0,
                  'ss_T_i': 18.0, 'ss_apertura': 5, 'ss_humidif': 0.0}
        col_dia, col_noche = st.columns(2)
        with col_dia:
            if st.button("☀️ Cargar DÍA", key="btn_a4_dia"):
                load_scenario(_p_a4d, 'a4_dia')
            if st.button("🔄 Restablecer DÍA", key="rst_a4_dia"):
                load_scenario(_p_a4d, 'a4_dia')
        with col_noche:
            if st.button("🌙 Cargar NOCHE", key="btn_a4_noche"):
                load_scenario(_p_a4n, 'a4_noche')
            if st.button("🔄 Restablecer NOCHE", key="rst_a4_noche"):
                load_scenario(_p_a4n, 'a4_noche')
        show_scenario_info(['a4_dia', 'a4_noche'])
        st.markdown("**📝 Tabla de registro:**")
        st.session_state['tbl_a4_data'] = st.data_editor(pd.DataFrame({
            'Escenario':                  ['☀️ Día', '🌙 Noche'],
            'HR (%)':                     _nan(2),
            'DPV (kPa)':                  _nan(2),
            'Condensación (g·m⁻²·h⁻¹)':  _nan(2),
        }), key='tbl_a4', use_container_width=True, hide_index=True,
        column_config={
            'Escenario':                 _col_txt(disabled=True),
            'HR (%)':                    _col_num(format="%.1f"),
            'DPV (kPa)':                 _col_num(format="%.2f"),
            'Condensación (g·m⁻²·h⁻¹)': _col_num(format="%.2f"),
        })

    # ===== NIVEL 2 =====
    st.markdown("---")
    st.subheader("Nivel 2: Casos prácticos")
    st.markdown(
        "En estas actividades no hay una única respuesta correcta. "
        "El objetivo es tomar decisiones de manejo justificadas con los resultados del simulador."
    )

    # --- B1 ---
    with st.expander("📙 B1 — Gestión del riesgo de Botrytis"):
        st.markdown("""
**Objetivo:** Aprender a reducir el riesgo de enfermedades fúngicas en condiciones nocturnas de alta humedad.

**Escenario inicial:** Noche de otoño. Cultivo maduro, temperatura exterior baja y humedad exterior muy alta. Las ventanas están casi cerradas para conservar el calor.

**Tarea:**
1. Carga el escenario y comprueba los valores de HR, DPV y condensación. Identifica el riesgo.
2. Con las herramientas disponibles (apertura de ventanas, temperatura interior, humidificación/deshumidificación), busca una combinación que lleve el DPV al rango óptimo **sin provocar condensación**.
3. Anota la solución que hayas encontrado y justifícala.

**Restricción:** No puedes bajar la temperatura interior por debajo de 15 °C (riesgo de daño por frío al cultivo).

**Reflexión:** ¿Qué es más eficaz en este escenario, ventilar o calentar? ¿Por qué calentar reduce la HR?
        """)
        _p_b1 = {'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Normal',
                 'ss_LAI': 4.0, 'ss_T_o': 10.0, 'ss_RH_o': 92, 'ss_R_sol': 0, 'ss_viento': 1.0,
                 'ss_T_i': 16.0, 'ss_apertura': 5, 'ss_humidif': 0.0}
        activity_buttons("▶ Cargar escenario B1", "btn_b1", "rst_b1", _p_b1, 'b1')
        st.markdown("**📝 Registro de soluciones exploradas:**")
        st.session_state['tbl_b1_data'] = st.data_editor(pd.DataFrame({
            'Acción probada':             ['Escenario inicial', '', '', ''],
            'Apertura (%)':               [5.0] + _nan(3),
            'T int (°C)':                 [16.0] + _nan(3),
            'Humidif. (L·m⁻²·h⁻¹)':      [0.0] + _nan(3),
            'DPV (kPa)':                  _nan(4),
            'Condensación (g·m⁻²·h⁻¹)':  _nan(4),
        }), key='tbl_b1', use_container_width=True, hide_index=True,
        column_config={
            'Acción probada':            _col_txt(),
            'Apertura (%)':              _col_num(format="%.0f"),
            'T int (°C)':                _col_num(format="%.1f"),
            'Humidif. (L·m⁻²·h⁻¹)':     _col_num(format="%.2f"),
            'DPV (kPa)':                 _col_num(format="%.2f"),
            'Condensación (g·m⁻²·h⁻¹)': _col_num(format="%.2f"),
        })

    # --- B2 ---
    with st.expander("📙 B2 — Estrés hídrico en verano"):
        st.markdown("""
**Objetivo:** Identificar situaciones de estrés hídrico severo y evaluar qué acción de control es más eficaz.

**Escenario inicial:** Día de verano muy caluroso. Temperatura exterior alta, radiación intensa y viento escaso. Las ventanas están abiertas al 60 %.

**Tarea:**
1. Carga el escenario y comprueba el DPV. ¿Hay estrés hídrico?
2. Prueba cada una de estas acciones **por separado** volviendo al escenario inicial entre prueba y prueba:
   - a) Abrir las ventanas al máximo (100 %).
   - b) Humidificar con 0,3 L m⁻² h⁻¹.
   - c) Bajar la temperatura interior a 28 °C (simulando sombreo o refrigeración).
3. Registra el DPV resultante de cada acción y ordénalas de más a menos eficaz.

**Reflexión:** ¿Cuál de las tres acciones tiene mayor impacto? ¿Es viable en todos los invernaderos? ¿Qué ocurre con la transpiración en cada caso?
        """)
        _p_b2 = {'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Normal',
                 'ss_LAI': 3.0, 'ss_T_o': 38.0, 'ss_RH_o': 30, 'ss_R_sol': 900, 'ss_viento': 1.5,
                 'ss_T_i': 40.0, 'ss_apertura': 60, 'ss_humidif': 0.0}
        activity_buttons("▶ Cargar escenario B2", "btn_b2", "rst_b2", _p_b2, 'b2')
        st.markdown("**📝 Tabla de resultados por acción:**")
        st.session_state['tbl_b2_data'] = st.data_editor(pd.DataFrame({
            'Acción':                     ['Escenario inicial', 'a) Apertura 100 %',
                                           'b) Humidif. 0,3 L·m⁻²·h⁻¹', 'c) T_int = 28 °C'],
            'DPV (kPa)':                  _nan(4),
            'Transpiración (g·m⁻²·h⁻¹)': _nan(4),
            '¿Más eficaz?':               ['', '', '', ''],
        }), key='tbl_b2', use_container_width=True, hide_index=True,
        column_config={
            'Acción':                    _col_txt(disabled=True),
            'DPV (kPa)':                 _col_num(format="%.2f"),
            'Transpiración (g·m⁻²·h⁻¹)':_col_num(format="%.2f"),
            '¿Más eficaz?':              _col_txt(),
        })

    # --- B3 ---
    with st.expander("📙 B3 — Comparativa de estanqueidades"):
        st.markdown("""
**Objetivo:** Comprender cómo la construcción del invernadero (estanqueidad) afecta al control de la humedad, especialmente en condiciones de viento.

**Escenario inicial:** Día de primavera con viento moderado y ventanas cerradas.

**Tarea:**
1. Carga el escenario. Las ventanas están cerradas (apertura = 0 %).
2. Cambia únicamente el parámetro **Estanqueidad** y registra el DPV para cada nivel:
   - Muy Estanco → Normal → Con Fugas → Totalmente Estanco
3. Repite el proceso con viento = 8 m/s y compara los resultados.

**Reflexión:** ¿En qué situación climática tiene más impacto la estanqueidad? ¿Qué ventaja tiene un invernadero "Totalmente Estanco" en invierno? ¿Y qué inconveniente tiene en verano?
        """)
        _p_b3 = {'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Normal',
                 'ss_LAI': 3.0, 'ss_T_o': 18.0, 'ss_RH_o': 65, 'ss_R_sol': 350, 'ss_viento': 5.0,
                 'ss_T_i': 24.0, 'ss_apertura': 0, 'ss_humidif': 0.0}
        activity_buttons("▶ Cargar escenario B3", "btn_b3", "rst_b3", _p_b3, 'b3')
        st.markdown("**📝 Tabla comparativa de estanqueidades:**")
        st.session_state['tbl_b3_data'] = st.data_editor(pd.DataFrame({
            'Estanqueidad':          ['Muy Estanco', 'Normal', 'Con Fugas', 'Totalmente Estanco'],
            'DPV viento 5 m/s (kPa)': _nan(4),
            'DPV viento 8 m/s (kPa)': _nan(4),
        }), key='tbl_b3', use_container_width=True, hide_index=True,
        column_config={
            'Estanqueidad':           _col_txt(disabled=True),
            'DPV viento 5 m/s (kPa)': _col_num(format="%.2f"),
            'DPV viento 8 m/s (kPa)': _col_num(format="%.2f"),
        })

    # ===== NIVEL 3 =====
    st.markdown("---")
    st.subheader("Nivel 3: Diseño y toma de decisiones")
    st.markdown(
        "Actividades abiertas. No existe una única solución correcta: el objetivo es diseñar, "
        "argumentar y optimizar usando el simulador como herramienta de cálculo."
    )

    # --- C1 ---
    with st.expander("📕 C1 — Dimensionado del sistema de humidificación"):
        st.markdown("""
**Objetivo:** Calcular la necesidad de aporte de vapor en función del estado del cultivo, para dimensionar un sistema de humidificación (fog system) en invierno.

**Escenario de partida:** Invernadero totalmente estanco en invierno (ventanas cerradas, sin ventilación natural). Sin cultivo todavía.

**Tarea:**
1. Carga el escenario con LAI = 0. Ajusta la humidificación hasta que el DPV entre en rango óptimo. Anota los L m⁻² h⁻¹ necesarios.
2. Repite para LAI = 1, 2, 3, 4 y 5, anotando en cada caso la humidificación necesaria.
3. Construye una tabla con los resultados: LAI → L m⁻² h⁻¹ requeridos.
4. Observa si la tendencia es lineal o no y explica por qué.

**Reflexión:** ¿Para qué sirve conocer esta tabla en la práctica? ¿Cómo cambiaría si la temperatura interior fuera 5 °C más alta?
        """)
        _p_c1 = {'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Totalmente Estanco',
                 'ss_LAI': 0.0, 'ss_T_o': 8.0, 'ss_RH_o': 75, 'ss_R_sol': 150, 'ss_viento': 2.0,
                 'ss_T_i': 20.0, 'ss_apertura': 0, 'ss_humidif': 0.0}
        activity_buttons("▶ Cargar escenario C1", "btn_c1", "rst_c1", _p_c1, 'c1')
        st.markdown("**📝 Tabla de dimensionado:**")
        st.session_state['tbl_c1_data'] = st.data_editor(pd.DataFrame({
            'LAI':                         [0, 1, 2, 3, 4, 5],
            'Humidif. necesaria (L·m⁻²·h⁻¹)': _nan(6),
            'DPV obtenido (kPa)':          _nan(6),
        }), key='tbl_c1', use_container_width=True, hide_index=True,
        column_config={
            'LAI':                              _col_num(disabled=True),
            'Humidif. necesaria (L·m⁻²·h⁻¹)':  _col_num(format="%.2f"),
            'DPV obtenido (kPa)':               _col_num(format="%.2f"),
        })

    # --- C2 ---
    with st.expander("📕 C2 — Protocolo de manejo estacional"):
        st.markdown("""
**Objetivo:** Diseñar un protocolo de consignas de manejo (apertura de ventanas, temperatura interior, humidificación) para las tres estaciones de mayor actividad agrícola.

**Tarea:**
1. Carga el escenario de cada estación y, modificando los parámetros de control, encuentra la combinación que mantenga el DPV en rango óptimo con las menores pérdidas de energía.
2. Anota los valores de consigna y el DPV resultante en la tabla de registro que aparece más abajo.
3. Compara las tres estaciones y razona cuál presenta el mayor desafío de manejo.

**Reflexión:** ¿Qué parámetro de control cambia más entre estaciones? ¿Por qué en verano es tan difícil mantener el rango óptimo de DPV?
        """)
        _p_c2i = {'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Normal',
                  'ss_LAI': 3.0, 'ss_T_o': 7.0, 'ss_RH_o': 80, 'ss_R_sol': 180, 'ss_viento': 3.0,
                  'ss_T_i': 18.0, 'ss_apertura': 0, 'ss_humidif': 0.0}
        _p_c2p = {'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Normal',
                  'ss_LAI': 3.0, 'ss_T_o': 20.0, 'ss_RH_o': 65, 'ss_R_sol': 550, 'ss_viento': 3.0,
                  'ss_T_i': 24.0, 'ss_apertura': 30, 'ss_humidif': 0.0}
        _p_c2v = {'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Normal',
                  'ss_LAI': 3.0, 'ss_T_o': 35.0, 'ss_RH_o': 35, 'ss_R_sol': 850, 'ss_viento': 2.0,
                  'ss_T_i': 38.0, 'ss_apertura': 80, 'ss_humidif': 0.0}
        col_inv, col_pri, col_ver = st.columns(3)
        with col_inv:
            if st.button("❄️ Invierno",        key="btn_c2_inv"): load_scenario(_p_c2i, 'c2_inv')
            if st.button("🔄 Rst. Invierno",   key="rst_c2_inv"): load_scenario(_p_c2i, 'c2_inv')
        with col_pri:
            if st.button("🌱 Primavera",       key="btn_c2_pri"): load_scenario(_p_c2p, 'c2_pri')
            if st.button("🔄 Rst. Primavera",  key="rst_c2_pri"): load_scenario(_p_c2p, 'c2_pri')
        with col_ver:
            if st.button("☀️ Verano",          key="btn_c2_ver"): load_scenario(_p_c2v, 'c2_ver')
            if st.button("🔄 Rst. Verano",     key="rst_c2_ver"): load_scenario(_p_c2v, 'c2_ver')
        show_scenario_info(['c2_inv', 'c2_pri', 'c2_ver'])
        st.markdown("**📝 Ficha de protocolo estacional:**")
        st.session_state['tbl_c2_data'] = st.data_editor(pd.DataFrame({
            'Estación':               ['❄️ Invierno', '🌱 Primavera', '☀️ Verano'],
            'Apertura (%)':           _nan(3),
            'T interior (°C)':        _nan(3),
            'Humidif. (L·m⁻²·h⁻¹)':  _nan(3),
            'DPV resultante (kPa)':   _nan(3),
        }), key='tbl_c2', use_container_width=True, hide_index=True,
        column_config={
            'Estación':              _col_txt(disabled=True),
            'Apertura (%)':          _col_num(format="%.0f"),
            'T interior (°C)':       _col_num(format="%.1f"),
            'Humidif. (L·m⁻²·h⁻¹)': _col_num(format="%.2f"),
            'DPV resultante (kPa)':  _col_num(format="%.2f"),
        })

    # --- C3 ---
    with st.expander("📕 C3 — Optimización del diseño: ratio de ventilación"):
        st.markdown("""
**Objetivo:** Determinar qué superficie de ventanas es necesaria para mantener un DPV óptimo en las peores condiciones de verano, y obtener el ratio ventana/suelo mínimo recomendable.

**Escenario de partida:** Peor caso de verano: alta temperatura, radiación máxima, viento escaso, cultivo en plena producción y ventanas abiertas al 100 %.

**Tarea:**
1. Carga el escenario. Con el área de ventanas en 400 m² (ratio 5 %), anota el DPV en la tabla.
2. Aumenta el área de ventanas progresivamente: 400 → 800 → 1200 → 1600 → 2000 m² (ratios 5–25 %).
3. Registra el DPV en cada caso y determina a partir de qué área de ventanas el DPV entra en rango óptimo.
4. Calcula el **ratio ventana/suelo** (%) mínimo recomendado para este tipo de clima.

**Reflexión:** ¿Qué consecuencias tiene construir un invernadero con un ratio de ventilación insuficiente? ¿Cómo afectaría a los costes de producción?
        """)
        _p_c3 = {'ss_area_suelo': 8000, 'ss_area_ventanas': 400, 'ss_estanqueidad': 'Normal',
                 'ss_LAI': 4.0, 'ss_T_o': 36.0, 'ss_RH_o': 30, 'ss_R_sol': 950, 'ss_viento': 1.5,
                 'ss_T_i': 39.0, 'ss_apertura': 100, 'ss_humidif': 0.0}
        activity_buttons("▶ Cargar escenario C3", "btn_c3", "rst_c3", _p_c3, 'c3')
        st.markdown("**📝 Tabla de optimización:**")
        st.session_state['tbl_c3_data'] = st.data_editor(pd.DataFrame({
            'Área ventanas (m²)': [400, 800, 1200, 1600, 2000],
            'Ratio (%):':         [5,   10,  15,   20,   25  ],
            'DPV (kPa)':          _nan(5),
            '¿En rango?':         [''] * 5,
        }), key='tbl_c3', use_container_width=True, hide_index=True,
        column_config={
            'Área ventanas (m²)': _col_num(disabled=True),
            'Ratio (%):':         _col_num(disabled=True, format="%.0f"),
            'DPV (kPa)':          _col_num(format="%.2f"),
            '¿En rango?':         _col_txt(),
        })

# --- FOOTER ---
footer_html = """
<div style="text-align: center;">
    <p>©</p>
    <p>María Alonso</p>
    <p>Francisco Javier Criado</p>
    <p>Joaquín Hernández</p>
    <p>Licencia <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">Creative Commons Attribution 4.0</a></p>
</div>
"""
st.sidebar.markdown(footer_html, unsafe_allow_html=True)
