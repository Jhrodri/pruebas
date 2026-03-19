import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
from pathlib import Path
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
st.title("💧 Simulador de humedad y transpiración en Invernaderos...**EN PRUEBAS**")
st.info("Este simulador no está en explotación y puede tener errores.")

with st.expander("ℹ️ ¿Qué se va calcular?"):
    st.markdown("""
    - Este simulador está adaptado del Modelo HORTITRANS de O. Jolliet
    - Se realiza un balance de masa de vapor y se resuelve analíticamente la presión de vapor en estado estacionario.
    - El modelo inicial necesita una temperatura interior fija por lo que es de uso directo en invernaderos climatizados.
    - Para invernaderos no climatizados se calcula el equilibrio pasivo que se alcanzaría si la temperatura interior no fuese fija despejando iterativamente la temperatura a partir de un balance de energía.
    - Como consigna se ha establecido el rango 0,4-1,6 kPa.
    """)

# --- Session State: Actividad 1 ---
_ACT1_APERTURAS = [0, 20, 40, 60, 80, 100]
_ACT1_COLS = ['HR (%)', 'DPV (kPa)', 'Transp. (g·m⁻²·h⁻¹)', 'Vent. (g·m⁻²·h⁻¹)', 'Cond. (g·m⁻²·h⁻¹)']

def _act1_default_df():
    return pd.DataFrame({
        'Apertura (%)': _ACT1_APERTURAS,
        'Tint (°C)': [30.0] * 6,
        'HR (%)': [None] * 6,
        'DPV (kPa)': [None] * 6,
        'Transp. (g·m⁻²·h⁻¹)': [None] * 6,
        'Vent. (g·m⁻²·h⁻¹)': [None] * 6,
        'Cond. (g·m⁻²·h⁻¹)': [None] * 6,
    }).astype({c: 'Float64' for c in _ACT1_COLS})

if 'act1_tabla' not in st.session_state:
    st.session_state['act1_tabla'] = _act1_default_df()
if 'act1_show_charts' not in st.session_state:
    st.session_state['act1_show_charts'] = False
if 'act1_nombre' not in st.session_state:
    st.session_state['act1_nombre'] = ''
for _i in range(1, 6):
    if f'act1_resp_{_i}' not in st.session_state:
        st.session_state[f'act1_resp_{_i}'] = ''

_ACT1_PREGUNTAS = [
    (
        "El intercambio de masas",
        "Observa la columna de HR. ¿Por qué disminuye la humedad relativa al abrir las ventanas "
        "si la temperatura interior (Tint) se mantiene exactamente igual en 30 °C? "
        "Justifica qué está ocurriendo.",
    ),
    (
        "El punto de saturación",
        "En la apertura 0, el valor de Cond. es de 64,47, pero en la apertura 20 cae a 0. "
        "Explica la relación que existe entre la Humedad Relativa (HR) y la aparición de agua "
        "líquida (condensación) en las paredes del invernadero.",
    ),
    (
        'La "fuerza" del aire',
        "El DPV aumenta conforme abrimos las ventanas. Explica con tus palabras qué le está "
        '"haciendo" el aire a la planta cuando el DPV sube de 0,03 a 1,47 y cómo se refleja eso '
        "en la columna de Transp.",
    ),
    (
        "Balance hídrico",
        "Si comparas la columna de Transp. (lo que la planta expulsa) con la de Vent. "
        "(lo que el aire saca al exterior), verás que a partir del 20 % de apertura, la "
        "ventilación es mucho más alta. ¿De dónde sale ese exceso de agua que la ventilación "
        "está moviendo si la planta transpirará mucho menos de lo que se ventila?",
    ),
    (
        "Control de variables",
        "Para esta simulación se ha decidido dejar la Tint fija en 30 °C. ¿Por qué crees que "
        "es necesario mantener la temperatura constante para entender el efecto real de la "
        "apertura de ventanas sobre la humedad y el DPV?",
    ),
]

# (df_col, st_label, pdf_label, color)
_CHART_SPECS = [
    ('HR (%)',               'Humedad Relativa (%)',      'Humedad Relativa (%)',  '#1f77b4'),
    ('DPV (kPa)',            'DPV (kPa)',                 'DPV (kPa)',             '#d62728'),
    ('Transp. (g·m⁻²·h⁻¹)', 'Transpiración (g·m⁻²·h⁻¹)', 'Transpiracion (g/m2/h)', '#2ca02c'),
    ('Vent. (g·m⁻²·h⁻¹)',   'Ventilación (g·m⁻²·h⁻¹)',  'Ventilacion (g/m2/h)',  '#ff7f0e'),
]


def _make_chart_buf(x, y, title, color):
    """Genera una imagen PNG de un gráfico matplotlib y la devuelve como BytesIO."""
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    mask = ~pd.isna(y)
    if mask.any():
        ax.plot(x[mask], y[mask], 'o-', color=color, linewidth=2, markersize=5)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Apertura de ventanas (%)')
    ax.set_ylabel(title)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_chart_buf_gen(x, y, title, color, xlabel, xticks=None):
    """Versión generalizada de _make_chart_buf con xlabel configurable."""
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    mask = ~pd.isna(y)
    if mask.any():
        ax.plot(x[mask], y[mask], 'o-', color=color, linewidth=2, markersize=5)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


# --- Session State y datos: Actividad 2 ---
_ACT2_LAI_VALS   = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
_ACT2_TINT_VALS  = [33.0, 32.5, 32.0, 31.5, 31.0, 30.5, 30.0]
_ACT2_RAD_VALS   = [900, 700, 500, 300, 120]
_ACT2_HUMID_VALS = [0.0, 0.5, 1.0, 1.4]

_ACT2_COLS_A = ['Tequil. (°C)', 'DPV equil. (kPa)', 'Transp. (g·m⁻²·h⁻¹)']
_ACT2_COLS_B = ['DPV (kPa)']
_ACT2_COLS_C = ['Tequil. (°C)', 'DPV equil. (kPa)']
_ACT2_COLS_D = ['Transp. (g·m⁻²·h⁻¹)', 'HR (%)', 'DPV (kPa)']


def _act2_default_df_a():
    return pd.DataFrame({
        'LAI': _ACT2_LAI_VALS,
        'Tequil. (°C)': [None] * 7,
        'DPV equil. (kPa)': [None] * 7,
        'Transp. (g·m⁻²·h⁻¹)': [None] * 7,
    }).astype({c: 'Float64' for c in _ACT2_COLS_A})


def _act2_default_df_b():
    return pd.DataFrame({
        'Tint (°C)': _ACT2_TINT_VALS,
        'DPV (kPa)': [None] * 7,
    }).astype({'DPV (kPa)': 'Float64'})


_ACT2_SOMBREO = [int(round((900 - r) / 900 * 100)) for r in _ACT2_RAD_VALS]


def _act2_default_df_c():
    return pd.DataFrame({
        'Rad. ext. (W/m²)': _ACT2_RAD_VALS,
        '% Sombreo': _ACT2_SOMBREO,
        'Tequil. (°C)': [None] * 5,
        'DPV equil. (kPa)': [None] * 5,
    }).astype({c: 'Float64' for c in _ACT2_COLS_C})


def _act2_default_df_d():
    return pd.DataFrame({
        'Humid. (L·m⁻²·h⁻¹)': _ACT2_HUMID_VALS,
        'Transp. (g·m⁻²·h⁻¹)': [None] * 4,
        'HR (%)': [None] * 4,
        'DPV (kPa)': [None] * 4,
    }).astype({c: 'Float64' for c in _ACT2_COLS_D})


for _k2, _v2 in [
    ('act2_tabla_a', _act2_default_df_a()),
    ('act2_tabla_b', _act2_default_df_b()),
    ('act2_tabla_c', _act2_default_df_c()),
    ('act2_tabla_d', _act2_default_df_d()),
    ('act2_show_charts_a', False),
    ('act2_show_charts_b', False),
    ('act2_show_charts_c', False),
    ('act2_show_charts_d', False),
    ('act2_nombre', ''),
]:
    if _k2 not in st.session_state:
        st.session_state[_k2] = _v2

for _i in range(1, 4):
    if f'act2_resp_a_{_i}' not in st.session_state:
        st.session_state[f'act2_resp_a_{_i}'] = ''
for _i in range(1, 3):
    if f'act2_resp_d_{_i}' not in st.session_state:
        st.session_state[f'act2_resp_d_{_i}'] = ''

# --- Session State y datos: Actividad 3 ---
_ACT3_COL_TRANSP = 'Transp. (g\u00b7m\u207b\u00b2\u00b7h\u207b\u00b9)'
_ACT3_COL_VENT   = 'Vent. (g\u00b7m\u207b\u00b2\u00b7h\u207b\u00b9)'
_ACT3_COL_TINT   = 'Tinterior (\u00b0C)'
_ACT3_COL_TEQUIL = 'Tequil. (\u00b0C)'
_ACT3_COL_TEXT   = 'Text (\u00b0C)'
_ACT3_FLOAT_COLS = [
    'HR (%)', 'DPV (kPa)', _ACT3_COL_TRANSP, _ACT3_COL_VENT,
    _ACT3_COL_TEXT, _ACT3_COL_TINT, _ACT3_COL_TEQUIL, 'DPVequil. (kPa)',
]


def _act3_default_df_a():
    df = pd.DataFrame({
        'HR (%)': [None],
        'DPV (kPa)': [None],
        _ACT3_COL_TRANSP: [None],
        _ACT3_COL_VENT: [None],
        _ACT3_COL_TEXT: [11.0],
        _ACT3_COL_TINT: [15.0],
        _ACT3_COL_TEQUIL: [None],
        'DPVequil. (kPa)': [None],
    })
    return df.astype({c: 'Float64' for c in _ACT3_FLOAT_COLS})


def _act3_default_df_b():
    df = pd.DataFrame({
        'HR (%)': [None],
        'DPV (kPa)': [None],
        _ACT3_COL_TRANSP: [None],
        _ACT3_COL_VENT: [None],
        _ACT3_COL_TEXT: [11.0],
        _ACT3_COL_TINT: [15.0],
        _ACT3_COL_TEQUIL: [None],
        'DPVequil. (kPa)': [None],
    })
    return df.astype({c: 'Float64' for c in _ACT3_FLOAT_COLS})


for _k3, _v3 in [
    ('act3_tabla_a', _act3_default_df_a()),
    ('act3_tabla_b', _act3_default_df_b()),
    ('act3_nombre', ''),
]:
    if _k3 not in st.session_state:
        st.session_state[_k3] = _v3

for _i in range(1, 5):
    if f'act3_resp_a_{_i}' not in st.session_state:
        st.session_state[f'act3_resp_a_{_i}'] = ''
for _i in range(1, 5):
    if f'act3_resp_b_{_i}' not in st.session_state:
        st.session_state[f'act3_resp_b_{_i}'] = ''

_ACT3_PREGUNTAS_A = [
    (
        "La transpiración nocturna",
        "¿Por qué la transpiración es cero? ¿Afectaría el LAI a ese valor?",
    ),
    (
        "La ventilación con ventanas cerradas",
        "¿A qué crees que puede deberse que, aunque las ventanas estén cerradas, "
        "haya ventilación? ¿Cómo es posible si las ventanas están a 0 % de apertura?",
    ),
    (
        "La inversión térmica",
        "Analiza la temperatura exterior y la temperatura de equilibrio. ¿Qué está pasando? "
        "¿Es normal que la temperatura de equilibrio sea inferior a la temperatura exterior?",
    ),
    (
        "El efecto de abrir las ventanas",
        "Como se ha visto, en condiciones de noche despejada (cielo sin nubes) la temperatura "
        "interior puede ser menor que la exterior. Abre las ventanas en el simulador y describe "
        "qué ocurre. ¿Es recomendable ventilar en estas condiciones?",
    ),
]

_ACT3_PREGUNTAS_B = [
#    (
#        "El efecto del tipo de cielo",
#        "¿Por qué únicamente cambia la temperatura de equilibrio al pasar de cielo despejado "
#        "a cielo nublado? ¿Qué parámetro del modelo es el responsable de ese cambio?",
#    ),
    (
        "La manta de nubes",
        "¿A qué se debe que la temperatura de equilibrio con cielo nublado sea mayor que la temperatura "
        "exterior? Explica el mecanismo físico.",
    ),
    (
        "Ventilación en cielo nublado",
        "Abre ahora las ventanas en el simulador y explica qué ocurre. "
        "¿Es diferente al caso de cielo despejado? ¿Por qué?",
    ),
    (
        "Decisión de manejo",
        "En condiciones de noche nublada, ¿abrirías las ventanas? Justifica tu respuesta "
        "teniendo en cuenta la temperatura de equilibrio y la temperatura interior.",
    ),
]

_ACT3_EXPLICACION = """
Durante la noche, el invernadero deja de ganar energía solar y empieza a perderla hacia el exterior. El comportamiento de las variables cambia radicalmente respecto al día.

TEMPERATURA
Cae de forma continua al no haber fuente de calor solar. La temperatura interior depende del tipo de cielo y de las características del material de cerramiento. Si se usan plásticos térmicos puede llegarse a conseguir incrementar algunos grados la temperatura interior con respecto al exterior.

DPV
Se desploma hasta valores cercanos a 0. Al enfriarse el aire pierde su capacidad de contener vapor y se satura (la Humedad Relativa roza el 100 %).

TRANSPIRACIÓN
Es mínima o nula. Los estomas de las plantas suelen estar cerrados y, además, el aire está tan saturado que no tiene "fuerza" para evaporar agua de las hojas.

CIELO DESPEJADO VS. CIELO NUBLADO
Sin plásticos térmicos y en condiciones de cielo claro, dentro de los invernaderos puede darse la inversión térmica: temperatura interior menor que la exterior. Cuando la noche es despejada el calor emitido por el invernadero escapa directamente hacia la atmósfera. Si el cielo está nublado, las nubes actúan como una "manta" que refleja el calor de vuelta a la superficie, manteniendo temperaturas de equilibrio más altas (por encima de la temperatura exterior).

CONSECUENCIA PRÁCTICA
En una noche despejada con inversión térmica, abrir las ventanas empeoraría las condiciones: el aire exterior está más caliente que el interior, por lo que entrar aire exterior calentaría el invernadero pero también aumentaría el DPV. En una noche nublada sin inversión térmica, la apertura de ventanas puede ser beneficiosa para renovar el aire húmedo, reducir el riesgo de enfermedades fúngicas y aproximar la temperatura interior a la de equilibrio.
"""

_ACT2_PREGUNTAS_A = [
    (
        "El efecto refrigerante",
        "A medida que el LAI (índice de área foliar) aumenta de 0,1 a 3,0, la temperatura "
        "de equilibrio baja 3 grados (de 37 °C a 34 °C). ¿A qué proceso biológico se debe "
        "este enfriamiento del aire?",
    ),
    (
        "Relación entre masa vegetal y humedad",
        "El DPV disminuye significativamente conforme el cultivo es más grande. Explica por "
        "qué disminuye el DPV cuando hay más hojas presentes en el invernadero.",
    ),
    (
        "La paradoja de la transpiración",
        "Si el DPV baja, el aire tiene menos \"fuerza\" para \"succionar\" agua de las hojas. "
        "Sin embargo, la transpiración total del sistema pasa de 20,35 a 473,79 g·m⁻²·h⁻¹. "
        "¿Cómo es posible que la transpiración total suba tanto si el aire está cada vez más húmedo?",
    ),
]

_ACT2_PREGUNTAS_D = [
    (
        "El control del ambiente",
        "Al aumentar la humidificación de 0 a 1,4 L·m⁻²·h⁻¹, la Humedad Relativa (HR) sube "
        "y el DPV baja. ¿Por qué el aire se vuelve \"menos agresivo\" para la planta al añadir "
        "agua al ambiente? Explica el proceso físico.",
    ),
    (
        "La respuesta de la planta",
        "Observa la columna de Transpiración. A pesar de que hay la misma cantidad de plantas, "
        "la transpiración disminuye conforme aumentamos la humidificación. ¿Por qué la planta "
        "\"suelta\" menos agua cuando se activan los nebulizadores?",
    ),
]


# --- Datos: Actividad 4 — Test de respuesta múltiple ---
_ACT4_PREGUNTAS = [
    {
        "pregunta": (
            "¿Qué ocurre con la Humedad Relativa (HR) interior cuando se abren las "
            "ventanas manteniendo la temperatura interior constante?"
        ),
        "opciones": [
            "Aumenta, porque entra aire exterior más húmedo.",
            "No cambia, porque la temperatura es la misma.",
            "Disminuye, porque el aire exterior menos saturado diluye el vapor interior.",
            "Aumenta, porque la ventilación favorece la transpiración del cultivo.",
        ],
        "correcta": 2,
        "explicacion": (
            "Al abrir las ventanas, el aire interior (más húmedo) se mezcla con el aire "
            "exterior (menos saturado). Aunque la temperatura es la misma, la cantidad "
            "absoluta de vapor de agua en el interior disminuye, y con ello la HR."
        ),
    },
    {
        "pregunta": "¿Qué mide el Déficit de Presión de Vapor (DPV)?",
        "opciones": [
            "La cantidad absoluta de vapor de agua en el aire (g/m³).",
            "La diferencia entre la presión de vapor de saturación y la presión de vapor real.",
            "La velocidad a la que se condensa el agua en la cubierta.",
            "La temperatura a la que el aire empieza a condensar (punto de rocío).",
        ],
        "correcta": 1,
        "explicacion": (
            "El DPV = Psat(T) − Preal. Cuantifica la \"sed\" del aire: cuánto vapor de "
            "agua más puede absorber antes de saturarse. A mayor DPV, mayor estrés hídrico "
            "para la planta."
        ),
    },
    {
        "pregunta": (
            "¿Cuál es el rango de DPV generalmente considerado óptimo "
            "para la mayoría de los cultivos hortícolas?"
        ),
        "opciones": [
            "0 – 0,4 kPa",
            "0,4 – 1,6 kPa",
            "1,6 – 3,0 kPa",
            "3,0 – 5,0 kPa",
        ],
        "correcta": 1,
        "explicacion": (
            "Por debajo de 0,4 kPa el aire está demasiado húmedo y pueden aparecer "
            "enfermedades fúngicas. Por encima de 1,6 kPa la planta cierra estomas "
            "para evitar la deshidratación, reduciendo la fotosíntesis y el crecimiento."
        ),
    },
    {
        "pregunta": "¿Qué representa el LAI (Leaf Area Index o Índice de Área Foliar)?",
        "opciones": [
            "La temperatura media de las hojas del cultivo.",
            "El cociente entre el área total de hojas y el área de suelo cultivado.",
            "La cantidad de luz solar absorbida por el cultivo.",
            "La velocidad de transpiración por unidad de superficie foliar.",
        ],
        "correcta": 1,
        "explicacion": (
            "LAI = Área foliar total / Área de suelo. Un LAI de 3 significa que hay "
            "3 m² de hojas por cada m² de suelo. A mayor LAI, mayor superficie "
            "transpirante y mayor capacidad de refrigeración evaporativa."
        ),
    },
    {
        "pregunta": (
            "¿Por qué un mayor LAI reduce la temperatura de equilibrio "
            "del invernadero en condiciones de verano?"
        ),
        "opciones": [
            "Porque las hojas reflejan la radiación solar hacia el exterior.",
            "Porque la sombra generada por el cultivo enfría el suelo.",
            "Porque la transpiración foliar consume calor latente y enfría el aire.",
            "Porque más hojas absorben más CO₂ y ese proceso genera frío.",
        ],
        "correcta": 2,
        "explicacion": (
            "La transpiración es un proceso de cambio de fase: el agua líquida en las hojas "
            "pasa a vapor consumiendo aproximadamente 2 450 J por gramo evaporado "
            "(calor latente de vaporización). Ese calor se \"roba\" al aire, enfriándolo."
        ),
    },
    {
        "pregunta": "¿Qué es la inversión térmica nocturna en un invernadero?",
        "opciones": [
            "Que la temperatura sube más rápido por la noche que durante el día.",
            "Que la cubierta está más caliente que el aire interior.",
            "Que la temperatura interior es menor que la temperatura exterior.",
            "Que el DPV interior supera al DPV exterior.",
        ],
        "correcta": 2,
        "explicacion": (
            "Con cielo despejado, el invernadero emite radiación de onda larga "
            "directamente hacia la atmósfera. El enfriamiento radiativo es tan intenso "
            "que la temperatura de equilibrio puede caer por debajo de la temperatura "
            "exterior, especialmente en invernaderos con plástico no térmico."
        ),
    },
    {
        "pregunta": (
            "¿Por qué la transpiración del cultivo es prácticamente nula "
            "durante la noche, aunque el LAI sea alto?"
        ),
        "opciones": [
            "Porque las raíces dejan de absorber agua al bajar la temperatura del suelo.",
            "Porque los estomas se cierran y el DPV nocturno es casi cero.",
            "Porque la condensación en hojas compensa la transpiración.",
            "Porque la presión atmosférica nocturna impide la salida de vapor.",
        ],
        "correcta": 1,
        "explicacion": (
            "Durante la noche los estomas se cierran (respuesta al ciclo circadiano y a "
            "la falta de luz). Además, el aire interior está casi saturado (DPV ≈ 0), "
            "por lo que no tiene capacidad de absorber más vapor: la \"fuerza\" de "
            "succión sobre las hojas es nula."
        ),
    },
    {
        "pregunta": (
            "¿Qué efecto tiene el cielo nublado (respecto al despejado) "
            "sobre la temperatura de equilibrio nocturna del invernadero?"
        ),
        "opciones": [
            "La reduce, porque las nubes bloquean la llegada de calor.",
            "No tiene efecto: por la noche el tipo de cielo es irrelevante.",
            "La aumenta, porque las nubes reflejan de vuelta la radiación de onda larga.",
            "La reduce, porque con nubes aumenta la velocidad del viento.",
        ],
        "correcta": 2,
        "explicacion": (
            "Las nubes actúan como una 'manta': absorben la radiación de onda larga "
            "emitida por el invernadero y la reemiten hacia la superficie. Esto frena "
            "el enfriamiento nocturno y eleva la temperatura de equilibrio por encima "
            "de la temperatura exterior."
        ),
    },
    {
        "pregunta": (
            "¿Por qué la refrigeración evaporativa (nebulización) "
            "es más eficiente que el sombreado para reducir el DPV en verano?"
        ),
        "opciones": [
            "Porque la nebulización no consume energía eléctrica.",
            "Porque al evaporar agua líquida se consume calor latente, "
            "enfriando el aire y aumentando su humedad simultáneamente.",
            "Porque el sombreado aumenta la temperatura interior al bloquear la ventilación.",
            "Porque la nebulización aumenta la velocidad del viento interior.",
        ],
        "correcta": 1,
        "explicacion": (
            "El sombreado reduce la energía entrante pero también reduce la fotosíntesis. "
            "Para bajar el DPV por debajo de 1,6 kPa solo con sombreado se necesitaría "
            "eliminar más del 85 % de la radiación. La nebulización, en cambio, consume "
            "calor latente (~2 450 J/g) enfriando y humidificando el aire con impacto "
            "mínimo sobre la radiación disponible para el cultivo."
        ),
    },
    {
        "pregunta": (
            "¿En qué condición aparece condensación en la cubierta del invernadero?"
        ),
        "opciones": [
            "Cuando la temperatura interior supera los 35 °C.",
            "Cuando la velocidad del viento exterior supera 3 m/s.",
            "Cuando la Humedad Relativa interior alcanza el 100 % (aire saturado).",
            "Cuando el DPV supera 1,6 kPa.",
        ],
        "correcta": 2,
        "explicacion": (
            "La condensación ocurre cuando el aire llega al punto de saturación "
            "(HR = 100 %). En ese momento cualquier superficie fría (cubierta, paredes) "
            "provoca que el vapor se deposite como agua líquida. Esto genera goteo "
            "sobre el cultivo y favorece el desarrollo de enfermedades fúngicas."
        ),
    },
]

# Session state Actividad 4
for _i in range(len(_ACT4_PREGUNTAS)):
    if f'act4_resp_{_i}' not in st.session_state:
        st.session_state[f'act4_resp_{_i}'] = None
if 'act4_submitted' not in st.session_state:
    st.session_state['act4_submitted'] = False
if 'act4_nombre' not in st.session_state:
    st.session_state['act4_nombre'] = ''


def _pdf_str(s):
    """Convierte texto Unicode a cadena compatible con la fuente Helvetica (Latin-1).
    Sustituye superíndices y otros caracteres fuera del rango Latin-1."""
    _MAP = {
        '\u207b': '-',   # ⁻  superscript minus
        '\u207f': 'n',   # ⁿ  superscript n
        '\u00b7': '.',   # ·  middle dot
        '\u2212': '-',   # −  minus sign
        '\u2013': '-',   # –  en dash
        '\u2014': '-',   # —  em dash
        '\u2018': "'",   # '  left single quote
        '\u2019': "'",   # '  right single quote / apostrophe
        '\u201c': '"',   # "  left double quote
        '\u201d': '"',   # "  right double quote
        '\u2026': '...',  # …  ellipsis
        '\u00b2': '2',   # ²  superscript 2
        '\u00b9': '1',   # ¹  superscript 1
        '\u00b3': '3',   # ³  superscript 3
    }
    result = ''.join(_MAP.get(c, c) for c in str(s))
    # Eliminar cualquier carácter que siga fuera del rango Latin-1
    return result.encode('latin-1', errors='replace').decode('latin-1')


def _generate_pdf(nombre, df_tabla, respuestas):
    """Genera el PDF de la actividad y lo devuelve como bytes."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Página 1: encabezado + tabla de datos ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'ACTIVIDAD 1: Efecto de la ventilación', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.set_font('Helvetica', '', 11)
#    pdf.cell(0, 7, 'Simulador HORTITRANS', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.ln(4)

    if nombre:
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(35, 8, 'Nombre:')
        pdf.set_font('Helvetica', '', 11)
        pdf.cell(0, 8, nombre, new_x='LMARGIN', new_y='NEXT')
        pdf.ln(3)

    # Escenario
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Escenario por defecto', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.set_fill_color(240, 240, 240)
    pdf.multi_cell(
        0, 6,
        'Área suelo: 8000 m2  |  Ventanas: 1600 m2  |  Estanqueidad: Normal  |  LAI: 3\n'
        'T exterior: 25 C  |  HR exterior: 80%  |  Radiación: 400 W/m2  |  Viento: 2 m/s\n'
        'T interior (fija): 30 C  |  Cielo: Despejado',
        border=1, fill=True,
    )
    pdf.ln(5)

    # Tabla de datos
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'A) Tabla de valores (apertura de ventanas)', new_x='LMARGIN', new_y='NEXT')

    col_w   = [24, 18, 22, 22, 30, 30, 30]
    headers = ['Apertura\n(%)', 'Tint\n(C)', 'HR\n(%)', 'DPV\n(kPa)',
               'Transp.\n(g/m2/h)', 'Vent.\n(g/m2/h)', 'Cond.\n(g/m2/h)']

    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(220, 230, 242)
    for w, h in zip(col_w, headers):
        pdf.multi_cell(w, 5, h, border=1, align='C', fill=True,
                       new_x='RIGHT', new_y='TOP', max_line_height=5)
    pdf.ln(10)

    pdf.set_font('Helvetica', '', 9)
    all_cols = ['Apertura (%)', 'Tint (°C)'] + _ACT1_COLS
    for _, row in df_tabla.iterrows():
        for j, col in enumerate(all_cols):
            val = row[col]
            if pd.isna(val):
                txt = ''
            elif col == 'Apertura (%)':
                txt = f'{int(val)}'
            elif col == 'Tint (°C)':
                txt = f'{val:.0f}'
            else:
                txt = f'{float(val):.2f}'
            pdf.cell(col_w[j], 8, txt, border=1, align='C')
        pdf.ln()
    pdf.ln(4)

    # ── Página 2: gráficos ──
    mask = df_tabla[_ACT1_COLS].notna().any(axis=1)
    df_plot = df_tabla[mask].copy()

    if not df_plot.empty:
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, 'Graficos de evolucion', new_x='LMARGIN', new_y='NEXT')
        pdf.ln(3)

        x = df_plot['Apertura (%)'].values
        bufs = [
            _make_chart_buf(x, df_plot[dc].values, pl, color)
            for dc, _, pl, color in _CHART_SPECS
        ]
        positions = [(10, 28), (110, 28), (10, 138), (110, 138)]
        for buf, (cx, cy) in zip(bufs, positions):
            pdf.image(buf, x=cx, y=cy, w=92)

    # ── Página 3+: cuestionario ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Cuestionario', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(4)

    for idx, (titulo, pregunta) in enumerate(_ACT1_PREGUNTAS, 1):
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 7, f'{idx}. {titulo}', new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        pdf.multi_cell(0, 5, pregunta)
        pdf.ln(2)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 5, 'Respuesta:', new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        resp = respuestas[idx - 1]
        if resp.strip():
            pdf.multi_cell(0, 5, resp, border=1)
        else:
            pdf.multi_cell(0, 18, '', border=1)
        pdf.ln(5)

    # ── Explicacion ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(0, 9, 'Explicación', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(3)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 7, 'El DPV: El concepto clave', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5,
        'El DPV (Déficit de Presión de Vapor) indica la diferencia entre la cantidad de '
        'humedad que el aire tiene y la que podría tener si estuviera saturado.')
    pdf.ln(2)
    pdf.multi_cell(0, 5,
        '- DPV bajo (0,03 kPa): El aire está "lleno". No "tira" del agua de la planta.\n'
        '- DPV alto (1,47 kPa): El aire no está "lleno". Cuanto más abrimos las ventanas, '
        'más sube el DPV, lo que significa que el ambiente tiene más capacidad de absorber agua.')
    pdf.ln(3)
    pdf.multi_cell(0, 5,
        'Cuando las ventanas estan cerradas (Apertura 0), el aire está casi saturado '
        '(99,2 % de humedad relativa y DPV casi cero). Como el aire no puede contener más '
        'vapor de agua, este se convierte en líquido sobre las paredes o plantas: eso es la '
        'condensación (devuelve 64,47 g de agua por cada metro cuadrado del invernadero cada hora).')
    pdf.ln(3)
    pdf.multi_cell(0, 5,
        'Que ocurre al abrir al 20 %?\n'
        'Por un lado entra aire del exterior que está más seco que el aire interior '
        '(en el exterior la temperatura es de 25 C y la HR es del 80 %, por lo que su DPV '
        'es de 0,6 kPa); pero además sale aire húmedo (el invernadero pierde 325 g de agua '
        'por cada metro cuadrado de suelo y hora). Al bajar la humedad, el fenómeno de la '
        'condensación desaparece por completo (pasa a 0). Esto es vital para evitar hongos '
        'en los cultivos.')
    pdf.ln(3)
    pdf.multi_cell(0, 5,
        'Conforme se va aumentando el porcentaje de apertura de ventanas sigue disminuyendo '
        'la humedad relativa y aumentando el DPV, la transpiración y el agua perdida por '
        'ventilación, pero cada vez los decrementos e incrementos son menores '
        '(la respuesta no es lineal). Se puede llegar incluso a tener dentro del '
        'invernadero menor humedad y mayor DPV que en el exterior, porque la temperatura '
        'interior esta fijada a 30 C.')
    pdf.ln(4)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 7, 'La Transpiración del cultivo (Transp.)', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5,
        'La transpiración es el proceso por el cual las plantas pierden agua en forma de '
        'vapor por sus estomas.')
    pdf.ln(2)
    pdf.multi_cell(0, 5,
        '1. Relación directa: A medida que el DPV aumenta (el aire se seca), la '
        'transpiración aumenta (pasa de 103 a 261 g/(m2h).\n'
        '2. Al haber menos humedad fuera de la hoja que dentro, el agua sale con más '
        'facilidad por difusión. Es como si el aire "succionara" el agua de la planta, '
        'y lo hará con más "fuerza" cuanto mayor sea el DPV.')
    pdf.ln(6)

    # ── B) Invernadero no climatizado ──
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(0, 9, 'B) ¿Que ocurriría en un invernadero no climatizado?', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(2)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5,
        'Con el escenario por defecto (apertura 50 %, LAI = 3), los valores del invernadero '
        'sin climatización son:')
    pdf.ln(3)

    # Tabla comparativa B)
    b_col_w  = [52, 20, 16, 20, 30, 34, 34]
    b_headers = ['', 'Tint (C)', 'HR (%)', 'DPV (kPa)', 'Transp.\n(g/(m2h)',
                 'Ventilación\n(g/(m2h)', 'Condensacion\n(g/(m2h)']
    b_rows = [
        ['Climatizado (T fija 30 C)', '30,0', '69,7', '1,29', '241,1', '434,87', '0'],
        ['No climatizado (T equilibrio)', '27,9', '-', '0,80', '-', '-', '-'],
    ]
    pdf.set_font('Helvetica', 'B', 7)
    pdf.set_fill_color(220, 230, 242)
    for w, h in zip(b_col_w, b_headers):
        pdf.multi_cell(w, 5, h, border=1, align='C', fill=True,
                       new_x='RIGHT', new_y='TOP', max_line_height=5)
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 8)
    for row in b_rows:
        for j, cell in enumerate(row):
            pdf.cell(b_col_w[j], 8, cell, border=1, align='C')
        pdf.ln()
    pdf.ln(4)

    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5,
        'La temperatura de equilibrio pasivo es 27,9 C y el DPV sería de 0,80 kPa. '
        'El cultivo se encuentra en muy buenas condiciones. El efecto invernadero es discreto '
        '(+2,9 C sobre la temperatura exterior) ya que el cultivo crecido (LAI = 3) está '
        'liberando mucha agua por transpiración y la ventilación es alta.')
    pdf.ln(3)
    pdf.set_fill_color(255, 243, 205)
    pdf.multi_cell(0, 5,
        'Prueba en el simulador: Cierra totalmente las ventanas (apertura = 0) y observa '
        'que ocurre en las condiciones de equilibrio pasivo.',
        border=1, fill=True)
    pdf.ln(3)
    pdf.multi_cell(0, 5,
        'La temperatura se dispara hasta los 41,3 C y el DPV a 3,69 kPa, con lo '
        'que el cultivo estaría sometido a un fuerte estres hídrico. Con las ventanas cerradas '
        'el invernadero tiene muy limitado el intercambio de energía con el exterior y el '
        'efecto invernadero se dispara hasta 16,3 C.')
    pdf.ln(3)
    pdf.multi_cell(0, 5,
        'Aumenta ahora el valor de radiación poco a poco y verás como sigue aumentando la '
        'temperatura hasta que el simulador deja de calcular, ya que las condiciones serían '
        'insostenibles para el cultivo.')

    return bytes(pdf.output())

def _generate_pdf_act2(nombre, df_a, df_b, df_c, df_d, resp_a, resp_d):
    """Genera el PDF de la Actividad 2 y lo devuelve como bytes."""
    from fpdf import FPDF

    # Variables para nombres de columna con caracteres Unicode especiales.
    # Definidas fuera de f-strings para compatibilidad con Python 3.11.
    _col_transp  = 'Transp. (g\u00b7m\u207b\u00b2\u00b7h\u207b\u00b9)'
    _col_humid   = 'Humid. (L\u00b7m\u207b\u00b2\u00b7h\u207b\u00b9)'
    _col_tint    = 'Tint (\u00b0C)'
    _col_tequil  = 'Tequil. (\u00b0C)'
    _col_rad     = 'Rad. ext. (W/m\u00b2)'

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Portada ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'ACTIVIDAD 2: El LAI y la Refrigeración Evaporativa',
             new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.set_font('Helvetica', '', 11)
#    pdf.cell(0, 7, 'Simulador HORTITRANS', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.ln(4)

    if nombre:
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(35, 8, 'Nombre:')
        pdf.set_font('Helvetica', '', 11)
        pdf.cell(0, 8, nombre, new_x='LMARGIN', new_y='NEXT')
        pdf.ln(3)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Escenario: Mediodía de verano', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.set_fill_color(240, 240, 240)
    pdf.multi_cell(
        0, 6,
        'Área suelo: 8000 m2  |  Ventanas: 1600 m2 (100% apertura)  |  Estanqueidad: Normal  |  LAI: 3\n'
        'T exterior: 31 C  |  HR exterior: 50%  |  Radiación: 900 W/m2  |  Viento: 2 m/s\n'
        'T interior (fija): 33 C  |  Cielo: Despejado  |  Humidificación: 0',
        border=1, fill=True,
    )
    pdf.ln(5)

    # ── A) Tabla LAI ──
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'A) Tabla: Efecto del LAI sobre el microclima', new_x='LMARGIN', new_y='NEXT')
    cw_a = [25, 45, 45, 55]
    hdr_a = ['LAI', 'Tequil. (C)', 'DPV equil. (kPa)', 'Transp. (g/(m2h))']
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(220, 230, 242)
    for w, h in zip(cw_a, hdr_a):
        pdf.cell(w, 8, h, border=1, align='C', fill=True)
    pdf.ln()
    pdf.set_font('Helvetica', '', 9)
    for _, row in df_a.iterrows():
        vals = [
            f"{row['LAI']:.1f}",
            f"{float(row['Tequil. (°C)']):.1f}" if pd.notna(row['Tequil. (°C)']) else '',
            f"{float(row['DPV equil. (kPa)']):.2f}" if pd.notna(row['DPV equil. (kPa)']) else '',
            f"{float(row[_col_transp]):.2f}" if pd.notna(row[_col_transp]) else '',
        ]
        for w, v in zip(cw_a, vals):
            pdf.cell(w, 8, v, border=1, align='C')
        pdf.ln()
    pdf.ln(4)

    mask_a = df_a[_ACT2_COLS_A].notna().any(axis=1)
    df_ap = df_a[mask_a].copy()
    if not df_ap.empty:
        x_a = df_ap['LAI'].values
        specs_a_pdf = [
            ('Tequil. (°C)', 'T equilibrio (C)', '#E74C3C'),
            ('DPV equil. (kPa)', 'DPV equilibrio (kPa)', '#F39C12'),
            (_col_transp, 'Transpiracion (g/m2/h)', '#27AE60'),
        ]
        bufs_a = [
            _make_chart_buf_gen(x_a, df_ap[dc].values, lbl, clr, 'LAI')
            for dc, lbl, clr in specs_a_pdf
        ]
        y_pos = pdf.get_y()
        for buf, cx in zip(bufs_a, [10, 75, 140]):
            pdf.image(buf, x=cx, y=y_pos, w=60)
        pdf.ln(55)

    # ── Cuestionario A ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Cuestionario A', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(4)
    for idx, (titulo, pregunta) in enumerate(_ACT2_PREGUNTAS_A, 1):
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 7, _pdf_str(f'{idx}. {titulo}'), new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        pdf.multi_cell(0, 5, _pdf_str(pregunta))
        pdf.ln(2)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 5, 'Respuesta:', new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        resp = resp_a[idx - 1]
        if resp.strip():
            pdf.multi_cell(0, 5, _pdf_str(resp), border=1)
        else:
            pdf.multi_cell(0, 18, '', border=1)
        pdf.ln(5)

    # ── Explicación A: Alternativas para reducir el DPV ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Explicación A: Alternativas para reducir el DPV', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(3)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 7, '1. Sombreado (Reducción de la Radiación Solar)', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5,
        'Si entra menos energía del sol, el aire y las hojas se calientan menos.\n'
        '- Encalado (Blanqueo): Se pinta el plastico o cristal con cal o productos blancos '
        'que reflejan la luz solar.\n'
        '- Mallas de sombreo: Redes de plástico (negras, aluminizadas, etc) sobre el cultivo o '
        'la estructura para filtrar la insolación.\n'
        '- Pantallas térmicas: Cortinas internas móviles que se cierran en las horas de '
        'máxima insolacion.'
    )
    pdf.ln(4)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 7, '2. Refrigeración Evaporativa (Uso del agua para enfriar)', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5,
        'Esta es la técnica más eficaz. Se basa en que el agua necesita calor para evaporarse: '
        'al forzar su evaporación dentro del invernadero, "roba" ese calor al aire, bajando '
        'la temperatura y subiendo la humedad (y bajando el DPV).\n'
        '- Nebulización (Misting/Fogging): Boquillas de alta presion crean gotas tan finas '
        'que se evaporan antes de tocar la hoja.\n'
        '- Panel evaporativo (Cooling Pad): Pared de material poroso empapado en agua; '
        'los ventiladores hacen pasar el aire exterior por él antes de que entre al cultivo.'
    )
    pdf.ln(6)

    # ── B) Tabla Tint ──
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'B) Tabla: Efecto de la temperatura interior sobre el DPV',
             new_x='LMARGIN', new_y='NEXT')
    cw_b = [50, 50]
    hdr_b = ['Tint (C)', 'DPV (kPa)']
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(220, 230, 242)
    for w, h in zip(cw_b, hdr_b):
        pdf.cell(w, 8, h, border=1, align='C', fill=True)
    pdf.ln()
    pdf.set_font('Helvetica', '', 9)
    for _, row in df_b.iterrows():
        vals = [
            f"{row[_col_tint]:.1f}",
            f"{float(row['DPV (kPa)']):.2f}" if pd.notna(row['DPV (kPa)']) else '',
        ]
        for w, v in zip(cw_b, vals):
            pdf.cell(w, 8, v, border=1, align='C')
        pdf.ln()
    pdf.ln(4)

    mask_b = df_b['DPV (kPa)'].notna()
    df_bp = df_b[mask_b].copy()
    if not df_bp.empty:
        buf_b = _make_chart_buf_gen(
            df_bp[_col_tint].values, df_bp['DPV (kPa)'].values,
            'DPV (kPa)', '#F39C12', 'Tint (C)',
        )
        pdf.image(buf_b, x=10, y=pdf.get_y(), w=90)
        pdf.ln(70)

    # ── C) Tabla Radiación ──
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'C) Tabla: Efecto del sombreado (radiación exterior)', new_x='LMARGIN', new_y='NEXT')
    cw_c = [46, 35, 38, 50]
    hdr_c = ['Rad. ext. (W/m2)', '% Sombreo', 'Tequil. (C)', 'DPV equil. (kPa)']
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(220, 230, 242)
    for w, h in zip(cw_c, hdr_c):
        pdf.cell(w, 8, h, border=1, align='C', fill=True)
    pdf.ln()
    pdf.set_font('Helvetica', '', 9)
    for _, row in df_c.iterrows():
        vals = [
            f"{int(row[_col_rad])}",
            f"{int(row['% Sombreo'])}",
            f"{float(row[_col_tequil]):.1f}" if pd.notna(row[_col_tequil]) else '',
            f"{float(row['DPV equil. (kPa)']):.2f}" if pd.notna(row['DPV equil. (kPa)']) else '',
        ]
        for w, v in zip(cw_c, vals):
            pdf.cell(w, 8, v, border=1, align='C')
        pdf.ln()
    pdf.ln(4)

    mask_c = df_c[_ACT2_COLS_C].notna().any(axis=1)
    df_cp = df_c[mask_c].copy()
    if not df_cp.empty:
        x_c = df_cp[_col_rad].values
        specs_c_pdf = [
            ('% Sombreo', '% Sombreo', '#8E44AD'),
            (_col_tequil, 'T equilibrio (C)', '#E74C3C'),
            ('DPV equil. (kPa)', 'DPV equilibrio (kPa)', '#F39C12'),
        ]
        bufs_c = [
            _make_chart_buf_gen(x_c, df_cp[dc].values, lbl, clr, 'Rad. ext. (W/m2)')
            for dc, lbl, clr in specs_c_pdf
        ]
        y_pos_c = pdf.get_y()
        for buf, cx in zip(bufs_c, [10, 75, 140]):
            pdf.image(buf, x=cx, y=y_pos_c, w=60)
        pdf.ln(55)

    # ── D) Tabla Humidificación ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'D) Tabla: Refrigeración evaporativa (humidificación)', new_x='LMARGIN', new_y='NEXT')
    cw_d = [48, 50, 30, 40]
    hdr_d = ['Humid. (L/(m2h))', 'Transp. (g/(m2h))', 'HR (%)', 'DPV (kPa)']
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(220, 230, 242)
    for w, h in zip(cw_d, hdr_d):
        pdf.cell(w, 8, h, border=1, align='C', fill=True)
    pdf.ln()
    pdf.set_font('Helvetica', '', 9)
    for _, row in df_d.iterrows():
        vals = [
            f"{row[_col_humid]:.1f}",
            f"{float(row[_col_transp]):.2f}" if pd.notna(row[_col_transp]) else '',
            f"{float(row['HR (%)']):.1f}" if pd.notna(row['HR (%)']) else '',
            f"{float(row['DPV (kPa)']):.2f}" if pd.notna(row['DPV (kPa)']) else '',
        ]
        for w, v in zip(cw_d, vals):
            pdf.cell(w, 8, v, border=1, align='C')
        pdf.ln()
    pdf.ln(4)

    mask_d = df_d[_ACT2_COLS_D].notna().any(axis=1)
    df_dp = df_d[mask_d].copy()
    if not df_dp.empty:
        x_d = df_dp[_col_humid].values
        specs_d_pdf = [
            (_col_transp, 'Transpiracion (g/m2/h)', '#27AE60'),
            ('HR (%)', 'HR (%)', '#1f77b4'),
            ('DPV (kPa)', 'DPV (kPa)', '#d62728'),
        ]
        bufs_d = [
            _make_chart_buf_gen(x_d, df_dp[dc].values, lbl, clr, 'Humidificacion (L/m2/h)')
            for dc, lbl, clr in specs_d_pdf
        ]
        y_pos_d = pdf.get_y()
        for buf, cx in zip(bufs_d, [10, 75, 140]):
            pdf.image(buf, x=cx, y=y_pos_d, w=60)
        pdf.ln(55)

    # ── Cuestionario D ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Cuestionario D', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(4)
    for idx, (titulo, pregunta) in enumerate(_ACT2_PREGUNTAS_D, 1):
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 7, _pdf_str(f'{idx}. {titulo}'), new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        pdf.multi_cell(0, 5, _pdf_str(pregunta))
        pdf.ln(2)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 5, 'Respuesta:', new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        resp = resp_d[idx - 1]
        if resp.strip():
            pdf.multi_cell(0, 5, _pdf_str(resp), border=1)
        else:
            pdf.multi_cell(0, 18, '', border=1)
        pdf.ln(5)

    # ── Explicación ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(0, 9, 'Explicación', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(3)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 7, 'El DPV en verano: el reto de la refrigeración', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5,
        'En un día de verano con alta radiación, el invernadero acumula mucho calor. '
        'Incluso con ventilación máxima, la temperatura interior se mantiene varios grados '
        'por encima del exterior. Cuanto mayor es la temperatura, mayor es la presión de '
        'vapor de saturación, por lo que el aire puede absorber mucho más vapor: el DPV '
        'se dispara.')
    pdf.ln(4)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 7, 'El efecto del LAI sobre la temperatura de equilibrio', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5,
        'La transpiración de las plantas actua como un sistema de refrigeracion natural. '
        'Al evaporar agua, las plantas roban calor latente al aire, enfriandolo. Cuanto '
        'mayor es el LAI (mas hojas), más agua se evapora y más se enfria el invernadero. '
        'Sin embargo, este enfriamiento es limitado: incluso con LAI = 3, el DPV sigue '
        'siendo muy elevado en condiciones de verano extremo.')
    pdf.ln(4)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 7, '¿Por qué la transpiracion total sube aunque el DPV baje cuando se incrementa el LAI?', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5,
        'Hay dos efectos contrapuestos:\n'
        '- El DPV baja (el aire esta más humedo, menos fuerza de succión por unidad de hoja).\n'
        '- El LAI sube (hay muchas mas hojas transpirando).\n\n'
        'El segundo efecto supera con creces al primero: el área foliar total crece 30 veces '
        '(de 0,1 a 3), mientras que el DPV solo baja un 32 %. El resultado es que la '
        'transpiración total del sistema aumenta enormemente.')
    pdf.ln(4)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 7, 'Sombreado vs. Refrigeración Evaporativa', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5,
        'El sombreado reduce la energía que entra, pero con un coste muy alto: reduce también '
        'la fotosíntesis y la producción. Para reducir el DPV por debajo de 1,6 kPa mediante '
        'sombreado solo, habría que eliminar mas del 85 % de la radiación, lo que es '
        'agronómicamente inviable.\n\n'
        'La refrigeración evaporativa (nebulización), en cambio, ataca directamente el '
        'problema: al añadir agua  al aire, se consume mayor cantidad de energía en evaporarla y esta energía no está '
        'disponible para calentar el aire, lo que enfría el aire y aumenta su humedad simultáneamente, '
        'reduciendo el DPV de forma eficaz con un impacto mínimo sobre la radiación '
        'disponible para el cultivo.')

    return bytes(pdf.output())


def _generate_pdf_act3(nombre, df_a, df_b, resp_a, resp_b):
    """Genera el PDF de la Actividad 3 y lo devuelve como bytes."""
    from fpdf import FPDF

    # Variables para nombres de columna con caracteres Unicode especiales (Python 3.11 compat.)
    _col_transp = _ACT3_COL_TRANSP
    _col_vent   = _ACT3_COL_VENT
    _col_tint   = _ACT3_COL_TINT
    _col_tequil = _ACT3_COL_TEQUIL
    _col_text   = _ACT3_COL_TEXT

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Portada ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 12, 'ACTIVIDAD 3: Comportamiento Nocturno', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.ln(4)
    if nombre.strip():
        pdf.set_font('Helvetica', '', 11)
        pdf.cell(0, 8, f'Estudiante: {nombre}', new_x='LMARGIN', new_y='NEXT', align='C')
        pdf.ln(2)

    # ── Contexto ──
    pdf.set_font('Helvetica', 'B', 12)
#    pdf.cell(0, 8, 'Contexto teorico', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5,
        'Durante la noche, el invernadero deja de ganar energía solar y empieza a perderla hacia '
        'el exterior. El comportamiento de las variables cambia radicalmente respecto al día:\n'
        '- Temperatura: cae de forma continua al no haber fuente de calor.\n'
        '- DPV: se desploma hasta valores cercanos a 0. Al enfriarse el aire se satura.\n'
        '- Transpiración: es mínima o nula. Los estomas suelen estar cerrados.\n\n'
        'El tipo de cielo es determinante: con cielo despejado el calor escapa directamente '
        'a la atmósfera (puede darse inversión térmica); con cielo nublado las nubes actuan '
        'como una manta que retiene el calor.')
    pdf.ln(6)

    # ── Parametros de entrada ──
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Parámetros del escenario (comunes a A y B)', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    params_rows = [
        ('Área suelo', '8 000 m2'), ('Área ventanas', '1 600 m2'),
        ('Estanqueidad', 'Normal'), ('LAI', '3'),
        ('T exterior', '11 oC'), ('HR exterior', '75 %'),
        ('Radiación solar', '0 W/m2'), ('Viento', '1 m/s'),
        ('T interior', '15 oC'), ('Apertura ventanas', '0 %'),
    ]
    cw_p = [80, 80]
    border = {'style': 'SINGLE'}
    for label, val in params_rows:
        pdf.cell(cw_p[0], 7, label, border=1)
        pdf.cell(cw_p[1], 7, val, border=1, new_x='LMARGIN', new_y='NEXT')
    pdf.ln(6)

    def _render_scenario_table(df, title):
        if df is None or df.empty:
            return
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 8, title, new_x='LMARGIN', new_y='NEXT')
        cols_show = ['HR (%)', 'DPV (kPa)', _col_transp, _col_vent,
                     _col_text, _col_tint, _col_tequil, 'DPVequil. (kPa)']
        hdrs_show = ['HR (%)', 'DPV (kPa)', 'Transp.(g/m2/h)', 'Vent.(g/m2/h)',
                     'Text (oC)', 'Tint (oC)', 'Tequil.(oC)', 'DPVeq.(kPa)']
        cw = [24, 22, 30, 26, 20, 20, 24, 28]
        pdf.set_font('Helvetica', 'B', 7)
        pdf.set_fill_color(220, 230, 242)
        for w, h in zip(cw, hdrs_show):
            pdf.cell(w, 7, h, border=1, align='C', fill=True)
        pdf.ln()
        pdf.set_font('Helvetica', '', 8)
        row = df.iloc[0]
        vals = []
        for col in cols_show:
            v = row.get(col)
            if pd.isna(v):
                vals.append('')
            elif col in ('HR (%)', _col_tint, _col_tequil, _col_text):
                vals.append(f"{float(v):.1f}")
            else:
                vals.append(f"{float(v):.2f}")
        for w, v in zip(cw, vals):
            pdf.cell(w, 7, v, border=1, align='C')
        pdf.ln()
        pdf.ln(4)

    # ── Escenario A ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'A) Cielo despejado - ventanas cerradas', new_x='LMARGIN', new_y='NEXT')
    _render_scenario_table(df_a, 'Tabla resumen escenario A')

    # ── Escenario B ──
    _render_scenario_table(df_b, 'B) Cielo nublado - ventanas cerradas')

    # ── Gráfico comparación ──
    row_a = df_a.iloc[0] if (df_a is not None and not df_a.empty) else None
    row_b = df_b.iloc[0] if (df_b is not None and not df_b.empty) else None
    any_a = row_a is not None and not all(pd.isna(row_a[c]) for c in ['HR (%)', 'DPV (kPa)', _col_tequil])
    any_b = row_b is not None and not all(pd.isna(row_b[c]) for c in ['HR (%)', 'DPV (kPa)', _col_tequil])

    if any_a or any_b:
        labels = ['Cielo despejado', 'Cielo nublado']
        def _get(row, col):
            if row is None or pd.isna(row.get(col)):
                return float('nan')
            return float(row[col])

        specs_cmp = [
            ('HR (%)', 'HR (%)', '#1f77b4'),
            ('DPV (kPa)', 'DPV (kPa)', '#d62728'),
            (_col_tequil, 'Tequil. (oC)', '#E74C3C'),
            (_col_vent, 'Vent. (g/(m2h))', '#ff7f0e'),
        ]
        bufs_cmp = []
        for col, lbl, clr in specs_cmp:
            vals_chart = [_get(row_a, col), _get(row_b, col)]
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            colors = [clr, '#95a5a6']
            ax.bar(labels, vals_chart, color=colors, edgecolor='grey', width=0.5)
            ax.set_title(lbl, fontsize=9, fontweight='bold')
            ax.set_ylabel(lbl, fontsize=7)
            ax.tick_params(axis='x', labelsize=7)
            ax.grid(True, axis='y', alpha=0.3)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            bufs_cmp.append(buf)

        pdf.ln(4)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 8, 'Graficos comparativos (A vs B)', new_x='LMARGIN', new_y='NEXT')
        y_pos = pdf.get_y()
        for i, buf in enumerate(bufs_cmp):
            cx = 10 + i * 48
            pdf.image(buf, x=cx, y=y_pos, w=46)
        pdf.ln(55)

    # ── Cuestionario A ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Cuestionario A - Cielo despejado', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(3)
    for idx, (titulo, pregunta) in enumerate(_ACT3_PREGUNTAS_A, 1):
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 7, _pdf_str(f'{idx}. {titulo}'), new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        pdf.multi_cell(0, 5, _pdf_str(pregunta))
        pdf.ln(2)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 5, 'Respuesta:', new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        resp = resp_a[idx - 1]
        if resp.strip():
            pdf.multi_cell(0, 5, _pdf_str(resp), border=1)
        else:
            pdf.multi_cell(0, 18, '', border=1)
        pdf.ln(5)

    # ── Cuestionario B ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Cuestionario B - Cielo nublado', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(3)
    for idx, (titulo, pregunta) in enumerate(_ACT3_PREGUNTAS_B, 1):
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 7, _pdf_str(f'{idx}. {titulo}'), new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        pdf.multi_cell(0, 5, _pdf_str(pregunta))
        pdf.ln(2)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 5, 'Respuesta:', new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        resp = resp_b[idx - 1]
        if resp.strip():
            pdf.multi_cell(0, 5, _pdf_str(resp), border=1)
        else:
            pdf.multi_cell(0, 18, '', border=1)
        pdf.ln(5)

    # ── Explicación ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(0, 9, 'Explicación', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(3)

    bloques = [
        ('Comportamiento nocturno del invernadero',
         'Durante la noche, sin radiación solar, el invernadero pierde calor hacia el exterior. '
         'La temperatura interior cae contínuamente. El aire se enfría, pierde capacidad de contener '
         'vapor de agua y se satura: la Humedad Relativa sube al 100 % y el DPV cae a 0. '
         'Los estomas de las plantas estan cerrados, por lo que la transpiración es nula o mínima.'),
        ('Inversión térmica con cielo despejado',
         'Con cielo despejado, el invernadero emite radiación de onda larga directamente hacia '
         'la atmósfera sin impedimentos. Esto produce un enfriamiento radiativo intenso: la temperatura '
         'de equilibrio puede caer por debajo de la temperatura exterior, fenomeno conocido como '
         'inversión térmica. En plásticos no térmicos este efecto es más pronunciado.'),
        ('El efecto del cielo nublado',
         'Las nubes actúan como una manta: absorben la radiación de onda larga emitida por el '
         'invernadero y la reemiten de vuelta. Esto reduce el enfriamiento radiativo y eleva la '
         'temperatura de equilibrio por encima de la temperatura exterior. En estas condiciones '
         'no se produce inversión térmica.'),
        ('Consecuencias prácticas para el manejo',
         'Con inversión térmica (cielo despejado), abrir las ventanas introduce aire exterior '
         'mas caliente '
         'Con cielo nublado y sin inversión térmica, una ventilación moderada puede renovar el '
         'aire saturado y reducir el riesgo de enfermedades fungicas, aproximando la temperatura '
         'interior a la de equilibrio, que es superior a la exterior.'),
    ]
    for titulo, texto in bloques:
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 7, titulo, new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 9)
        pdf.multi_cell(0, 5, texto)
        pdf.ln(4)

    return bytes(pdf.output())


def _generate_pdf_act4(nombre, respuestas):
    """Genera el PDF del test de la Actividad 4 y lo devuelve como bytes."""
    from fpdf import FPDF

    n_total  = len(_ACT4_PREGUNTAS)
    n_ok     = sum(
        1 for i, q in enumerate(_ACT4_PREGUNTAS)
        if respuestas.get(i) == q['correcta']
    )

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Cabecera ──
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 12, 'ACTIVIDAD 4: Test de Conocimiento', new_x='LMARGIN', new_y='NEXT', align='C')
    if nombre.strip():
        pdf.set_font('Helvetica', '', 11)
        pdf.cell(0, 8, f'Estudiante: {nombre}', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.ln(3)

    # ── Puntuación ──
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_fill_color(220, 230, 242)
    pct = n_ok / n_total * 100
    pdf.cell(0, 10, f'Resultado: {n_ok} / {n_total}  ({pct:.0f} %)',
             new_x='LMARGIN', new_y='NEXT', align='C', fill=True)
    pdf.ln(6)

    # ── Preguntas ──
    letras = ['a', 'b', 'c', 'd']
    for i, q in enumerate(_ACT4_PREGUNTAS):
        resp_idx = respuestas.get(i)
        es_ok    = resp_idx == q['correcta']

        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_x(pdf.l_margin)
        num_txt = _pdf_str(f'{i+1}. {q["pregunta"]}')
        pdf.multi_cell(0, 6, num_txt, new_x='LMARGIN', new_y='NEXT')
        pdf.ln(1)

        pdf.set_font('Helvetica', '', 9)
        for j, op in enumerate(q['opciones']):
            marca = ''
            if j == resp_idx:
                marca = '[TU RESPUESTA]  '
            if j == q['correcta']:
                marca += '[CORRECTA]'
            linea = _pdf_str(f'  {letras[j]}) {op}  {marca}')
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, linea, new_x='LMARGIN', new_y='NEXT')
        pdf.ln(2)

        # ── Veredicto ──
        if resp_idx is None:
            pdf.set_font('Helvetica', 'I', 8)
            pdf.cell(0, 5, 'Sin respuesta.', new_x='LMARGIN', new_y='NEXT')
        elif es_ok:
            pdf.set_font('Helvetica', 'B', 9)
            pdf.cell(0, 5, 'CORRECTO', new_x='LMARGIN', new_y='NEXT')
        else:
            pdf.set_font('Helvetica', 'B', 9)
            pdf.cell(0, 5, f'INCORRECTO  (correcta: {letras[q["correcta"]]})',
                     new_x='LMARGIN', new_y='NEXT')

        # ── Explicación ──
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 4, _pdf_str(f'Explicación: {q["explicacion"]}'), new_x='LMARGIN', new_y='NEXT')
        pdf.ln(5)

        if pdf.get_y() > 250 and i < n_total - 1:
            pdf.add_page()

    return bytes(pdf.output())


# --- Session State: valores por defecto ---
_defaults = {
    'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Normal',
    'ss_LAI': 3.0, 'ss_T_o': 25.0, 'ss_RH_o': 80, 'ss_R_sol': 400, 'ss_viento': 2.0,
    'ss_T_i': 30.0, 'ss_apertura': 50, 'ss_humidif': 0.0,
    'ss_cielo': 'Cielo despejado',
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

_ESCENARIOS = {
    "— Selecciona un escenario —": None,
    "📋 Actividad 1 — Efecto de la ventilación": {
        'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Normal',
        'ss_LAI': 3.0, 'ss_T_o': 25.0, 'ss_RH_o': 80, 'ss_R_sol': 400, 'ss_viento': 2.0,
        'ss_T_i': 30.0, 'ss_apertura': 50, 'ss_humidif': 0.0, 'ss_cielo': 'Cielo despejado',
    },
    "📋 Actividad 2 — LAI y Refrigeración Evaporativa": {
        'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Normal',
        'ss_LAI': 3.0, 'ss_T_o': 31.0, 'ss_RH_o': 50, 'ss_R_sol': 900, 'ss_viento': 2.0,
        'ss_T_i': 33.0, 'ss_apertura': 100, 'ss_humidif': 0.0, 'ss_cielo': 'Cielo despejado',
    },
    "📋 Actividad 3 — Comportamiento Nocturno": {
        'ss_area_suelo': 8000, 'ss_area_ventanas': 1600, 'ss_estanqueidad': 'Normal',
        'ss_LAI': 3.0, 'ss_T_o': 11.0, 'ss_RH_o': 75, 'ss_R_sol': 0, 'ss_viento': 1.0,
        'ss_T_i': 15.0, 'ss_apertura': 0, 'ss_humidif': 0.0, 'ss_cielo': 'Cielo despejado',
    },
    "☀️ Mediodía de verano":  {'ss_R_sol': 900,  'ss_T_o': 31.0, 'ss_RH_o': 50, 'ss_T_i': 33.0,
                               'ss_viento': 2.0, 'ss_apertura': 100, 'ss_LAI': 3.0},
    "❄️ Mediodía de invierno": {'ss_R_sol': 500,  'ss_T_o': 17.0, 'ss_RH_o': 55, 'ss_T_i': 21.0,
                                'ss_viento': 2.0, 'ss_apertura': 30, 'ss_LAI': 3.0},
    "🌙 Noche de invierno":    {'ss_R_sol': 0,    'ss_T_o': 11.0, 'ss_RH_o': 75, 'ss_T_i': 15.0,
                                'ss_viento': 1.0, 'ss_apertura': 0,  'ss_LAI': 3.0},
    "🌃 Noche de verano":      {'ss_R_sol': 0,    'ss_T_o': 24.0, 'ss_RH_o': 70, 'ss_T_i': 25.0,
                                'ss_viento': 1.5, 'ss_apertura': 10, 'ss_LAI': 3.0},
}

with st.sidebar:
    st.header("Parámetros de Entrada")
    st.markdown("Procura que los valores sean coherentes")
    escenario_sel = st.selectbox(
        "📋 Cargar un escenario predefinido",
        options=list(_ESCENARIOS.keys()),
        index=0,
        key="ss_escenario_sel",
    )
    if escenario_sel != "— Selecciona un escenario —":
        if st.button("▶ Aplicar escenario", key="btn_aplicar_escenario"):
            load_scenario(_ESCENARIOS[escenario_sel], escenario_sel)
    st.markdown("---")

    with st.expander("1. Parámetros del Invernadero", expanded=True):
        p_A_suelo = st.number_input(
            "Área del suelo (m²)", 10, 10000, step=10, key="ss_area_suelo"
        )
        p_A_vent_total = st.number_input(
            "Área total de ventanas (m²)", 0, 2000, step=5, key="ss_area_ventanas",
            help="laterales + cenitales."
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
        p_w = st.slider("Velocidad del viento (m/s)", 0.0, 5.0, step=0.5, key="ss_viento")
        _CIELO_OPTS = ['Cielo nublado', 'Cielo despejado']
        p_cielo = st.selectbox(
            "Estado del cielo", _CIELO_OPTS,
            index=_CIELO_OPTS.index(st.session_state['ss_cielo']),
            key="ss_cielo"
#            help="Cielo nublado: ΔT_cielo = 10 °C · Cielo despejado: ΔT_cielo = 20 °C"
        )
        p_delta_T_sky = 10 if p_cielo == 'Cielo nublado' else 20

    with st.expander("4. Consignas de Control Interno", expanded=True):
        p_T_i = st.slider("Temperatura interior (°C)", 0.0, 45.0, step=0.5, key="ss_T_i")
        p_angulo_vent = st.slider("Apertura de ventanas (%)", 0, 100, step=1, key="ss_apertura")
        p_L_m2_h = st.number_input(
            "Humid./Deshumid.(Lm⁻²h⁻¹)",
            min_value=-2.0, max_value=2.5, step=0.05, key="ss_humidif",
            help="\\+ humidifica; - deshumidifica."
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

# ===== TABS =====
tab_sim, tab_act1, tab_act2, tab_act3, tab_act4 = st.tabs([
    "🌡️ Simulador", "📋 Actividad 1", "📋 Actividad 2", "📋 Actividad 3", "📝 Test (Act. 4)"
])

# ───────────────────────── TAB 1: SIMULADOR ─────────────────────────
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

        col_flujo1, col_flujo2 = st.columns(2)
        with col_flujo1:
            st.metric(label="Ventilación (g m⁻² h⁻¹)", value=f"{E_v_gmh:.2f}", help="Vapor de agua que sale por la ventilación.")
        with col_flujo2:
            st.metric(label="Condensación (g m⁻² h⁻¹)", value=f"{E_c_gmh:.2f}", help="Vapor de agua que se condensa en la cubierta.")

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

        st.subheader("🌡️ Condiciones de equilibrio pasivo")
        st.markdown(
            "Condiciones interiores si el invernadero operara **sin calefacción ni refrigeración activa**."
            
        )
        if eq['T_eq'] - p_T_o > 20:
            st.warning("Comprueba los datos introducidos o el grado de apertura de las ventanas. Los valores no son coherentes para los invernaderos mediterráneos y/o provocarían la muerte del cultivo.")
        else:
            HR_sim   = results.get('RH_i', 0)
            e_si_con = 0.61078 * np.exp(17.27 * p_T_i / (p_T_i + 237.3))
            e_i_abs  = HR_sim / 100.0 * e_si_con
            T_eq_val = eq['T_eq']
            e_si_eq  = 0.61078 * np.exp(17.27 * T_eq_val / (T_eq_val + 237.3))
            DPV_eq   = max(0.0, e_si_eq - e_i_abs)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("T equilibrio pasivo",
                          f"{T_eq_val:.1f} °C",
                          delta=f"{dT:+.1f} °C vs consigna",
                          delta_color="off")
            with c2:
                st.metric("DPV en equilibrio",
                          f"{DPV_eq:.2f} kPa",
                          help=f"DPV calculado con HR del escenario actual ({HR_sim:.1f} %) a la T de equilibrio.")

            if e_i_abs > e_si_eq:
                st.warning("⚠️ Riesgo de condensación en cubierta en condiciones de equilibrio pasivo.")

            conv_txt = (f"✓ {eq['iters']} iteraciones"
                        if eq['converged']
                        else f"⚠️ No convergió ({eq['iters']} iteraciones)")
            st.caption(conv_txt)


# ───────────────────────── TAB 2: ACTIVIDAD 1 ─────────────────────────
with tab_act1:
    st.header("ACTIVIDAD 1: Efecto de la ventilación")

    # ── Escenario ──
    st.subheader("Escenario por defecto")
    st.info(
        "**Invernadero climatizado** — Carga este escenario en el simulador "
        "(menú lateral › *📋 Actividad 1*) y luego cambia únicamente la **Apertura de ventanas**."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
| Parámetro | Valor |
|---|---|
| Área del suelo | 8 000 m² |
| Área de ventanas | 1 600 m² |
| Estanqueidad | Normal |
| LAI (cultivo) | 3 |
""")
    with col_b:
        st.markdown("""
| Parámetro | Valor |
|---|---|
| T exterior | 25 °C |
| HR exterior | 80 % |
| Radiación solar | 400 W/m² |
| Viento | 2 m/s |
| Cielo | Despejado |
| **T interior (fija)** | **30 °C** |
""")

    st.markdown("**Valores de referencia del escenario climatizado (apertura 50 %):**")
    df_ref = pd.DataFrame({
        '': ['Climatizado'],
        'HR (%)': ['69,7'],
        'DPV (kPa)': ['1,29'],
        'Transp. (g·m⁻²·h⁻¹)': ['241,1'],
        'Ventilación (g·m⁻²·h⁻¹)': ['434,87'],
        'Condensación (g·m⁻²·h⁻¹)': ['0'],
    })        
    st.dataframe(df_ref, hide_index=True, use_container_width=True)    
        
        

    st.divider()

    # ── Instrucciones ──
    st.subheader("A) Rellena la tabla variando la apertura de ventanas")
    st.markdown("""
1. En el **Simulador** (pestaña izquierda), asegúrate de tener cargado el escenario de la Actividad 1.
2. Cambia el slider **Apertura de ventanas (%)** a cada valor de la tabla (0, 20, 40, 60, 80, 100).
3. Anota los valores que aparecen en el simulador en las columnas de la tabla.
4. Cuando hayas completado todas las filas, pulsa **Generar gráficos**.

> **Nota:** la columna *Tint* está fijada a 30 °C en todas las simulaciones.
""")

    # ── Tabla editable ──
    df_edited = st.data_editor(
        st.session_state['act1_tabla'],
        column_config={
            'Apertura (%)': st.column_config.NumberColumn(
                'Apertura (%)', disabled=True, format="%d"
            ),
            'Tint (°C)': st.column_config.NumberColumn(
                'Tint (°C)', disabled=True, format="%.0f"
            ),
            'HR (%)': st.column_config.NumberColumn(
                'HR (%)', min_value=0.0, max_value=100.0, format="%.1f",
                help="Humedad relativa interior (%)"
            ),
            'DPV (kPa)': st.column_config.NumberColumn(
                'DPV (kPa)', min_value=0.0, format="%.2f",
                help="Déficit de Presión de Vapor (kPa)"
            ),
            'Transp. (g·m⁻²·h⁻¹)': st.column_config.NumberColumn(
                'Transp. (g·m⁻²·h⁻¹)', min_value=0.0, format="%.2f",
                help="Transpiración del cultivo"
            ),
            'Vent. (g·m⁻²·h⁻¹)': st.column_config.NumberColumn(
                'Vent. (g·m⁻²·h⁻¹)', min_value=0.0, format="%.2f",
                help="Agua que sale por ventilación"
            ),
            'Cond. (g·m⁻²·h⁻¹)': st.column_config.NumberColumn(
                'Cond. (g·m⁻²·h⁻¹)', min_value=0.0, format="%.2f",
                help="Condensación en la cubierta"
            ),
        },
        hide_index=True,
        use_container_width=True,
        key='act1_editor',
    )

    generar = st.button("📊 Generar gráficos", type="primary", use_container_width=True)

    if generar:
        # Check at least one row has data
        has_data = df_edited[_ACT1_COLS].notna().any().any()
        if not has_data:
            st.warning("Rellena al menos una fila de la tabla antes de generar los gráficos.")
        else:
            st.session_state['act1_tabla'] = df_edited
            st.session_state['act1_show_charts'] = True

    # ── Gráficos ──
    if st.session_state['act1_show_charts']:
        df_chart = st.session_state['act1_tabla']
        # Only rows with at least one filled value
        mask = df_chart[_ACT1_COLS].notna().any(axis=1)
        df_plot = df_chart[mask].copy()
        x = df_plot['Apertura (%)']

        st.divider()
        st.subheader("Gráficos de evolución")

        _chart_specs = [
            ('HR (%)',               'Humedad Relativa (%)',             '#1f77b4'),
            ('DPV (kPa)',            'DPV (kPa)',                        '#d62728'),
            ('Transp. (g·m⁻²·h⁻¹)', 'Transpiración (g·m⁻²·h⁻¹)',       '#2ca02c'),
            ('Vent. (g·m⁻²·h⁻¹)',   'Ventilación (g·m⁻²·h⁻¹)',         '#ff7f0e'),
        ]

        col_g1, col_g2 = st.columns(2)
        for i, (col_key, y_label, color) in enumerate(_chart_specs):
            y = df_plot[col_key]
            valid = y.notna()
            fig = go.Figure()
            if valid.any():
                fig.add_trace(go.Scatter(
                    x=x[valid], y=y[valid],
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    name=y_label,
                ))
            fig.update_layout(
                title=dict(text=y_label, font=dict(size=14)),
                xaxis=dict(title='Apertura de ventanas (%)', tickmode='array',
                           tickvals=_ACT1_APERTURAS),
                yaxis=dict(title=y_label),
                height=320,
                margin=dict(l=10, r=10, t=45, b=40),
                showlegend=False,
            )
            target_col = col_g1 if i % 2 == 0 else col_g2
            with target_col:
                st.plotly_chart(fig, use_container_width=True, key=f"act1_chart_{i}")

    st.divider()

    # ── Nombre del estudiante ──
    st.subheader("Datos del estudiante")
    st.text_input(
        "Nombre y apellidos",
        placeholder="Escribe tu nombre aquí…",
        key='act1_nombre',
    )

    st.divider()

    # ── Cuestionario ──
    st.subheader("Cuestionario")
    st.caption("Responde a cada pregunta en el cuadro correspondiente.")

    _Q_BOLD = [
        ("El intercambio de masas",
         "Observa la columna de **HR**. ¿Por qué disminuye la humedad relativa al abrir "
         "las ventanas si la temperatura interior (**Tint**) se mantiene exactamente igual "
         "en 30 °C? Justifica qué está ocurriendo."),
        ("El punto de saturación",
         "En la apertura 0, el valor de **Cond.** es de 64,47, pero en la apertura 20 cae a 0. "
         "Explica la relación que existe entre la Humedad Relativa (HR) y la aparición de agua "
         "líquida (condensación) en las paredes del invernadero."),
        ('La "fuerza" del aire',
         "El **DPV** aumenta conforme abrimos las ventanas. Explica con tus palabras qué le "
         'está "haciendo" el aire a la planta cuando el DPV sube de 0,03 a 1,47 y cómo se '
         "refleja eso en la columna de **Transp.**"),
        ("Balance hídrico",
         "Si comparas la columna de **Transp.** (lo que la planta expulsa) con la de **Vent.** "
         "(lo que el aire saca al exterior), verás que a partir del 20 % de apertura, la "
         "ventilación es mucho más alta. ¿De dónde sale ese exceso de agua que la ventilación "
         "está moviendo si la planta transpira mucho menos de lo que se ventila?"),
        ("Control de variables",
         "Para estas simulaciones se ha decidido dejar la **Tint** fija en 30 °C. ¿Por qué "
         "crees que es necesario mantener la temperatura constante para entender el efecto "
         "real de la apertura de ventanas sobre la humedad y el DPV?"),
    ]

    for idx, (titulo, enunciado) in enumerate(_Q_BOLD, 1):
        st.markdown(f"**{idx}. {titulo}**")
        st.markdown(enunciado)
        st.text_area(
            label=f"Tu respuesta a la pregunta {idx}",
            placeholder="Escribe tu respuesta aquí…",
            height=120,
            key=f'act1_resp_{idx}',
            label_visibility='collapsed',
        )
        st.markdown("")  # spacing

    st.divider()

    # ── Explicación (desplegable) ──
    with st.expander("💡 Explicación"):

        st.markdown("### El DPV: El concepto clave")
        st.markdown(
            "El **DPV (Déficit de Presión de Vapor)** indica la diferencia entre la cantidad "
            "de humedad que el aire tiene y la que *podría* tener si estuviera saturado."
        )
        st.markdown(
            "- **DPV bajo (0,03 kPa):** El aire está \"lleno\". No \"tira\" del agua de la planta.\n"
            "- **DPV alto (1,47 kPa):** El aire no está \"lleno\". Cuanto más abrimos las ventanas, "
            "más sube el DPV, lo que significa que el ambiente tiene más capacidad de absorber agua."
        )

        st.markdown(
            "Cuando las ventanas están cerradas (**Apertura 0**), el aire está casi saturado "
            "(99,2 % de humedad relativa y DPV casi cero). Como el aire no puede contener más "
            "vapor de agua, este se convierte en líquido sobre las paredes o plantas: eso es la "
            "**condensación** (devuelve 64,47 g de agua por cada metro cuadrado del invernadero "
            "cada hora)."
        )

        st.markdown(
            "**¿Qué ocurre al abrir al 20 %?**  \n"
            "Por un lado entra aire del exterior que está más seco que el aire interior "
            "(en el exterior la temperatura es de 25 °C y la HR es del 80 %, por lo que su DPV "
            "es de 0,6 kPa); pero además sale aire húmedo (el invernadero pierde 325 g de agua "
            "por cada metro cuadrado de suelo y hora). Al bajar la humedad, el fenómeno de la "
            "condensación desaparece por completo (pasa a 0). Esto es vital para evitar hongos "
            "en los cultivos."
        )

        st.markdown(
            "Conforme se va aumentando el porcentaje de apertura de ventanas sigue disminuyendo "
            "la humedad relativa y aumentando el DPV, la transpiración y el agua perdida por "
            "ventilación, pero cada vez los decrementos e incrementos son menores "
            "**(la respuesta no es lineal)**. Se puede llegar incluso a tener dentro del "
            "invernadero menor humedad y mayor DPV que en el exterior, porque la temperatura "
            "interior está fijada a 30 °C."
        )

        st.divider()

        st.markdown("### La Transpiración del cultivo (Transp.)")
        st.markdown(
            "La transpiración es el proceso por el cual las plantas pierden agua en forma de "
            "vapor por sus estomas."
        )
        st.markdown(
            "1. **Relación directa:** A medida que el DPV aumenta (el aire se seca), la "
            "**transpiración aumenta** (pasa de 103 a 261 g·m⁻²·h⁻¹).\n"
            "2. Al haber menos humedad fuera de la hoja que dentro, el agua sale con más "
            "facilidad por difusión. Es como si el aire \"succionara\" el agua de la planta, "
            "y lo hará con más \"fuerza\" cuanto mayor sea el DPV."
        )

    st.divider()

    # ── B) Invernadero no climatizado ──
    st.subheader("B) ¿Qué ocurriría en un invernadero no climatizado?")
    st.markdown(
        "Con el escenario por defecto (apertura 50 %, LAI = 3), los valores del invernadero "
        "sin climatización son:"
    )

    df_b = pd.DataFrame({
        "": ["Climatizado (T fija 30 °C)", "No climatizado (T equilibrio)"],
        "Tint (°C)": ["30,0", "27,9"],
        "HR (%)": ["69,7", "—"],
        "DPV (kPa)": ["1,29", "0,80"],
        "Transp. (g·m⁻²·h⁻¹)": ["241,1", "—"],
        "Ventilación (g·m⁻²·h⁻¹)": ["434,87", "—"],
        "Condensación (g·m⁻²·h⁻¹)": ["0", "—"],
    })
    st.dataframe(df_b, hide_index=True, use_container_width=True)

    st.markdown(
        "La temperatura de equilibrio pasivo es **27,9 °C** y el DPV sería de **0,80 kPa**. "
        "El cultivo se encuentra en muy buenas condiciones. El efecto invernadero es discreto "
        "(+2,9 °C sobre la temperatura exterior) ya que el cultivo crecido (LAI = 3) está "
        "liberando mucha agua por transpiración y la ventilación es alta."
    )

    st.info(
        "**Prueba en el simulador:** Cierra totalmente las ventanas (apertura = 0) y observa "
        "qué ocurre en las condiciones de equilibrio pasivo."
    )
    st.markdown(
        "La temperatura se dispara hasta los **41,3 °C** y el DPV a **3,69 kPa**, con lo "
        "que el cultivo estaría sometido a un fuerte estrés hídrico. Con las ventanas cerradas "
        "el invernadero tiene muy limitado el intercambio de energía con el exterior y el "
        "efecto invernadero se dispara hasta 16,3 °C."
    )
    st.markdown(
        "Aumenta ahora el valor de radiación poco a poco y verás cómo sigue aumentando la "
        "temperatura hasta que el simulador deja de calcular, ya que las condiciones serían "
        "insostenibles para el cultivo."
    )

    st.divider()

    # ── Exportar ──
    st.subheader("Exportar actividad")
    st.markdown(
        "Genera un **PDF** con el escenario, la tabla de valores, los gráficos y tus "
        "respuestas al cuestionario."
    )

    if st.button("📄 Generar PDF y descargar", type="primary"):
        respuestas = [st.session_state.get(f'act1_resp_{i}', '') for i in range(1, 6)]
        df_export  = st.session_state['act1_tabla']

        try:
            pdf_bytes = _generate_pdf(
                nombre=st.session_state.get('act1_nombre', ''),
                df_tabla=df_export,
                respuestas=respuestas,
            )
            st.download_button(
                label="⬇️ Descargar PDF",
                data=pdf_bytes,
                file_name="Actividad1_Ventilacion.pdf",
                mime="application/pdf",
                type="primary",
            )
        except ImportError:
            st.error(
                "La librería **fpdf2** no está instalada. "
                "Ejecuta `pip install fpdf2` y reinicia la aplicación."
            )


# ───────────────────────── TAB 3: ACTIVIDAD 2 ─────────────────────────
with tab_act2:
    st.header("ACTIVIDAD 2: El LAI y la Refrigeración Evaporativa")

    # ── Escenario ──
    st.subheader("Escenario: Mediodía de verano")
    st.info(
        "**Invernadero climatizado** — Carga este escenario en el simulador "
        "(menú lateral › *📋 Actividad 2*) y luego modifica el parámetro indicado en cada apartado."
    )
    col_a2, col_b2 = st.columns(2)
    with col_a2:
        st.markdown("""
| Parámetro | Valor |
|---|---|
| Área del suelo | 8 000 m² |
| Área de ventanas | 1 600 m² (100 % apertura) |
| Estanqueidad | Normal |
| LAI (cultivo) | 3 |
""")
    with col_b2:
        st.markdown("""
| Parámetro | Valor |
|---|---|
| T exterior | 31 °C |
| HR exterior | 50 % |
| Radiación solar | 900 W/m² |
| Viento | 2 m/s |
| Cielo | Despejado |
| **T interior (fija)** | **33 °C** |
| Humidificación | 0 L·m⁻²·h⁻¹ |
""")
    st.markdown(
        "En un día típico de verano, incluso en un invernadero muy bien ventilado "
        "(superficies de ventanas = 20 % del suelo), las temperaturas son muy elevadas y "
        "provocan DPV muy altos que estresan los cultivos. El grado de desarrollo del cultivo "
        "puede modificar el microclima interior gracias a la transpiración. Vamos a comprobarlo."
    )

    st.divider()

    # ═══════════════════════════════════════════════
    # A) Efecto del LAI
    # ═══════════════════════════════════════════════
    st.subheader("A) Efecto del LAI sobre el microclima")
    st.markdown("""
1. Carga el escenario **Actividad 2** en el simulador.
2. Cambia el slider **LAI** a cada valor de la tabla (0,1 · 0,5 · 1 · 1,5 · 2 · 2,5 · 3).
3. En el apartado **Condiciones de equilibrio pasivo**, anota la *T de equilibrio* y el *DPV en equilibrio*.
4. Anota también la *Transpiración* del apartado superior.
5. Cuando hayas completado todas las filas, pulsa **Generar gráficos A**.
""")

    df_edited_a = st.data_editor(
        st.session_state['act2_tabla_a'],
        column_config={
            'LAI': st.column_config.NumberColumn('LAI', disabled=True, format="%.1f"),
            'Tequil. (°C)': st.column_config.NumberColumn(
                'Tequil. (°C)', min_value=0.0, format="%.1f",
                help="Temperatura de equilibrio pasivo (°C)"),
            'DPV equil. (kPa)': st.column_config.NumberColumn(
                'DPV equil. (kPa)', min_value=0.0, format="%.2f",
                help="DPV en condiciones de equilibrio pasivo"),
            'Transp. (g·m⁻²·h⁻¹)': st.column_config.NumberColumn(
                'Transp. (g·m⁻²·h⁻¹)', min_value=0.0, format="%.2f",
                help="Transpiración del cultivo"),
        },
        hide_index=True,
        use_container_width=True,
        key='act2_editor_a',
    )

    if st.button("📊 Generar gráficos A", type="primary", use_container_width=True):
        if not df_edited_a[_ACT2_COLS_A].notna().any().any():
            st.warning("Rellena al menos una fila antes de generar los gráficos.")
        else:
            st.session_state['act2_tabla_a'] = df_edited_a
            st.session_state['act2_show_charts_a'] = True

    if st.session_state['act2_show_charts_a']:
        df_ca = st.session_state['act2_tabla_a']
        mask_a = df_ca[_ACT2_COLS_A].notna().any(axis=1)
        df_ca = df_ca[mask_a]
        x_a = df_ca['LAI'].values
        specs_a = [
            ('Tequil. (°C)',       'T equilibrio (°C)',       '#E74C3C'),
            ('DPV equil. (kPa)',   'DPV equilibrio (kPa)',    '#F39C12'),
            ('Transp. (g·m⁻²·h⁻¹)', 'Transpiración (g·m⁻²·h⁻¹)', '#27AE60'),
        ]
        cols_a = st.columns(3)
        for (dc, lbl, clr), col in zip(specs_a, cols_a):
            with col:
                fig_a = go.Figure()
                fig_a.add_trace(go.Scatter(
                    x=x_a, y=df_ca[dc].values, mode='lines+markers',
                    line=dict(color=clr, width=2), marker=dict(size=7)))
                fig_a.update_layout(
                    title=lbl, xaxis_title='LAI', yaxis_title=lbl,
                    height=300, margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=dict(tickvals=_ACT2_LAI_VALS),
                )
                st.plotly_chart(fig_a, use_container_width=True, key=f"act2_chart_a_{dc}")

    st.divider()

    # ── Cuestionario A ──
    st.subheader("Cuestionario")
    st.markdown("Responde a cada pregunta en el cuadro correspondiente.")
    for idx, (titulo, pregunta) in enumerate(_ACT2_PREGUNTAS_A, 1):
        st.markdown(f"**{idx}. {titulo}**")
        st.markdown(pregunta)
        st.text_area(
            label=f"Tu respuesta A{idx}",
            placeholder="Escribe tu respuesta aquí…",
            height=120,
            key=f'act2_resp_a_{idx}',
            label_visibility='collapsed',
        )
        st.markdown("")

    # ── Explicación A: alternativas para reducir el DPV ──
    with st.expander("💡 Explicación: Alternativas para reducir el DPV"):
        st.markdown("### 1. Sombreado (Reducción de la Radiación Solar)")
        st.markdown("Si entra menos energía del sol, el aire y las hojas se calientan menos.")
        st.markdown(
            "- **Encalado (Blanqueo):** Se pinta el plástico o cristal con cal o productos blancos que reflejan la luz solar.\n"
            "- **Mallas de sombreo:** Redes de plástico (negras, aluminizadas) sobre el cultivo o la estructura para filtrar la insolación.\n"
            "- **Pantallas térmicas:** Cortinas internas móviles que se cierran en las horas de máxima insolación."
        )
        st.markdown("### 2. Refrigeración Evaporativa (Uso del agua para enfriar)")
        st.markdown(
            "Esta es la técnica más eficaz. Se basa en que el agua necesita calor para evaporarse: "
            "al forzar su evaporación dentro del invernadero, \"roba\" ese calor al aire, "
            "bajando la temperatura y subiendo la humedad (y bajando el DPV)."
        )
        st.markdown(
            "- **Nebulización (Misting/Fogging):** Boquillas de alta presión crean gotas tan finas que se evaporan antes de tocar la hoja.\n"
            "- **Panel evaporativo (Cooling Pad):** Pared de material poroso empapado en agua; los ventiladores hacen pasar el aire exterior por él antes de que entre al cultivo."
        )

    st.divider()

    # ═══════════════════════════════════════════════
    # B) Efecto de la temperatura interior
    # ═══════════════════════════════════════════════
    st.subheader("B) Efecto de la temperatura interior sobre el DPV")
    st.markdown(
        "Vamos a localizar la temperatura interior que baja el DPV a un valor menor de **1,6 kPa** "
        "(rango de seguridad para el cultivo)."
    )
    st.markdown("""
1. Mantén el escenario de la Actividad 2 (LAI = 3, Radiación = 900 W/m²).
2. Cambia el slider **Temperatura interior (°C)** a cada valor de la tabla.
3. Anota el **DPV** que aparece en el simulador (valor climatizado, no equilibrio).
4. Cuando hayas completado todas las filas, pulsa **Generar gráfico B**.
5. Comprueba que el valor obtenido está en torno a los **30ºC**.
""")

    df_edited_b = st.data_editor(
        st.session_state['act2_tabla_b'],
        column_config={
            'Tint (°C)': st.column_config.NumberColumn('Tint (°C)', disabled=True, format="%.1f"),
            'DPV (kPa)': st.column_config.NumberColumn(
                'DPV (kPa)', min_value=0.0, format="%.2f",
                help="DPV con esa temperatura interior"),
        },
        hide_index=True,
        use_container_width=True,
        key='act2_editor_b',
    )

    if st.button("📊 Generar gráfico B", type="primary", use_container_width=True):
        if not df_edited_b[_ACT2_COLS_B].notna().any().any():
            st.warning("Rellena al menos una fila antes de generar el gráfico.")
        else:
            st.session_state['act2_tabla_b'] = df_edited_b
            st.session_state['act2_show_charts_b'] = True

    if st.session_state['act2_show_charts_b']:
        df_cb = st.session_state['act2_tabla_b']
        mask_b = df_cb['DPV (kPa)'].notna()
        df_cb = df_cb[mask_b]
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(
            x=df_cb['Tint (°C)'].values, y=df_cb['DPV (kPa)'].values,
            mode='lines+markers', line=dict(color='#F39C12', width=2), marker=dict(size=7)))
        fig_b.add_hline(y=1.6, line_dash='dash', line_color='red',
                        annotation_text='Límite 1,6 kPa', annotation_position='bottom right')
        fig_b.update_layout(
            title='DPV vs Temperatura interior', xaxis_title='Tint (°C)', yaxis_title='DPV (kPa)',
            height=350, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_b, use_container_width=True, key="act2_chart_b")

    st.divider()

    # ═══════════════════════════════════════════════
    # C) Efecto del sombreado
    # ═══════════════════════════════════════════════
    st.subheader("C) Efecto del sombreado (reducción de la radiación exterior)")
    st.markdown(
        "Ya conocemos la temperatura a la que deberíamos estar para que el cultivo estuviera "
        "dentro del rango de seguridad (DPV < 1,6 kPa). Vamos a intentar alcanzarla "
        "modificando la radiación exterior."
    )
    st.markdown("""
1. Mantén el escenario de la Actividad 2 (LAI = 3, Tint = 33 °C).
2. Cambia el slider **Radiación solar exterior (W/m²)** a cada valor de la tabla.
3. Anota la **T de equilibrio** y el **DPV en equilibrio** del simulador.
4. Cuando hayas completado todas las filas, pulsa **Generar gráficos C**.
""")

    df_edited_c = st.data_editor(
        st.session_state['act2_tabla_c'],
        column_config={
            'Rad. ext. (W/m²)': st.column_config.NumberColumn('Rad. ext. (W/m²)', disabled=True, format="%d"),
            '% Sombreo': st.column_config.NumberColumn(
                '% Sombreo', disabled=True, format="%d"),
            'Tequil. (°C)': st.column_config.NumberColumn(
                'Tequil. (°C)', min_value=0.0, format="%.1f",
                help="Temperatura de equilibrio pasivo"),
            'DPV equil. (kPa)': st.column_config.NumberColumn(
                'DPV equil. (kPa)', min_value=0.0, format="%.2f",
                help="DPV en condiciones de equilibrio pasivo"),
        },
        hide_index=True,
        use_container_width=True,
        key='act2_editor_c',
    )

    if st.button("📊 Generar gráficos C", type="primary", use_container_width=True):
        if not df_edited_c[_ACT2_COLS_C].notna().any().any():
            st.warning("Rellena al menos una fila antes de generar los gráficos.")
        else:
            st.session_state['act2_tabla_c'] = df_edited_c
            st.session_state['act2_show_charts_c'] = True

    if st.session_state['act2_show_charts_c']:
        df_cc = st.session_state['act2_tabla_c']
        mask_c = df_cc[_ACT2_COLS_C].notna().any(axis=1)
        df_cc = df_cc[mask_c]
        x_c = df_cc['Rad. ext. (W/m²)'].values
        specs_c = [
            ('% Sombreo',        '% Sombreo',              '#8E44AD'),
            ('Tequil. (°C)',     'T equilibrio (°C)',       '#E74C3C'),
            ('DPV equil. (kPa)', 'DPV equilibrio (kPa)',   '#F39C12'),
        ]
        cols_c = st.columns(3)
        for (dc, lbl, clr), col in zip(specs_c, cols_c):
            with col:
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(
                    x=x_c, y=df_cc[dc].values, mode='lines+markers',
                    line=dict(color=clr, width=2), marker=dict(size=7)))
                fig_c.update_layout(
                    title=lbl, xaxis_title='Rad. ext. (W/m²)', yaxis_title=lbl,
                    height=300, margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_c, use_container_width=True, key=f"act2_chart_c_{dc}")

    st.info(
        "Para conseguir las condiciones ideales mediante limitación de la radiación, "
        "el % de sombreo debería ser muy elevado, lo que reduciría drásticamente la "
        "productividad (la producción es proporcional a la radiación interceptada). "
        "El sombreado ayuda a bajar ligeramente las temperaturas, pero al reducir la "
        "radiación también reduce la transpiración."
    )

    st.divider()

    # ═══════════════════════════════════════════════
    # D) Refrigeración evaporativa
    # ═══════════════════════════════════════════════
    st.subheader("D) Refrigeración evaporativa: efecto de la humidificación")
    st.markdown(
        "La refrigeración evaporativa sí es muy efectiva. Probémoslo añadiendo agua al aire "
        "mediante el parámetro **Humid./Deshumid.** de la barra lateral. "
        "**Nota:** en este apartado anota los valores directos del simulador (no los de equilibrio)."
    )
    st.markdown("""
1. Restaura el escenario Actividad 2 completo (Tint = 33 °C, LAI = 3, Rad = 900 W/m²).
2. Cambia el parámetro **Humid./Deshumid.** a cada valor de la tabla.
3. Anota la **Transpiración**, la **HR** y el **DPV** del simulador.
4. Cuando hayas completado todas las filas, pulsa **Generar gráficos D**.
""")

    df_edited_d = st.data_editor(
        st.session_state['act2_tabla_d'],
        column_config={
            'Humid. (L·m⁻²·h⁻¹)': st.column_config.NumberColumn(
                'Humid. (L·m⁻²·h⁻¹)', disabled=True, format="%.1f"),
            'Transp. (g·m⁻²·h⁻¹)': st.column_config.NumberColumn(
                'Transp. (g·m⁻²·h⁻¹)', min_value=0.0, format="%.2f",
                help="Transpiración del cultivo"),
            'HR (%)': st.column_config.NumberColumn(
                'HR (%)', min_value=0.0, max_value=100.0, format="%.1f",
                help="Humedad relativa interior"),
            'DPV (kPa)': st.column_config.NumberColumn(
                'DPV (kPa)', min_value=0.0, format="%.2f",
                help="Déficit de Presión de Vapor interior"),
        },
        hide_index=True,
        use_container_width=True,
        key='act2_editor_d',
    )

    if st.button("📊 Generar gráficos D", type="primary", use_container_width=True):
        if not df_edited_d[_ACT2_COLS_D].notna().any().any():
            st.warning("Rellena al menos una fila antes de generar los gráficos.")
        else:
            st.session_state['act2_tabla_d'] = df_edited_d
            st.session_state['act2_show_charts_d'] = True

    if st.session_state['act2_show_charts_d']:
        df_cd = st.session_state['act2_tabla_d']
        mask_d = df_cd[_ACT2_COLS_D].notna().any(axis=1)
        df_cd = df_cd[mask_d]
        x_d = df_cd['Humid. (L·m⁻²·h⁻¹)'].values
        specs_d = [
            ('Transp. (g·m⁻²·h⁻¹)', 'Transpiración (g·m⁻²·h⁻¹)', '#27AE60'),
            ('HR (%)',               'Humedad Relativa (%)',        '#1f77b4'),
            ('DPV (kPa)',            'DPV (kPa)',                   '#d62728'),
        ]
        cols_d = st.columns(3)
        for (dc, lbl, clr), col in zip(specs_d, cols_d):
            with col:
                fig_d = go.Figure()
                fig_d.add_trace(go.Scatter(
                    x=x_d, y=df_cd[dc].values, mode='lines+markers',
                    line=dict(color=clr, width=2), marker=dict(size=7)))
                if dc == 'DPV (kPa)':
                    fig_d.add_hline(y=1.6, line_dash='dash', line_color='red',
                                    annotation_text='Límite 1,6 kPa',
                                    annotation_position='top right')
                fig_d.update_layout(
                    title=lbl, xaxis_title='Humidificación (L·m⁻²·h⁻¹)', yaxis_title=lbl,
                    height=300, margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_d, use_container_width=True, key=f"act2_chart_d_{dc}")

    st.divider()

    # ── Cuestionario D ──
    st.subheader("Cuestionario")
    st.markdown("Responde a cada pregunta en el cuadro correspondiente.")
    for idx, (titulo, pregunta) in enumerate(_ACT2_PREGUNTAS_D, 1):
        st.markdown(f"**{idx}. {titulo}**")
        st.markdown(pregunta)
        st.text_area(
            label=f"Tu respuesta D{idx}",
            placeholder="Escribe tu respuesta aquí…",
            height=120,
            key=f'act2_resp_d_{idx}',
            label_visibility='collapsed',
        )
        st.markdown("")

    st.divider()

    # ── Explicación ──
    with st.expander("💡 Explicación"):
        st.markdown("### El DPV en verano: el reto de la refrigeración")
        st.markdown(
            "En un día de verano con alta radiación, el invernadero acumula mucho calor. "
            "Incluso con ventilación máxima, la temperatura interior se mantiene varios grados "
            "por encima del exterior. Cuanto mayor es la temperatura, mayor es la presión de "
            "vapor de saturación, por lo que el aire puede \"absorber\" mucho más vapor: el DPV "
            "se dispara."
        )
        st.divider()
        st.markdown("### El efecto del LAI sobre la temperatura de equilibrio")
        st.markdown(
            "La transpiración de las plantas actúa como un sistema de refrigeración natural. "
            "Al evaporar agua, las plantas \"roban\" calor latente al aire, enfriándolo. Cuanto "
            "mayor es el LAI (más hojas), más agua se evapora y más se enfría el invernadero. "
            "Sin embargo, este enfriamiento es limitado: incluso con LAI = 3, el DPV sigue "
            "siendo muy elevado en condiciones de verano extremo."
        )
        st.divider()
        st.markdown("### ¿Por qué la transpiración total sube aunque el DPV baje cuando se incrementa LAI?")
        st.markdown(
            "Hay dos efectos contrapuestos:\n\n"
            "- El **DPV** baja (el aire está más húmedo → menos \"fuerza\" de succión por unidad de hoja).\n"
            "- El **LAI** sube (hay muchas más hojas transpirando).\n\n"
            "El segundo efecto supera con creces al primero: el área foliar total crece 30 veces "
            "(de 0,1 a 3), mientras que el DPV solo baja un 32 %. El resultado es que la "
            "transpiración total del sistema aumenta enormemente."
        )
        st.divider()
        st.markdown("### Sombreado vs. Refrigeración Evaporativa")
        st.markdown(
            "El sombreado reduce la energía que entra, pero con un coste muy alto: reduce también "
            "la fotosíntesis y la producción. Para reducir el DPV por debajo de 1,6 kPa mediante "
            "sombreado solo, habría que eliminar más del 85 % de la radiación, lo que es "
            "agronómicamente inviable.\n\n"
            "La **refrigeración evaporativa** (nebulización), en cambio, ataca directamente el "
            "problema: al añadir agua se consume mayor cantidad de energía en evaporarla y esta energía no está disponible para calentar el aire "
            ", lo que enfría el aire y aumenta su humedad simultáneamente, reduciendo "
            "el DPV de forma eficaz con un impacto mínimo sobre la radiación disponible para el cultivo."
        )

    # ── Nombre ──
    st.text_input(
        "Nombre del estudiante (aparecerá en el PDF)",
        placeholder="Nombre y apellidos",
        key='act2_nombre',
    )

    st.divider()

    # ── Exportar ──
    st.subheader("Exportar actividad")
    st.markdown(
        "Genera un **PDF** con el escenario, las tablas de valores, los gráficos y tus "
        "respuestas al cuestionario."
    )
    if st.button("📄 Generar PDF y descargar", type="primary", key="act2_pdf_btn"):
        resp_a = [st.session_state.get(f'act2_resp_a_{i}', '') for i in range(1, 4)]
        resp_d = [st.session_state.get(f'act2_resp_d_{i}', '') for i in range(1, 3)]
        try:
            pdf_bytes2 = _generate_pdf_act2(
                nombre=st.session_state.get('act2_nombre', ''),
                df_a=st.session_state['act2_tabla_a'],
                df_b=st.session_state['act2_tabla_b'],
                df_c=st.session_state['act2_tabla_c'],
                df_d=st.session_state['act2_tabla_d'],
                resp_a=resp_a,
                resp_d=resp_d,
            )
            st.download_button(
                label="⬇️ Descargar PDF",
                data=pdf_bytes2,
                file_name="Actividad2_LAI_RefrigeracionEvaporativa.pdf",
                mime="application/pdf",
                type="primary",
                key="act2_pdf_download",
            )
        except ImportError:
            st.error(
                "La librería **fpdf2** no está instalada. "
                "Ejecuta `pip install fpdf2` y reinicia la aplicación."
            )


# ───────────────────────── TAB 4: ACTIVIDAD 3 ─────────────────────────
with tab_act3:
    st.header("ACTIVIDAD 3: Comportamiento Nocturno del Invernadero")

    # ── Escenario ──
    st.subheader("Escenario: Noche de invierno")
    st.info(
        "Carga el escenario **📋 Actividad 3** en el simulador (menú lateral) y sigue "
        "las instrucciones de cada apartado."
    )
    col_a3, col_b3 = st.columns(2)
    with col_a3:
        st.markdown("""
| Parámetro | Valor |
|---|---|
| Área del suelo | 8 000 m² |
| Área de ventanas | 1 600 m² |
| Estanqueidad | Normal |
| LAI (cultivo) | 3 |
""")
    with col_b3:
        st.markdown("""
| Parámetro | Valor |
|---|---|
| T exterior | 11 °C |
| HR exterior | 75 % |
| Radiación solar | 0 W/m² |
| Viento | 1 m/s |
| **T interior (fija)** | **15 °C** |
| Apertura ventanas | 0 % |
| Humidificación | 0 L·m⁻²·h⁻¹ |
""")

    st.markdown(
        "Durante la noche, el invernadero deja de ganar energía solar y comienza a perderla. "
        "Las temperaturas caen, el aire se satura de humedad (HR ≈ 100 %, DPV ≈ 0) y la "
        "transpiración del cultivo es prácticamente nula. "
        "El tipo de cielo (despejado o nublado) determina cuánto calor se pierde hacia el exterior."
    )
    st.divider()

    # ══════════════════════════════════════════════
    # A) Cielo despejado
    # ══════════════════════════════════════════════
    st.subheader("A) Cielo despejado — ventanas cerradas")
    st.markdown("""
1. Carga el escenario **Actividad 3** en el simulador. El cielo debe estar en **Cielo despejado** y la apertura de ventanas en **0 %**.
2. Lee los valores del simulador y anótalos en la tabla:
   - **HR (%)**, **DPV (kPa)** y **Transpiración** → sección *Condiciones Interiores*.
   - **Ventilación** → métrica *Ventilación* del simulador.
   - **Text** y **Tinterior** → son los valores del escenario (ya rellenos).
   - **Tequil.** y **DPVequil.** → sección *Condiciones de equilibrio pasivo*.
""")

    df_edited_3a = st.data_editor(
        st.session_state['act3_tabla_a'],
        column_config={
            'HR (%)': st.column_config.NumberColumn(
                'HR (%)', min_value=0.0, max_value=100.0, format="%.1f",
                help="Humedad relativa interior (%)"),
            'DPV (kPa)': st.column_config.NumberColumn(
                'DPV (kPa)', min_value=0.0, format="%.2f",
                help="Déficit de presión de vapor interior"),
            _ACT3_COL_TRANSP: st.column_config.NumberColumn(
                _ACT3_COL_TRANSP, min_value=0.0, format="%.2f",
                help="Transpiración del cultivo"),
            _ACT3_COL_VENT: st.column_config.NumberColumn(
                _ACT3_COL_VENT, min_value=0.0, format="%.2f",
                help="Ventilación total"),
            _ACT3_COL_TEXT: st.column_config.NumberColumn(
                _ACT3_COL_TEXT, format="%.1f", disabled=True,
                help="Temperatura exterior (fija del escenario)"),
            _ACT3_COL_TINT: st.column_config.NumberColumn(
                _ACT3_COL_TINT, format="%.1f", disabled=True,
                help="Temperatura interior consigna (fija del escenario)"),
            _ACT3_COL_TEQUIL: st.column_config.NumberColumn(
                _ACT3_COL_TEQUIL, format="%.1f",
                help="Temperatura de equilibrio pasivo"),
            'DPVequil. (kPa)': st.column_config.NumberColumn(
                'DPVequil. (kPa)', min_value=0.0, format="%.2f",
                help="DPV en condiciones de equilibrio pasivo"),
        },
        hide_index=True,
        use_container_width=True,
        key='act3_editor_a',
    )
    st.session_state['act3_tabla_a'] = df_edited_3a

    st.divider()

    # ── Cuestionario A ──
    st.subheader("Cuestionario A")
    st.markdown("Responde a cada pregunta en el cuadro correspondiente.")
    for idx, (titulo, pregunta) in enumerate(_ACT3_PREGUNTAS_A, 1):
        st.markdown(f"**{idx}. {titulo}**")
        st.markdown(pregunta)
        st.text_area(
            label=f"Tu respuesta A{idx}",
            placeholder="Escribe tu respuesta aquí…",
            height=120,
            key=f'act3_resp_a_{idx}',
            label_visibility='collapsed',
        )
        st.markdown("")

    st.divider()

    # ══════════════════════════════════════════════
    # B) Cielo nublado
    # ══════════════════════════════════════════════
    st.subheader("B) Cielo nublado — ventanas cerradas")
    st.markdown("""
1. En el simulador, cambia el selector **Estado del cielo** a **Cielo nublado** (mantén el resto igual).
2. Lee los nuevos valores y anótalos en la tabla B (mismas variables que antes).
""")

    df_edited_3b = st.data_editor(
        st.session_state['act3_tabla_b'],
        column_config={
            'HR (%)': st.column_config.NumberColumn(
                'HR (%)', min_value=0.0, max_value=100.0, format="%.1f"),
            'DPV (kPa)': st.column_config.NumberColumn(
                'DPV (kPa)', min_value=0.0, format="%.2f"),
            _ACT3_COL_TRANSP: st.column_config.NumberColumn(
                _ACT3_COL_TRANSP, min_value=0.0, format="%.2f"),
            _ACT3_COL_VENT: st.column_config.NumberColumn(
                _ACT3_COL_VENT, min_value=0.0, format="%.2f"),
            _ACT3_COL_TEXT: st.column_config.NumberColumn(
                _ACT3_COL_TEXT, format="%.1f", disabled=True),
            _ACT3_COL_TINT: st.column_config.NumberColumn(
                _ACT3_COL_TINT, format="%.1f", disabled=True),
            _ACT3_COL_TEQUIL: st.column_config.NumberColumn(
                _ACT3_COL_TEQUIL, format="%.1f"),
            'DPVequil. (kPa)': st.column_config.NumberColumn(
                'DPVequil. (kPa)', min_value=0.0, format="%.2f"),
        },
        hide_index=True,
        use_container_width=True,
        key='act3_editor_b',
    )
    st.session_state['act3_tabla_b'] = df_edited_3b

    st.divider()

    # ── Cuestionario B ──
    st.subheader("Cuestionario B")
    st.markdown("Responde a cada pregunta en el cuadro correspondiente.")
    for idx, (titulo, pregunta) in enumerate(_ACT3_PREGUNTAS_B, 1):
        st.markdown(f"**{idx}. {titulo}**")
        st.markdown(pregunta)
        st.text_area(
            label=f"Tu respuesta B{idx}",
            placeholder="Escribe tu respuesta aquí…",
            height=120,
            key=f'act3_resp_b_{idx}',
            label_visibility='collapsed',
        )
        st.markdown("")

    st.divider()

    # ── Explicación ──
    with st.expander("💡 Explicación"):
        st.markdown("### Comportamiento nocturno del invernadero")
        st.markdown(
            "Durante la noche, sin radiación solar, el invernadero pierde calor hacia el exterior. "
            "La temperatura interior cae continuamente. El aire se enfría, pierde capacidad de contener "
            "vapor de agua y se satura: la **Humedad Relativa** sube al 100 % y el **DPV** cae a 0. "
            "Los estomas de las plantas están cerrados, por lo que la **transpiración es nula o mínima**."
        )
        st.divider()
        st.markdown("### Inversión térmica con cielo despejado")
        st.markdown(
            "Con cielo despejado, el invernadero emite radiación de onda larga directamente hacia "
            "la atmósfera sin impedimentos. Esto produce un **enfriamiento radiativo intenso**: la temperatura "
            "de equilibrio puede caer **por debajo de la temperatura exterior**, fenómeno conocido como "
            "**inversión térmica**. En invernaderos con plástico no térmico este efecto es más pronunciado."
        )
        st.divider()
        st.markdown("### El efecto del cielo nublado")
        st.markdown(
            "Las nubes actúan como una **manta**: absorben la radiación de onda larga emitida por el "
            "invernadero y la reemiten de vuelta hacia la superficie. Esto reduce el enfriamiento radiativo "
            "y eleva la temperatura de equilibrio **por encima de la temperatura exterior**. "
            "En estas condiciones no se produce inversión térmica."
        )
        st.divider()
        st.markdown("### Consecuencias prácticas para el manejo")
        st.markdown(
            "- **Cielo despejado (inversión térmica):** abrir las ventanas introduce aire exterior "
            "más caliente y si se está cerca de la saturación genera déficit.\n"
            "- **Cielo nublado (sin inversión térmica):** una ventilación moderada puede renovar el "
            "aire saturado y reducir el riesgo de enfermedades fúngicas aunque baja las temperaturas del aire."
        )

    st.markdown("### ¿Qué ocurriría en un invernadero no climatizado?")
    st.markdown(
        "En un invernadero **no climatizado**, la temperatura interior no está controlada por ninguna "
        "consigna: evoluciona libremente hasta alcanzar la **temperatura de equilibrio** marcada por "
        "el balance energético entre las pérdidas hacia el exterior y los aportes internos.\n\n"
        "- Con **cielo despejado**, la temperatura de equilibrio es inferior a la exterior (inversión "
        "térmica), por lo que la temperatura interior del invernadero no climatizado sería incluso "
        "más baja que la del exterior: mayor riesgo de daños por frío en cultivos sensibles.\n"
        "- Con **cielo nublado**, la temperatura de equilibrio es superior a la exterior, por lo que "
        "el invernadero no climatizado conserva mejor el calor y protege mejor al cultivo de las "
        "bajas temperaturas que el exterior."
    )

    # ── Nombre ──
    st.text_input(
        "Nombre del estudiante (aparecerá en el PDF)",
        placeholder="Nombre y apellidos",
        key='act3_nombre',
    )

    st.divider()

    # ── Exportar ──
    st.subheader("Exportar actividad")
    st.markdown(
        "Genera un **PDF** con el escenario, las tablas de valores, los gráficos comparativos "
        "y tus respuestas al cuestionario."
    )
    if st.button("📄 Generar PDF y descargar", type="primary", key="act3_pdf_btn"):
        resp_a3 = [st.session_state.get(f'act3_resp_a_{i}', '') for i in range(1, 5)]
        resp_b3 = [st.session_state.get(f'act3_resp_b_{i}', '') for i in range(1, 5)]
        try:
            pdf_bytes3 = _generate_pdf_act3(
                nombre=st.session_state.get('act3_nombre', ''),
                df_a=st.session_state['act3_tabla_a'],
                df_b=st.session_state['act3_tabla_b'],
                resp_a=resp_a3,
                resp_b=resp_b3,
            )
            st.download_button(
                label="⬇️ Descargar PDF",
                data=pdf_bytes3,
                file_name="Actividad3_ComportamientoNocturno.pdf",
                mime="application/pdf",
                type="primary",
                key="act3_pdf_download",
            )
        except ImportError:
            st.error(
                "La librería **fpdf2** no está instalada. "
                "Ejecuta `pip install fpdf2` y reinicia la aplicación."
            )


# ───────────────────────── TAB 5: ACTIVIDAD 4 — TEST ─────────────────────────
with tab_act4:
    st.header("ACTIVIDAD 4: Test de Conocimiento")
    st.markdown(
        "Pon a prueba lo que has aprendido con el simulador. "
        "Responde las **10 preguntas** de opción múltiple (solo una respuesta es correcta) "
        "y pulsa **Comprobar respuestas** cuando hayas terminado."
    )
    st.divider()

    letras_ui = ['a', 'b', 'c', 'd']

    # ── Preguntas ──
    for i, q in enumerate(_ACT4_PREGUNTAS):
        st.markdown(f"**{i+1}. {q['pregunta']}**")

        opciones_con_letra = [f"{letras_ui[j]}) {op}" for j, op in enumerate(q['opciones'])]
        opciones_con_letra_none = ["— Selecciona una opción —"] + opciones_con_letra

        # Índice actual guardado en session_state (None → 0 = placeholder)
        sel_idx = st.session_state.get(f'act4_resp_{i}')
        radio_default = 0 if sel_idx is None else sel_idx + 1

        eleccion = st.radio(
            label=f"Pregunta {i+1}",
            options=opciones_con_letra_none,
            index=radio_default,
            key=f'act4_radio_{i}',
            label_visibility='collapsed',
        )

        # Guardamos la selección (None si aún en placeholder)
        if eleccion == "— Selecciona una opción —":
            st.session_state[f'act4_resp_{i}'] = None
        else:
            st.session_state[f'act4_resp_{i}'] = opciones_con_letra_none.index(eleccion) - 1

        # Feedback inmediato tras submit
        if st.session_state['act4_submitted']:
            resp_idx = st.session_state.get(f'act4_resp_{i}')
            if resp_idx is None:
                st.warning("Sin respuesta.")
            elif resp_idx == q['correcta']:
                st.success(f"✅ Correcto — {q['explicacion']}")
            else:
                correcta_txt = f"{letras_ui[q['correcta']]}) {q['opciones'][q['correcta']]}"
                st.error(
                    f"❌ Incorrecto. La respuesta correcta es: **{correcta_txt}**\n\n"
                    f"{q['explicacion']}"
                )
        st.markdown("")

    st.divider()

    # ── Botones ──
    col_chk, col_rst = st.columns([3, 1])
    with col_chk:
        if st.button("✅ Comprobar respuestas", type="primary", use_container_width=True):
            sin_resp = [i+1 for i in range(len(_ACT4_PREGUNTAS))
                        if st.session_state.get(f'act4_resp_{i}') is None]
            if sin_resp:
                st.warning(
                    f"Tienes {len(sin_resp)} pregunta(s) sin responder "
                    f"({', '.join(str(n) for n in sin_resp)}). Puedes enviar igualmente."
                )
            st.session_state['act4_submitted'] = True
            st.rerun()
    with col_rst:
        if st.button("🔄 Reiniciar test", use_container_width=True):
            for i in range(len(_ACT4_PREGUNTAS)):
                st.session_state[f'act4_resp_{i}'] = None
                if f'act4_radio_{i}' in st.session_state:
                    del st.session_state[f'act4_radio_{i}']
            st.session_state['act4_submitted'] = False
            st.rerun()

    # ── Marcador ──
    if st.session_state['act4_submitted']:
        n_ok = sum(
            1 for i, q in enumerate(_ACT4_PREGUNTAS)
            if st.session_state.get(f'act4_resp_{i}') == q['correcta']
        )
        n_total = len(_ACT4_PREGUNTAS)
        pct = n_ok / n_total * 100
        if pct >= 80:
            st.success(f"🏆 Puntuación: **{n_ok} / {n_total}** ({pct:.0f} %) — ¡Excelente!")
        elif pct >= 60:
            st.info(f"📊 Puntuación: **{n_ok} / {n_total}** ({pct:.0f} %) — Bien, sigue repasando.")
        else:
            st.warning(f"📊 Puntuación: **{n_ok} / {n_total}** ({pct:.0f} %) — Repasa los conceptos del simulador.")

    st.divider()

    # ── Nombre y exportar ──
    st.text_input(
        "Nombre del estudiante (aparecerá en el PDF)",
        placeholder="Nombre y apellidos",
        key='act4_nombre',
    )

    st.subheader("Exportar resultados")
    st.markdown("Genera un **PDF** con tus respuestas, la corrección y las explicaciones.")

    if st.button("📄 Generar PDF y descargar", type="primary", key="act4_pdf_btn"):
        if not st.session_state['act4_submitted']:
            st.warning("Pulsa primero **Comprobar respuestas** antes de exportar.")
        else:
            respuestas_export = {
                i: st.session_state.get(f'act4_resp_{i}')
                for i in range(len(_ACT4_PREGUNTAS))
            }
            try:
                pdf_bytes4 = _generate_pdf_act4(
                    nombre=st.session_state.get('act4_nombre', ''),
                    respuestas=respuestas_export,
                )
                st.download_button(
                    label="⬇️ Descargar PDF",
                    data=pdf_bytes4,
                    file_name="Actividad4_Test_Conocimiento.pdf",
                    mime="application/pdf",
                    type="primary",
                    key="act4_pdf_download",
                )
            except ImportError:
                st.error(
                    "La librería **fpdf2** no está instalada. "
                    "Ejecuta `pip install fpdf2` y reinicia la aplicación."
                )


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
