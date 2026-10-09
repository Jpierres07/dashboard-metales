import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore', category=RuntimeWarning)
st.set_page_config(page_title='Observatorio de metales', page_icon='⛏️', layout='wide')

COLUMNAS = {
    'Cobre': 'Cobre (US$/TM)', 'Oro': 'Oro (US$/onza troy)',
    'Plata': 'Plata (US$/onza troy)', 'Zinc': 'Zinc (US$/TM)',
    'Plomo': 'Plomo (US$/TM)'
}
UNIDADES = {m: ('US$/TM' if m in ['Cobre', 'Zinc', 'Plomo'] else 'US$/onza troy') for m in COLUMNAS}
METODOS = ['Promedio móvil simple', 'Promedio móvil doble', 'SES', 'Holt']

@st.cache_data
def cargar(archivo):
    df = pd.read_csv(archivo, sep=';', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    # Normalizar el encabezado de fecha, si fuese necesario.
    primera = df.columns[0]
    df = df.rename(columns={primera: 'AñoMes'})
    faltan = [x for x in ['AñoMes', *COLUMNAS.values()] if x not in df.columns]
    if faltan:
        raise ValueError(f'Faltan columnas: {faltan}. Disponibles: {list(df.columns)}')
    p = df['AñoMes'].astype(str).str.strip().str.upper().str.extract(r'^(\d{4})M(0?[1-9]|1[0-2])$')
    df['Fecha'] = pd.to_datetime(p[0] + '-' + p[1].str.zfill(2) + '-01', errors='coerce')
    for metal, col in COLUMNAS.items():
        df[metal] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Fecha']).sort_values('Fecha').drop_duplicates('Fecha', keep='last').set_index('Fecha')
    return df[list(COLUMNAS)].asfreq('MS')


def serie_metal(df, metal):
    s = df[metal].astype(float)
    if s.notna().sum() < 18:
        return pd.Series(dtype=float)
    s = s.loc[s.first_valid_index():s.last_valid_index()]
    return s.interpolate(method='time', limit_area='inside')


def pronosticar(nombre, s, horizonte, ventana=12):
    s = s.dropna().astype(float)
    if len(s) < max(5, ventana * 2 if nombre == 'Promedio móvil doble' else ventana):
        raise ValueError('Datos insuficientes para este modelo')
    if nombre == 'Promedio móvil simple':
        pred = np.repeat(s.iloc[-ventana:].mean(), horizonte)
    elif nombre == 'Promedio móvil doble':
        # Método de Brown: promedio móvil de orden n aplicado dos veces.
        m1 = s.rolling(ventana).mean()
        m2 = m1.rolling(ventana).mean()
        nivel = 2 * m1.iloc[-1] - m2.iloc[-1]
        pendiente = 2 * (m1.iloc[-1] - m2.iloc[-1]) / (ventana - 1)
        pred = nivel + pendiente * np.arange(1, horizonte + 1)
    elif nombre == 'SES':
        ajuste = SimpleExpSmoothing(s, initialization_method='estimated').fit(optimized=True)
        pred = ajuste.forecast(horizonte).to_numpy()
    elif nombre == 'Holt':
        ajuste = Holt(s, initialization_method='estimated').fit(optimized=True)
        pred = ajuste.forecast(horizonte).to_numpy()
    else:
        raise ValueError('Modelo desconocido')
    pred = np.asarray(pred, dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError('Pronóstico no finito')
    return pred


def evaluar(s, ventana=12, meses=12):
    # Evaluación con origen móvil: cada predicción usa únicamente meses anteriores.
    n = min(meses, len(s) - max(30, 2 * ventana + 3))
    if n < 3:
        return pd.DataFrame(), {}
    resultados, predicciones = [], {}
    for nombre in METODOS:
        fechas, reales, estimados = [], [], []
        try:
            for i in range(len(s) - n, len(s)):
                estimado = pronosticar(nombre, s.iloc[:i], 1, ventana)[0]
                fechas.append(s.index[i])
                reales.append(float(s.iloc[i]))
                estimados.append(float(estimado))
            reales, estimados = np.array(reales), np.array(estimados)
            mae = np.mean(np.abs(reales - estimados))
            rmse = np.sqrt(np.mean((reales - estimados) ** 2))
            mask = reales != 0
            mape = np.mean(np.abs((reales[mask] - estimados[mask]) / reales[mask])) * 100 if mask.any() else np.nan
            resultados.append({'Modelo': nombre, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape})
            predicciones[nombre] = pd.Series(estimados, index=fechas)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            continue
    tabla = pd.DataFrame(resultados)
    if not tabla.empty:
        tabla = tabla.sort_values('RMSE').reset_index(drop=True)
    return tabla, predicciones


def indicadores(s):
    ult = s.iloc[-1]
    mes = (ult / s.iloc[-2] - 1) * 100 if len(s) > 1 and s.iloc[-2] != 0 else np.nan
    base = s.iloc[-7] if len(s) >= 7 else s.iloc[0]
    seis = (ult / base - 1) * 100 if base != 0 else np.nan
    vol = s.pct_change(fill_method=None).std() * 100
    return ult, mes, seis, vol


def tabla_mercado(df):
    filas = []
    for metal in COLUMNAS:
        s = serie_metal(df, metal)
        if len(s) < 2:
            continue
        precio, mensual, semestral, vol = indicadores(s)
        filas.append({'Metal': metal, 'Precio (US$)': precio, 'Unidad': UNIDADES[metal],
                      'Cambio mensual (%)': mensual, 'Cambio 6 meses (%)': semestral,
                      'Volatilidad mensual (%)': vol})
    return pd.DataFrame(filas)


def grafico_linea(s, titulo, unidad, ultimos=120):
    s = s.tail(ultimos)
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, name='Precio', mode='lines', line={'width': 3}))
    fig.update_layout(title=titulo, xaxis_title='Mes', yaxis_title=unidad, template='plotly_white', height=430)
    return fig

st.sidebar.title('⛏️ Observatorio minero')
archivo = st.sidebar.file_uploader('Cargar CSV (opcional)', type=['csv'])
try:
    datos = cargar(archivo if archivo is not None else ('Data(2).csv' if __import__('pathlib').Path('Data(2).csv').exists() else 'Data.csv'))
except Exception as exc:
    st.error(f'No se pudo cargar el CSV: {exc}')
    st.info('Coloca Data(2).csv o Data.csv junto a app_metales_corregido.py, o carga el archivo desde la barra lateral.')
    st.stop()

pagina = st.sidebar.radio('Sección', [
    '🏠 Panorama del mercado', '🔎 Explorar un metal', '🔮 Pronósticos',
    '⚖️ Riesgo y comparación', '📊 Sustento estadístico'
])
metal = st.sidebar.selectbox('Metal', list(COLUMNAS))
horizonte = st.sidebar.select_slider('Meses a pronosticar', options=[1, 3, 6, 9, 12], value=6)
ventana = st.sidebar.slider('Ventana de promedios móviles (meses)', 3, 18, 12)
s = serie_metal(datos, metal)
mercado = tabla_mercado(datos)
if s.empty:
    st.error(f'No hay suficientes datos para {metal}.')
    st.stop()

st.sidebar.caption(f'Último mes del archivo: {datos.index.max():%m/%Y}')
st.sidebar.caption('Si existen meses internos sin dato, se interpolan para el análisis. Verifica su cantidad en Sustento estadístico.')

if pagina == '🏠 Panorama del mercado':
    st.title('⛏️ Panorama del mercado de metales')
    st.write('Compare el comportamiento de cinco metales antes de estudiar una posible inversión.')
    cols = st.columns(5)
    for i, m in enumerate(COLUMNAS):
        fila = mercado.loc[mercado.Metal == m]
        if not fila.empty:
            r = fila.iloc[0]
            cols[i].metric(m, f"US$ {r['Precio (US$)']:,.2f}", f"{r['Cambio mensual (%)']:+.2f}%")
            cols[i].caption(UNIDADES[m])
    st.subheader('¿Qué metal ha aumentado más en términos porcentuales?')
    periodo = st.selectbox('Periodo', ['12 meses', '5 años', 'Todo el historial'])
    cantidad = {'12 meses': 12, '5 años': 60, 'Todo el historial': None}[periodo]
    fig = go.Figure()
    for m in COLUMNAS:
        sm = serie_metal(datos, m)
        if cantidad:
            sm = sm.tail(cantidad)
        if len(sm) >= 2 and sm.iloc[0] != 0:
            fig.add_trace(go.Scatter(x=sm.index, y=100 * sm / sm.iloc[0], name=m, mode='lines'))
    fig.add_hline(y=100, line_dash='dash')
    fig.update_layout(yaxis_title='Índice base 100 (sin unidad)', template='plotly_white', height=480)
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Índice 120 = aumento de 20% desde el inicio del periodo elegido. No iguala los precios ni sus unidades.')
    st.dataframe(mercado.style.format({'Precio (US$)': '{:,.2f}', 'Cambio mensual (%)': '{:+.2f}',
                                       'Cambio 6 meses (%)': '{:+.2f}', 'Volatilidad mensual (%)': '{:.2f}'}),
                 hide_index=True, use_container_width=True)

elif pagina == '🔎 Explorar un metal':
    st.title(f'🔎 Conozcamos el {metal.lower()}')
    precio, mensual, semestral, vol = indicadores(s)
    c1, c2, c3 = st.columns(3)
    c1.metric('Último precio observado', f'US$ {precio:,.2f}', UNIDADES[metal])
    c2.metric('Variación mensual', f'{mensual:+.2f}%')
    c3.metric('Variación de 6 meses', f'{semestral:+.2f}%')
    ultimos = st.selectbox('Historia visible', [12, 36, 60, 120, len(s)], index=3 if len(s) >= 120 else 0)
    st.plotly_chart(grafico_linea(s, f'Precio histórico del {metal.lower()}', UNIDADES[metal], ultimos), use_container_width=True)
    texto = 'al alza' if semestral > 2 else 'a la baja' if semestral < -2 else 'relativamente estable'
    st.info(f'En los últimos seis meses el precio ha estado {texto}. Su volatilidad mensual histórica es {vol:.2f}%.')

elif pagina == '🔮 Pronósticos':
    st.title(f'🔮 Pronóstico del {metal.lower()}')
    st.write('Los cuatro métodos se evalúan con pronósticos de un mes adelante, actualizando el origen de predicción sin mirar el futuro.')
    with st.spinner('Calculando modelos...'):
        tabla, predicciones = evaluar(s, ventana)
    if tabla.empty:
        st.warning('No hay suficientes datos para comparar los cuatro métodos.')
        st.stop()
    st.dataframe(tabla.style.format({'MAE': '{:,.3f}', 'RMSE': '{:,.3f}', 'MAPE (%)': '{:.2f}'}),
                 hide_index=True, use_container_width=True)
    ganador = tabla.iloc[0]['Modelo']
    st.success(f'Menor RMSE de evaluación: {ganador}')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.tail(36).index, y=s.tail(36), name='Precio observado'))
    fig.add_trace(go.Scatter(x=predicciones[ganador].index, y=predicciones[ganador],
                             name='Predicción histórica (1 mes)', line={'dash': 'dash'}))
    fig.update_layout(title='Evaluación: observado frente a estimado', template='plotly_white', yaxis_title=UNIDADES[metal])
    st.plotly_chart(fig, use_container_width=True)
    futuro = pronosticar(ganador, s, horizonte, ventana)
    fechas = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=horizonte, freq='MS')
    proyeccion = pd.DataFrame({'Fecha': fechas, 'Pronóstico': futuro})
    cambio = (futuro[-1] / s.iloc[-1] - 1) * 100 if s.iloc[-1] != 0 else np.nan
    c1, c2, c3 = st.columns(3)
    c1.metric('Último precio', f'US$ {s.iloc[-1]:,.2f}')
    c2.metric(f'Precio proyectado a {horizonte} meses', f'US$ {futuro[-1]:,.2f}')
    c3.metric('Cambio proyectado', f'{cambio:+.2f}%')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.tail(36).index, y=s.tail(36), name='Histórico'))
    fig.add_trace(go.Scatter(x=[s.index[-1], *fechas], y=[s.iloc[-1], *futuro],
                             name='Proyección', line={'dash': 'dash', 'width': 3}))
    fig.update_layout(title='Escenario de precios futuros', yaxis_title=UNIDADES[metal], template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(proyeccion.style.format({'Pronóstico': '{:,.2f}'}), hide_index=True)
    st.download_button('Descargar pronóstico CSV', proyeccion.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f'pronostico_{metal.lower()}.csv', mime='text/csv')
    st.warning('La evaluación mide pronósticos a un mes; la precisión a 3–12 meses puede ser diferente. No se garantizan ganancias.')

elif pagina == '⚖️ Riesgo y comparación':
    st.title('⚖️ Riesgo y comparación')
    st.write('La volatilidad mide cuánto fluctúan históricamente las variaciones porcentuales mensuales de cada metal.')
    fig = px.bar(mercado.sort_values('Volatilidad mensual (%)'), x='Volatilidad mensual (%)', y='Metal',
                 orientation='h', text='Volatilidad mensual (%)', title='Volatilidad histórica mensual')
    fig.update_traces(texttemplate='%{text:.2f}%')
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(mercado, x='Volatilidad mensual (%)', y='Cambio 6 meses (%)', text='Metal',
                     title='Crecimiento reciente frente a volatilidad histórica')
    fig.update_traces(textposition='top center', marker={'size': 13})
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Un mayor crecimiento pasado no implica mayor rentabilidad futura. La volatilidad no mide todos los riesgos de invertir.')

else:
    st.title('📊 Sustento estadístico')
    st.subheader('1. Descomposición clásica aditiva y multiplicativa')
    if len(s) >= 24:
        tipo = st.radio('Tipo', ['Aditiva', 'Multiplicativa'], horizontal=True)
        if tipo == 'Multiplicativa' and (s <= 0).any():
            st.error('La descomposición multiplicativa requiere precios positivos.')
        else:
            des = seasonal_decompose(s, model='additive' if tipo == 'Aditiva' else 'multiplicative',
                                     period=12, extrapolate_trend='freq')
            componentes = {'Original': des.observed, 'Tendencia': des.trend,
                           'Estacionalidad': des.seasonal, 'Residual': des.resid}
            seleccion = st.selectbox('Componente', list(componentes))
            st.plotly_chart(grafico_linea(componentes[seleccion], f'{tipo}: {seleccion}',
                                           UNIDADES[metal] if tipo == 'Aditiva' or seleccion == 'Original' else 'Índice / factor', len(s)),
                            use_container_width=True)
            st.write('Aditiva: Y = T + E + R' if tipo == 'Aditiva' else 'Multiplicativa: Y = T × E × R')
            st.download_button('Descargar componentes', pd.DataFrame(componentes).to_csv().encode('utf-8-sig'),
                               file_name=f'descomposicion_{metal.lower()}.csv', mime='text/csv')
    else:
        st.warning('La descomposición mensual necesita al menos 24 meses.')
    st.subheader('2. Promedios móviles simple y doble')
    pm1 = s.rolling(ventana).mean()
    pm2 = pm1.rolling(ventana).mean()
    fig = go.Figure()
    for nombre, valores in [('Precio observado', s), ('PM simple', pm1), ('PM doble', pm2)]:
        fig.add_trace(go.Scatter(x=valores.tail(120).index, y=valores.tail(120), name=nombre))
    fig.update_layout(template='plotly_white', yaxis_title=UNIDADES[metal])
    st.plotly_chart(fig, use_container_width=True)
    st.caption('El promedio móvil doble aplica un segundo promedio móvil sobre el primero; para pronosticar se usa la extrapolación de Brown.')
    st.subheader('3. Comparación: PM simple, PM doble, SES y Holt')
    tabla, _ = evaluar(s, ventana)
    if not tabla.empty:
        st.dataframe(tabla.style.format({'MAE': '{:.3f}', 'RMSE': '{:.3f}', 'MAPE (%)': '{:.2f}'}),
                     hide_index=True, use_container_width=True)
    st.caption(f'Meses internos sin precio original para {metal}: '
               f'{datos[metal].loc[datos[metal].first_valid_index():datos[metal].last_valid_index()].isna().sum()} '
               '(interpolados solamente para el análisis).')
    st.info('MAE y RMSE están en la unidad de precio del metal; MAPE está en porcentaje. '
            'La descomposición se muestra como análisis descriptivo y no participa en la competencia de pronósticos.')

st.divider()
st.caption('Proyecto académico. Los pronósticos son estimaciones estadísticas, no recomendaciones financieras personalizadas.')
