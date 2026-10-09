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


from pathlib import Path

st.sidebar.title('⛏️ Metales para invertir')
archivo = st.sidebar.file_uploader('Cargar otro CSV (opcional)', type=['csv'])
try:
    ruta = next((str(Path(x)) for x in ['Data.csv', 'Data(2).csv', 'Data(1).csv'] if Path(x).exists()), 'Data.csv')
    datos = cargar(archivo if archivo is not None else ruta)
except Exception as exc:
    st.error(f'No se pudo abrir el CSV: {exc}')
    st.info('Coloca Data.csv en la misma carpeta que este programa, o cárgalo desde la barra lateral.')
    st.stop()

pagina = st.sidebar.radio('¿Qué quieres conocer?', [
    '🏠 Resumen para inversionistas',
    '📈 Conoce cada metal',
    '🔮 ¿Qué puede pasar?',
    '⚖️ Compara las alternativas',
    '🎓 Cómo hicimos el análisis'
])
metal = st.sidebar.selectbox('Metal que quieres analizar', list(COLUMNAS))
ventana = 12
s = serie_metal(datos, metal)
mercado = tabla_mercado(datos)
st.sidebar.caption(f'Datos hasta {datos.index.max():%m/%Y} · Precios históricos, no cotizaciones en vivo')


def lenguaje_cambio(x):
    if x > 2:
        return 'subida'
    if x < -2:
        return 'bajada'
    return 'estabilidad aproximada'


def calcular_resumen(s):
    tabla, pred = evaluar(s, ventana)
    if tabla.empty:
        return None
    mejor = str(tabla.iloc[0]['Modelo'])
    futuro = pronosticar(mejor, s, 12, ventana)
    ultimo = float(s.iloc[-1])
    return {'modelo': mejor, 'error': float(tabla.iloc[0]['MAPE (%)']),
            'tabla': tabla, 'historicas': pred, 'futuro': futuro,
            'variaciones': {h: (float(futuro[h-1]) / ultimo - 1)*100 for h in (3,6,12)}}


@st.cache_data(show_spinner=False)
def resultados_metal(serie):
    return calcular_resumen(serie)


def figura_indice(meses):
    fig = go.Figure()
    for m in COLUMNAS:
        sm = serie_metal(datos, m).tail(meses)
        if len(sm) >= 2 and sm.iloc[0] > 0:
            fig.add_trace(go.Scatter(x=sm.index, y=100*sm/sm.iloc[0], name=m, mode='lines'))
    fig.add_hline(y=100, line_dash='dash', line_color='gray')
    fig.update_layout(title='¿Cuánto cambió el precio de cada metal?',
                      yaxis_title='Índice: todos empiezan en 100',
                      xaxis_title='Mes', hovermode='x unified', height=440,
                      margin=dict(t=55,b=35))
    return fig


if pagina == '🏠 Resumen para inversionistas':
    st.title('⛏️ ¿En qué metal podríamos invertir?')
    st.write('Conoce los precios, identifica los movimientos y compara las perspectivas de cinco metales.')
    st.info('Los datos llegan hasta agosto de 2026. No son precios en vivo ni promesas de ganancias.')
    cols = st.columns(5)
    for i, m in enumerate(COLUMNAS):
        r = mercado.loc[mercado['Metal'] == m].iloc[0]
        cols[i].metric(m, f"US$ {r['Precio (US$)']:,.2f}", f"{r['Cambio mensual (%)']:+.1f}% último mes")
        cols[i].caption(UNIDADES[m])
    st.subheader('¿Cuál ha subido más?')
    periodo = st.radio('Compara los últimos:', ['12 meses', '5 años'], horizontal=True)
    st.plotly_chart(figura_indice(12 if periodo == '12 meses' else 60), use_container_width=True)
    st.caption('Todos empiezan en 100 para comparar porcentajes. No significa que cuesten lo mismo.')
    st.subheader('Lo que deberías saber antes de elegir')
    a,b,c = st.columns(3)
    a.info('📈 **Crecimiento pasado**: cuánto cambió el precio en el periodo.')
    b.info('🌊 **Fluctuaciones**: cuánto se movió el precio de un mes a otro.')
    c.info('🔮 **Pronóstico**: escenario estimado, que puede fallar.')

elif pagina == '📈 Conoce cada metal':
    st.title(f'📈 ¿Cómo se ha comportado el {metal.lower()}?')
    precio, mes, seis, vol = indicadores(s)
    a,b,c = st.columns(3)
    a.metric('Último precio registrado', f'US$ {precio:,.2f}')
    b.metric('Cambio en el último mes', f'{mes:+.2f}%')
    c.metric('Cambio en seis meses', f'{seis:+.2f}%')
    st.caption(f'Precio en {UNIDADES[metal]}. Fecha del último registro: {s.index[-1]:%m/%Y}.')
    lapso = st.radio('¿Qué periodo deseas ver?', ['Últimos 12 meses','Últimos 5 años','Todo el historial'], horizontal=True)
    n = {'Últimos 12 meses':12,'Últimos 5 años':60,'Todo el historial':len(s)}[lapso]
    st.plotly_chart(grafico_linea(s, f'Historia del precio del {metal.lower()}', UNIDADES[metal], n), use_container_width=True)
    st.success(f'En los últimos seis meses hubo una {lenguaje_cambio(seis)} del precio ({seis:+.1f}%).')
    st.write('**¿Qué debe considerar un inversionista?** Las subidas anteriores no garantizan que el precio siga aumentando. También puede caer.')

elif pagina == '🔮 ¿Qué puede pasar?':
    st.title(f'🔮 ¿Qué podría pasar con el {metal.lower()}?')
    st.write('Estimamos cómo podría cambiar su precio usando los métodos aprendidos en clase.')
    with st.spinner('Calculando pronósticos históricos y futuros...'):
        r = resultados_metal(s)
    if r is None:
        st.warning('No hay datos suficientes para elaborar un pronóstico.')
        st.stop()
    st.success(f"El método que mejor funcionó en las pruebas históricas fue **{r['modelo']}**. En pronósticos de un mes, su error porcentual promedio fue **{r['error']:.2f}%**.")
    st.subheader('¿Cuánto podría subir o bajar?')
    cols = st.columns(3)
    for col,h in zip(cols,(3,6,12)):
        variacion = r['variaciones'][h]
        precio_fut = r['futuro'][h-1]
        col.metric(f'En {h} meses', f'{variacion:+.2f}%', f'US$ {precio_fut:,.2f} estimados', delta_color='off')
        col.caption(f'Mes estimado: {(s.index[-1]+pd.DateOffset(months=h)):%m/%Y}')
    h = st.select_slider('Muestra el escenario hasta:', options=[3,6,12], value=6, format_func=lambda x:f'{x} meses')
    fechas = pd.date_range(s.index[-1]+pd.offsets.MonthBegin(1), periods=h, freq='MS')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.tail(36).index, y=s.tail(36).values, name='Precio que ya conocemos', mode='lines', line=dict(width=3)))
    fig.add_trace(go.Scatter(x=[s.index[-1],*fechas], y=[s.iloc[-1],*r['futuro'][:h]],
                             name='Lo que estima el modelo', mode='lines+markers', line=dict(dash='dash',width=3)))
    fig.update_layout(title='Precio pasado y escenario futuro', yaxis_title=UNIDADES[metal],
                      xaxis_title='Mes', hovermode='x unified', height=440)
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"Según {r['modelo']}, el cambio estimado a {h} meses es {r['variaciones'][h]:+.2f}%. Es un escenario, no una ganancia asegurada.")
    st.warning('El error histórico mostrado corresponde a pronósticos de **un mes**, no mide directamente la precisión de los escenarios a 3, 6 o 12 meses. No se muestran intervalos de incertidumbre.')
    with st.expander('¿Qué tan bien funcionó el modelo con meses ya conocidos?'):
        pred = r['historicas'][r['modelo']]
        f = go.Figure()
        f.add_trace(go.Scatter(x=s.tail(24).index,y=s.tail(24),name='Precio real'))
        f.add_trace(go.Scatter(x=pred.index,y=pred.values,name='Precio que se había estimado',line=dict(dash='dash')))
        f.update_layout(yaxis_title=UNIDADES[metal],height=350)
        st.plotly_chart(f,use_container_width=True)
        st.caption('Cuando ambas líneas se acercan, el modelo se aproximó mejor al precio real.')

elif pagina == '⚖️ Compara las alternativas':
    st.title('⚖️ Compara los cinco metales')
    st.write('Mira sus posibles cambios de precio y las fluctuaciones que tuvieron en el pasado.')
    registros=[]
    with st.spinner('Comparando los cinco metales...'):
        for m in COLUMNAS:
            sm=serie_metal(datos,m)
            rr=resultados_metal(sm)
            if rr:
                registros.append({'Metal':m,'Cambio previsto a 3 meses (%)':rr['variaciones'][3],
                                  'Cambio previsto a 6 meses (%)':rr['variaciones'][6],
                                  'Cambio previsto a 12 meses (%)':rr['variaciones'][12],
                                  'Fluctuación histórica mensual (%)':float(sm.pct_change(fill_method=None).std()*100),
                                  'Modelo utilizado':rr['modelo'], 'Error histórico a 1 mes (%)':rr['error']})
    tabla=pd.DataFrame(registros)
    plazo=st.radio('¿Cuándo piensas comparar?', ['3 meses','6 meses','12 meses'],index=1,horizontal=True)
    columna=f'Cambio previsto a {plazo} (%)'
    orden=tabla.sort_values(columna)
    fig=go.Figure(go.Bar(x=orden[columna],y=orden['Metal'],orientation='h',text=[f'{v:+.1f}%' for v in orden[columna]],textposition='outside'))
    fig.add_vline(x=0,line_color='gray')
    fig.update_layout(title=f'Cambios de precio estimados a {plazo}',xaxis_title='Cambio proyectado (%)',height=400)
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Barras a la derecha de cero: aumento estimado; a la izquierda: disminución estimada.')
    st.subheader('¿Cuáles han tenido más altibajos?')
    f=px.bar(tabla.sort_values('Fluctuación histórica mensual (%)'),x='Fluctuación histórica mensual (%)',y='Metal',orientation='h',
             title='Variación de los cambios mensuales: mayor barra = más fluctuaciones')
    f.update_layout(height=390)
    st.plotly_chart(f,use_container_width=True)
    st.info('Una subida proyectada alta puede venir acompañada de fuertes fluctuaciones. Esta comparación no incluye comisiones, impuestos ni todos los riesgos de inversión.')
    with st.expander('Ver cifras y descargar comparación'):
        st.dataframe(tabla.style.format({c:'{:+.2f}%' for c in tabla.columns if '(%)' in c}),hide_index=True,use_container_width=True)
        st.download_button('Descargar comparación CSV',tabla.to_csv(index=False).encode('utf-8-sig'),'comparacion_metales.csv','text/csv')

else:
    st.title('🎓 ¿Cómo obtuvimos estos resultados?')
    st.write('Esta sección es para explicar los procedimientos estudiados en clase. No es necesario mostrarla al público general.')
    st.subheader(f'Descomposición clásica del {metal.lower()}')
    tipo=st.radio('Tipo de descomposición',['Aditiva','Multiplicativa'],horizontal=True)
    if len(s)>=24 and (tipo=='Aditiva' or (s>0).all()):
        des=seasonal_decompose(s,model='additive' if tipo=='Aditiva' else 'multiplicative',period=12,extrapolate_trend='freq')
        componentes={'Precio observado':des.observed,'Tendencia':des.trend,'Estacionalidad':des.seasonal,'Residuo':des.resid}
        componente=st.selectbox('Componente a visualizar',list(componentes))
        unidad=UNIDADES[metal] if componente in ['Precio observado','Tendencia'] or tipo=='Aditiva' else 'Factor (sin unidad)'
        st.plotly_chart(grafico_linea(componentes[componente],f'{tipo}: {componente}',unidad,len(s)),use_container_width=True)
        st.latex(r'Y_t=T_t+E_t+R_t' if tipo=='Aditiva' else r'Y_t=T_t\times E_t\times R_t')
        st.download_button('Descargar componentes',pd.DataFrame(componentes).to_csv().encode('utf-8-sig'),'componentes.csv','text/csv')
    else:
        st.warning('No hay suficientes meses o existen valores no positivos para esta descomposición.')
    st.subheader('Promedios móviles')
    pm1=s.rolling(ventana).mean()
    pm2=pm1.rolling(ventana).mean()
    f=go.Figure()
    for nombre,serie in [('Precio real',s),('Promedio móvil simple',pm1),('Promedio móvil doble',pm2)]:
        f.add_trace(go.Scatter(x=serie.tail(60).index,y=serie.tail(60).values,name=nombre))
    f.update_layout(yaxis_title=UNIDADES[metal],height=390)
    st.plotly_chart(f,use_container_width=True)
    st.caption('Se emplean ventanas de 12 meses. Para proyectar con el promedio móvil doble se usa la extrapolación de Brown.')
    st.subheader('Comparación de métodos estudiados')
    r=resultados_metal(s)
    if r:
        st.dataframe(r['tabla'].style.format({'MAE':'{:,.2f}','RMSE':'{:,.2f}','MAPE (%)':'{:.2f}%'}),hide_index=True,use_container_width=True)
        st.write('**MAE:** error absoluto promedio. **RMSE:** penaliza más los errores grandes. **MAPE:** error porcentual promedio.')
        st.caption('Evaluación histórica de un mes adelante, actualizando el origen de pronóstico. Se elige el menor RMSE.')
    faltantes=datos[metal].loc[datos[metal].first_valid_index():datos[metal].last_valid_index()].isna().sum()
    st.caption(f'Meses internos sin precio original: {faltantes}. Si existen, se interpolan para el análisis.')

st.divider()
