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
    '⚖️ Riesgos y simulador',
    '🎯 Nuestra recomendación',
    '🎓 Análisis académico'
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


@st.cache_data(show_spinner=False)
def comparacion_general(df):
    registros=[]
    for m in COLUMNAS:
        sm=serie_metal(df,m)
        rr=resultados_metal(sm)
        if rr is None:
            continue
        vol=float(sm.pct_change(fill_method=None).tail(60).std()*100)
        registros.append({'Metal':m,'Último precio':float(sm.iloc[-1]),
            'Unidad':UNIDADES[m], 'Modelo':rr['modelo'],
            'Error a 1 mes (%)':rr['error'],
            '3 meses (%)':rr['variaciones'][3],
            '6 meses (%)':rr['variaciones'][6],
            '12 meses (%)':rr['variaciones'][12],
            'Fluctuación mensual (%)':vol})
    return pd.DataFrame(registros)


def mostrar_barras(tabla, plazo):
    col=f'{plazo} meses (%)'
    orden=tabla.sort_values(col)
    fig=go.Figure(go.Bar(x=orden[col],y=orden['Metal'],orientation='h',
        text=[f'{v:+.2f}%' for v in orden[col]],textposition='outside',
        marker_color=['#1a9b7a' if v>=0 else '#d76767' for v in orden[col]]))
    fig.add_vline(x=0,line_color='gray')
    fig.update_layout(title=f'Cambio estimado de precio en {plazo} meses',
        xaxis_title='Variación respecto al último precio observado (%)',
        margin=dict(l=20,r=95,t=55,b=35),height=385)
    st.plotly_chart(fig,use_container_width=True)


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
    lapso = st.radio('¿Qué periodo deseas ver?', ['Últimos 5 años','Últimos 12 meses','Todo el historial'], horizontal=True)
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
    with st.expander('📊 Comparar los pronósticos de los cuatro modelos', expanded=False):
        st.write('Cada línea representa un método diferente. Se calculan todos con la misma información histórica y el mismo horizonte. La línea destacada corresponde al método con menor RMSE en la evaluación histórica a un mes.')
        modelos_futuros = {}
        fechas_comparacion = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq='MS')
        grafico_comparacion = go.Figure()
        grafico_comparacion.add_trace(go.Scatter(
            x=s.tail(24).index, y=s.tail(24).values,
            name='Precio real', mode='lines', line=dict(color='#f0f0f0', width=3)))
        colores = {'Promedio móvil simple': '#f4a261', 'Promedio móvil doble': '#9b5de5',
                   'SES': '#00b4d8', 'Holt': '#2a9d8f'}
        for metodo in METODOS:
            try:
                valores = pronosticar(metodo, s, h, ventana)
                modelos_futuros[metodo] = valores
                es_ganador = metodo == r['modelo']
                grafico_comparacion.add_trace(go.Scatter(
                    x=[s.index[-1], *fechas_comparacion],
                    y=[float(s.iloc[-1]), *valores],
                    name=metodo + (' (menor RMSE)' if es_ganador else ''),
                    mode='lines+markers',
                    line=dict(color=colores[metodo], width=4 if es_ganador else 2,
                              dash='solid' if es_ganador else 'dash'),
                    marker=dict(size=6 if es_ganador else 4)))
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                st.caption(f'No fue posible calcular {metodo}: {exc}')
        grafico_comparacion.update_layout(
            title=f'Cuatro escenarios de precio del {metal.lower()} a {h} meses',
            xaxis_title='Mes', yaxis_title=UNIDADES[metal],
            hovermode='x unified', height=470,
            legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(grafico_comparacion, use_container_width=True)
        if modelos_futuros:
            ultimo_real = float(s.iloc[-1])
            resumen_modelos = pd.DataFrame([
                {'Método': metodo,
                 f'Precio en {h} meses ({UNIDADES[metal]})': float(valores[-1]),
                 'Cambio desde último precio real (%)': (float(valores[-1]) / ultimo_real - 1) * 100,
                 'RMSE histórico (1 mes)': float(r['tabla'].set_index('Modelo').loc[metodo, 'RMSE'])
                     if metodo in r['tabla']['Modelo'].values else np.nan}
                for metodo, valores in modelos_futuros.items()
            ])
            st.dataframe(resumen_modelos.style.format({
                f'Precio en {h} meses ({UNIDADES[metal]})': 'US$ {:,.2f}',
                'Cambio desde último precio real (%)': '{:+.2f}%',
                'RMSE histórico (1 mes)': '{:,.2f}'
            }), hide_index=True, use_container_width=True)
            st.download_button('Descargar comparación de los cuatro métodos',
                resumen_modelos.to_csv(index=False).encode('utf-8-sig'),
                f'pronosticos_{metal.lower()}_{h}m.csv', 'text/csv')
        st.caption('SES y el promedio móvil simple generan pronósticos horizontales; Holt y el promedio móvil doble pueden proyectar una tendencia. Los escenarios no son intervalos de confianza. El menor RMSE histórico a un mes no garantiza el menor error a 3, 6 o 12 meses.')
    with st.expander('¿Qué tan bien funcionaron los modelos con meses ya conocidos?'):
        st.write('Comparamos los precios reales con las predicciones hechas para cada mes utilizando solamente la información disponible hasta el mes anterior.')
        modo = st.radio('¿Qué deseas observar?',
            ['Ver solo el mejor modelo', 'Comparar los cuatro modelos'],
            horizontal=True, key=f'validacion_{metal}')
        fig_validacion = go.Figure()
        historicas = r['historicas']
        fechas_evaluadas = sorted(set().union(*(set(p.index) for p in historicas.values())))
        reales = s.reindex(fechas_evaluadas)
        fig_validacion.add_trace(go.Scatter(
            x=reales.index, y=reales.values, name='Precio real',
            mode='lines+markers', line=dict(color='#f0f0f0', width=4)))
        colores_validacion = {
            'Promedio móvil simple': '#f4a261',
            'Promedio móvil doble': '#9b5de5',
            'SES': '#00b4d8', 'Holt': '#2a9d8f'
        }
        visibles = [r['modelo']] if modo == 'Ver solo el mejor modelo' else METODOS
        for metodo in visibles:
            if metodo not in historicas:
                continue
            pred_hist = historicas[metodo]
            ganador = metodo == r['modelo']
            fig_validacion.add_trace(go.Scatter(
                x=pred_hist.index, y=pred_hist.values,
                name=metodo + (' (menor RMSE)' if ganador else ''),
                mode='lines+markers',
                line=dict(color=colores_validacion[metodo],
                          width=3 if ganador else 2, dash='dash'),
                marker=dict(size=6 if ganador else 4)))
        fig_validacion.update_layout(
            title='Predicciones históricas a un mes frente a precios reales',
            xaxis_title='Mes', yaxis_title=UNIDADES[metal],
            hovermode='x unified', height=430,
            legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(fig_validacion, use_container_width=True)
        st.caption('Cuanto más cerca está la predicción del precio real, menor fue el error de ese mes. Los cuatro métodos se evalúan en los mismos meses.')
        if modo == 'Comparar los cuatro modelos':
            st.write('**Resumen de precisión histórica (pronósticos a un mes)**')
            st.dataframe(r['tabla'].style.format({
                'MAE': '{:,.2f}', 'RMSE': '{:,.2f}', 'MAPE (%)': '{:.2f}%'
            }), hide_index=True, use_container_width=True)
            st.caption('MAE y RMSE se expresan en las unidades del precio del metal. MAPE es el error porcentual promedio. El menor RMSE identifica el modelo destacado.')
            tabla_descarga = pd.DataFrame({'Fecha': reales.index, 'Precio real': reales.values})
            for metodo in METODOS:
                if metodo in historicas:
                    tabla_descarga[metodo] = historicas[metodo].reindex(reales.index).values
            st.download_button('Descargar validación de los cuatro modelos',
                tabla_descarga.to_csv(index=False).encode('utf-8-sig'),
                f'validacion_{metal.lower()}.csv', 'text/csv')
        else:
            st.info(f"El método con menor RMSE fue {r['modelo']}, con un error porcentual promedio de {r['error']:.2f}% en predicciones a un mes.")

elif pagina == '⚖️ Riesgos y simulador':
    st.title('⚖️ ¿Cuánto podría ganar o perder si cambia el precio?')
    st.write('Explora el cambio de valor de una inversión hipotética. No incluye comisiones, impuestos, diferencias de compra y venta ni otros costos.')
    tabla=comparacion_general(datos)
    if tabla.empty:
        st.warning('No se pudieron calcular comparaciones.')
        st.stop()
    st.subheader('¿Qué metal ha tenido más altibajos?')
    f=px.bar(tabla.sort_values('Fluctuación mensual (%)'),
        x='Fluctuación mensual (%)',y='Metal',orientation='h',
        text='Fluctuación mensual (%)',
        title='Movimientos de precios durante los últimos cinco años')
    f.update_traces(texttemplate='%{text:.2f}%',textposition='outside')
    f.update_layout(height=365,xaxis_title='Fluctuación mensual típica (%)',margin=dict(r=80))
    st.plotly_chart(f,use_container_width=True)
    st.caption('Una barra más larga indica movimientos mensuales históricamente más grandes; no mide todas las formas de riesgo.')
    st.subheader('🧮 Simula una inversión')
    a,b,c=st.columns(3)
    capital=a.number_input('Capital en dólares (US$)',min_value=100.0,max_value=100000000.0,value=10000.0,step=500.0)
    elegido=b.selectbox('Metal para simular',list(tabla['Metal']))
    plazo=c.selectbox('Plazo', [3,6,12],index=1,format_func=lambda n:f'{n} meses')
    variacion=float(tabla.set_index('Metal').loc[elegido,f'{plazo} meses (%)'])
    escenario=st.radio('¿Qué escenario quieres observar?',
        ['Pronóstico del modelo','Si el precio sube 5%','Si el precio baja 5%'],horizontal=True)
    tasa={'Pronóstico del modelo':variacion,'Si el precio sube 5%':5.0,'Si el precio baja 5%':-5.0}[escenario]
    valor=capital*(1+tasa/100)
    x,y,z=st.columns(3)
    x.metric('Capital inicial',f'US$ {capital:,.2f}')
    y.metric('Valor estimado',f'US$ {valor:,.2f}')
    z.metric('Cambio estimado',f'US$ {valor-capital:+,.2f}',f'{tasa:+.2f}%',delta_color='off')
    if escenario=='Pronóstico del modelo':
        st.info(f'El modelo {tabla.set_index("Metal").loc[elegido,"Modelo"]} estima un cambio de precio de {variacion:+.2f}% para {elegido.lower()} en {plazo} meses. Es una simulación, no una rentabilidad garantizada.')
    else:
        st.info('Este es un escenario hipotético de sensibilidad, no un pronóstico estadístico.')
    st.caption('El cálculo supone que el valor de la inversión cambia en la misma proporción que el precio del metal. No considera la forma de inversión ni sus costos.')

elif pagina == '🎯 Nuestra recomendación':
    st.title('🎯 ¿Qué alternativa merece una evaluación más detallada?')
    st.write('Comparamos crecimiento proyectado y fluctuaciones históricas para apoyar una decisión, no para prometer resultados.')
    tabla=comparacion_general(datos)
    if tabla.empty:
        st.warning('No hay información suficiente para comparar.')
        st.stop()
    plazo=st.radio('Horizonte que desea evaluar', [3,6,12],index=1,horizontal=True,format_func=lambda n:f'{n} meses')
    mostrar_barras(tabla,plazo)
    st.subheader('Comparación sencilla')
    mostrar=tabla[['Metal','Modelo',f'{plazo} meses (%)','Fluctuación mensual (%)']].copy()
    mostrar.columns=['Metal','Método de pronóstico','Cambio estimado (%)','Fluctuación histórica mensual (%)']
    st.dataframe(mostrar.style.format({'Cambio estimado (%)':'{:+.2f}%',
        'Fluctuación histórica mensual (%)':'{:.2f}%'}),hide_index=True,use_container_width=True)
    orden=tabla.sort_values(f'{plazo} meses (%)',ascending=False)
    primero=orden.iloc[0]
    st.info(f'**Mayor crecimiento de precio proyectado a {plazo} meses:** {primero["Metal"]} ({primero[f"{plazo} meses (%)"]:+.2f}%). Esto es solo un criterio de comparación, no una recomendación automática de compra.')
    st.warning('Antes de invertir, evalúe el riesgo de caídas, el costo de comprar y vender, el tipo de instrumento, su plazo y la incertidumbre del pronóstico. Los errores históricos fueron medidos a un mes, no a todo el horizonte elegido.')
    with st.expander('Descargar comparación para la exposición'):
        st.download_button('Descargar CSV',tabla.to_csv(index=False).encode('utf-8-sig'),
            'comparacion_gerencial_metales.csv','text/csv')

else:
    st.title('🎓 Análisis académico: ¿cómo obtuvimos los resultados?')
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

