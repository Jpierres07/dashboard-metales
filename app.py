import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore', category=RuntimeWarning)
st.set_page_config(page_title='Observatorio de metales', page_icon='â›ï¸', layout='wide')

COLUMNAS = {
    'Cobre': 'Cobre (US$/TM)', 'Oro': 'Oro (US$/onza troy)',
    'Plata': 'Plata (US$/onza troy)', 'Zinc': 'Zinc (US$/TM)',
    'Plomo': 'Plomo (US$/TM)'
}
UNIDADES = {m: ('US$/TM' if m in ['Cobre', 'Zinc', 'Plomo'] else 'US$/onza troy') for m in COLUMNAS}
METODOS = ['Promedio mÃ³vil simple', 'Promedio mÃ³vil doble', 'SES', 'Holt']

@st.cache_data
def cargar(archivo):
    df = pd.read_csv(archivo, sep=';', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    # Normalizar el encabezado de fecha, si fuese necesario.
    primera = df.columns[0]
    df = df.rename(columns={primera: 'AÃ±oMes'})
    faltan = [x for x in ['AÃ±oMes', *COLUMNAS.values()] if x not in df.columns]
    if faltan:
        raise ValueError(f'Faltan columnas: {faltan}. Disponibles: {list(df.columns)}')
    p = df['AÃ±oMes'].astype(str).str.strip().str.upper().str.extract(r'^(\d{4})M(0?[1-9]|1[0-2])$')
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
    if len(s) < max(5, ventana * 2 if nombre == 'Promedio mÃ³vil doble' else ventana):
        raise ValueError('Datos insuficientes para este modelo')
    if nombre == 'Promedio mÃ³vil simple':
        pred = np.repeat(s.iloc[-ventana:].mean(), horizonte)
    elif nombre == 'Promedio mÃ³vil doble':
        # MÃ©todo de Brown: promedio mÃ³vil de orden n aplicado dos veces.
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
        raise ValueError('PronÃ³stico no finito')
    return pred


def evaluar(s, ventana=12, meses=12):
    # EvaluaciÃ³n con origen mÃ³vil: cada predicciÃ³n usa Ãºnicamente meses anteriores.
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

st.sidebar.title('â›ï¸ Metales para invertir')
archivo = st.sidebar.file_uploader('Cargar otro CSV (opcional)', type=['csv'])
try:
    ruta = next((str(Path(x)) for x in ['Data.csv', 'Data(2).csv', 'Data(1).csv'] if Path(x).exists()), 'Data.csv')
    datos = cargar(archivo if archivo is not None else ruta)
except Exception as exc:
    st.error(f'No se pudo abrir el CSV: {exc}')
    st.info('Coloca Data.csv en la misma carpeta que este programa, o cÃ¡rgalo desde la barra lateral.')
    st.stop()

pagina = st.sidebar.radio('Â¿QuÃ© quieres conocer?', [
    'ðŸ  Resumen para inversionistas',
    'ðŸ“ˆ Conoce cada metal',
    'ðŸ”® Â¿QuÃ© puede pasar?',
    'âš–ï¸ Riesgos y simulador',
    'ðŸŽ¯ Nuestra recomendaciÃ³n',
    'ðŸŽ“ AnÃ¡lisis acadÃ©mico'
])
metal = st.sidebar.selectbox('Metal que quieres analizar', list(COLUMNAS))
ventana = 12
s = serie_metal(datos, metal)
mercado = tabla_mercado(datos)
st.sidebar.caption(f'Datos hasta {datos.index.max():%m/%Y} Â· Precios histÃ³ricos, no cotizaciones en vivo')


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
    fig.update_layout(title='Â¿CuÃ¡nto cambiÃ³ el precio de cada metal?',
                      yaxis_title='Ãndice: todos empiezan en 100',
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
        registros.append({'Metal':m,'Ãšltimo precio':float(sm.iloc[-1]),
            'Unidad':UNIDADES[m], 'Modelo':rr['modelo'],
            'Error a 1 mes (%)':rr['error'],
            '3 meses (%)':rr['variaciones'][3],
            '6 meses (%)':rr['variaciones'][6],
            '12 meses (%)':rr['variaciones'][12],
            'FluctuaciÃ³n mensual (%)':vol})
    return pd.DataFrame(registros)


def mostrar_barras(tabla, plazo):
    col=f'{plazo} meses (%)'
    orden=tabla.sort_values(col)
    fig=go.Figure(go.Bar(x=orden[col],y=orden['Metal'],orientation='h',
        text=[f'{v:+.2f}%' for v in orden[col]],textposition='outside',
        marker_color=['#1a9b7a' if v>=0 else '#d76767' for v in orden[col]]))
    fig.add_vline(x=0,line_color='gray')
    fig.update_layout(title=f'Cambio estimado de precio en {plazo} meses',
        xaxis_title='VariaciÃ³n respecto al Ãºltimo precio observado (%)',
        margin=dict(l=20,r=95,t=55,b=35),height=385)
    st.plotly_chart(fig,use_container_width=True)


if pagina == 'ðŸ  Resumen para inversionistas':
    st.title('â›ï¸ Â¿En quÃ© metal podrÃ­amos invertir?')
    st.write('Conoce los precios, identifica los movimientos y compara las perspectivas de cinco metales.')
    st.info('Los datos llegan hasta agosto de 2026. No son precios en vivo ni promesas de ganancias.')
    cols = st.columns(5)
    for i, m in enumerate(COLUMNAS):
        r = mercado.loc[mercado['Metal'] == m].iloc[0]
        cols[i].metric(m, f"US$ {r['Precio (US$)']:,.2f}", f"{r['Cambio mensual (%)']:+.1f}% Ãºltimo mes")
        cols[i].caption(UNIDADES[m])
    st.subheader('Â¿CuÃ¡l ha subido mÃ¡s?')
    periodo = st.radio('Compara los Ãºltimos:', ['12 meses', '5 aÃ±os'], horizontal=True)
    st.plotly_chart(figura_indice(12 if periodo == '12 meses' else 60), use_container_width=True)
    st.caption('Todos empiezan en 100 para comparar porcentajes. No significa que cuesten lo mismo.')
    st.subheader('Lo que deberÃ­as saber antes de elegir')
    a,b,c = st.columns(3)
    a.info('ðŸ“ˆ **Crecimiento pasado**: cuÃ¡nto cambiÃ³ el precio en el periodo.')
    b.info('ðŸŒŠ **Fluctuaciones**: cuÃ¡nto se moviÃ³ el precio de un mes a otro.')
    c.info('ðŸ”® **PronÃ³stico**: escenario estimado, que puede fallar.')

elif pagina == 'ðŸ“ˆ Conoce cada metal':
    st.title(f'ðŸ“ˆ Â¿CÃ³mo se ha comportado el {metal.lower()}?')
    precio, mes, seis, vol = indicadores(s)
    a,b,c = st.columns(3)
    a.metric('Ãšltimo precio registrado', f'US$ {precio:,.2f}')
    b.metric('Cambio en el Ãºltimo mes', f'{mes:+.2f}%')
    c.metric('Cambio en seis meses', f'{seis:+.2f}%')
    st.caption(f'Precio en {UNIDADES[metal]}. Fecha del Ãºltimo registro: {s.index[-1]:%m/%Y}.')
    lapso = st.radio('Â¿QuÃ© periodo deseas ver?', ['Ãšltimos 5 aÃ±os','Ãšltimos 12 meses','Todo el historial'], horizontal=True)
    n = {'Ãšltimos 12 meses':12,'Ãšltimos 5 aÃ±os':60,'Todo el historial':len(s)}[lapso]
    st.plotly_chart(grafico_linea(s, f'Historia del precio del {metal.lower()}', UNIDADES[metal], n), use_container_width=True)
    st.success(f'En los Ãºltimos seis meses hubo una {lenguaje_cambio(seis)} del precio ({seis:+.1f}%).')
    st.write('**Â¿QuÃ© debe considerar un inversionista?** Las subidas anteriores no garantizan que el precio siga aumentando. TambiÃ©n puede caer.')

elif pagina == 'ðŸ”® Â¿QuÃ© puede pasar?':
    st.title(f'ðŸ”® Â¿QuÃ© podrÃ­a pasar con el {metal.lower()}?')
    st.write('Estimamos cÃ³mo podrÃ­a cambiar su precio usando los mÃ©todos aprendidos en clase.')
    with st.spinner('Calculando pronÃ³sticos histÃ³ricos y futuros...'):
        r = resultados_metal(s)
    if r is None:
        st.warning('No hay datos suficientes para elaborar un pronÃ³stico.')
        st.stop()
    st.success(f"El mÃ©todo que mejor funcionÃ³ en las pruebas histÃ³ricas fue **{r['modelo']}**. En pronÃ³sticos de un mes, su error porcentual promedio fue **{r['error']:.2f}%**.")
    st.subheader('Â¿CuÃ¡nto podrÃ­a subir o bajar?')
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
    st.info(f"SegÃºn {r['modelo']}, el cambio estimado a {h} meses es {r['variaciones'][h]:+.2f}%. Es un escenario, no una ganancia asegurada.")
    st.warning('El error histÃ³rico mostrado corresponde a pronÃ³sticos de **un mes**, no mide directamente la precisiÃ³n de los escenarios a 3, 6 o 12 meses. No se muestran intervalos de incertidumbre.')
    with st.expander('ðŸ“Š Comparar los pronÃ³sticos de los cuatro modelos', expanded=False):
        st.write('Cada lÃ­nea representa un mÃ©todo diferente. Se calculan todos con la misma informaciÃ³n histÃ³rica y el mismo horizonte. La lÃ­nea destacada corresponde al mÃ©todo con menor RMSE en la evaluaciÃ³n histÃ³rica a un mes.')
        modelos_futuros = {}
        fechas_comparacion = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq='MS')
        grafico_comparacion = go.Figure()
        grafico_comparacion.add_trace(go.Scatter(
            x=s.tail(24).index, y=s.tail(24).values,
            name='Precio real', mode='lines', line=dict(color='#f0f0f0', width=3)))
        colores = {'Promedio mÃ³vil simple': '#f4a261', 'Promedio mÃ³vil doble': '#9b5de5',
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
                {'MÃ©todo': metodo,
                 f'Precio en {h} meses ({UNIDADES[metal]})': float(valores[-1]),
                 'Cambio desde Ãºltimo precio real (%)': (float(valores[-1]) / ultimo_real - 1) * 100,
                 'RMSE histÃ³rico (1 mes)': float(r['tabla'].set_index('Modelo').loc[metodo, 'RMSE'])
                     if metodo in r['tabla']['Modelo'].values else np.nan}
                for metodo, valores in modelos_futuros.items()
            ])
            st.dataframe(resumen_modelos.style.format({
                f'Precio en {h} meses ({UNIDADES[metal]})': 'US$ {:,.2f}',
                'Cambio desde Ãºltimo precio real (%)': '{:+.2f}%',
                'RMSE histÃ³rico (1 mes)': '{:,.2f}'
            }), hide_index=True, use_container_width=True)
            st.download_button('Descargar comparaciÃ³n de los cuatro mÃ©todos',
                resumen_modelos.to_csv(index=False).encode('utf-8-sig'),
                f'pronosticos_{metal.lower()}_{h}m.csv', 'text/csv')
        st.caption('SES y el promedio mÃ³vil simple generan pronÃ³sticos horizontales; Holt y el promedio mÃ³vil doble pueden proyectar una tendencia. Los escenarios no son intervalos de confianza. El menor RMSE histÃ³rico a un mes no garantiza el menor error a 3, 6 o 12 meses.')
    with st.expander('Â¿QuÃ© tan bien funcionaron los modelos con meses ya conocidos?'):
        st.write('Comparamos los precios reales con las predicciones hechas para cada mes utilizando solamente la informaciÃ³n disponible hasta el mes anterior.')
        modo = st.radio('Â¿QuÃ© deseas observar?',
            ['Precio real + un modelo', 'Comparar los cuatro modelos'],
            horizontal=True, key=f'validacion_{metal}')
        modelo_elegido = None
        if modo == 'Precio real + un modelo':
            modelo_elegido = st.selectbox(
                'Selecciona el mÃ©todo para compararlo con el precio real',
                METODOS,
                index=METODOS.index(r['modelo']),
                key=f'modelo_validacion_{metal}'
            )
        fig_validacion = go.Figure()
        historicas = r['historicas']
        fechas_evaluadas = sorted(set().union(*(set(p.index) for p in historicas.values())))
        reales = s.reindex(fechas_evaluadas)
        fig_validacion.add_trace(go.Scatter(
            x=reales.index, y=reales.values, name='Precio real',
            mode='lines+markers', line=dict(color='#f0f0f0', width=4)))
        colores_validacion = {
            'Promedio mÃ³vil simple': '#f4a261',
            'Promedio mÃ³vil doble': '#9b5de5',
            'SES': '#00b4d8', 'Holt': '#2a9d8f'
        }
        visibles = [modelo_elegido] if modelo_elegido else METODOS
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
            title='Predicciones histÃ³ricas a un mes frente a precios reales',
            xaxis_title='Mes', yaxis_title=UNIDADES[metal],
            hovermode='x unified', height=430,
            legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(fig_validacion, use_container_width=True)
        st.caption('Cuanto mÃ¡s cerca estÃ¡ la predicciÃ³n del precio real, menor fue el error de ese mes. Los cuatro mÃ©todos se evalÃºan en los mismos meses.')
        if modo == 'Comparar los cuatro modelos':
            st.write('**Resumen de precisiÃ³n histÃ³rica (pronÃ³sticos a un mes)**')
            st.dataframe(r['tabla'].style.format({
                'MAE': '{:,.2f}', 'RMSE': '{:,.2f}', 'MAPE (%)': '{:.2f}%'
            }), hide_index=True, use_container_width=True)
            st.caption('MAE y RMSE se expresan en las unidades del precio del metal. MAPE es el error porcentual promedio. El menor RMSE identifica el modelo destacado.')
            tabla_descarga = pd.DataFrame({'Fecha': reales.index, 'Precio real': reales.values})
            for metodo in METODOS:
                if metodo in historicas:
                    tabla_descarga[metodo] = historicas[metodo].reindex(reales.index).values
            st.download_button('Descargar validaciÃ³n de los cuatro modelos',
                tabla_descarga.to_csv(index=False).encode('utf-8-sig'),
                f'validacion_{metal.lower()}.csv', 'text/csv')
        else:
            st.info(f"EstÃ¡s comparando el precio real con {modelo_elegido}. El mÃ©todo con menor RMSE histÃ³rico fue {r['modelo']}, con un MAPE de {r['error']:.2f}% en predicciones a un mes.")

elif pagina == 'âš–ï¸ Riesgos y simulador':
    st.title('âš–ï¸ Â¿CuÃ¡nto podrÃ­a ganar o perder si cambia el precio?')
    st.write('Explora el cambio de valor de una inversiÃ³n hipotÃ©tica. No incluye comisiones, impuestos, diferencias de compra y venta ni otros costos.')
    tabla=comparacion_general(datos)
    if tabla.empty:
        st.warning('No se pudieron calcular comparaciones.')
        st.stop()
    st.subheader('Â¿QuÃ© metal ha tenido mÃ¡s altibajos?')
    f=px.bar(tabla.sort_values('FluctuaciÃ³n mensual (%)'),
        x='FluctuaciÃ³n mensual (%)',y='Metal',orientation='h',
        text='FluctuaciÃ³n mensual (%)',
        title='Movimientos de precios durante los Ãºltimos cinco aÃ±os')
    f.update_traces(texttemplate='%{text:.2f}%',textposition='outside')
    f.update_layout(height=365,xaxis_title='FluctuaciÃ³n mensual tÃ­pica (%)',margin=dict(r=80))
    st.plotly_chart(f,use_container_width=True)
    st.caption('Una barra mÃ¡s larga indica movimientos mensuales histÃ³ricamente mÃ¡s grandes; no mide todas las formas de riesgo.')
    st.subheader('ðŸ“… Â¿QuÃ© meses han sido mÃ¡s favorables histÃ³ricamente?')
    st.write('Comparamos los cambios porcentuales mensuales histÃ³ricos; esto no predice ganancias futuras.')
    e1, e2 = st.columns(2)
    metal_mes = e1.selectbox('Metal para analizar', list(COLUMNAS), index=list(COLUMNAS).index(metal), key='mes_metal')
    anos_mes = e2.selectbox('Periodo histÃ³rico', [5, 10, 15, 20], index=1, format_func=lambda n: f'Ãšltimos {n} aÃ±os')
    serie_mes = serie_metal(datos, metal_mes).dropna()
    corte_mes = serie_mes.index.max() - pd.DateOffset(years=anos_mes)
    serie_mes = serie_mes.loc[serie_mes.index >= corte_mes]
    cambios_mes = serie_mes.pct_change(fill_method=None).mul(100).replace([np.inf, -np.inf], np.nan).dropna()
    if len(cambios_mes) >= 24:
        df_mes = pd.DataFrame({'Mes_num': cambios_mes.index.month, 'Cambio': cambios_mes.values})
        resumen_mes = df_mes.groupby('Mes_num')['Cambio'].agg(Media='mean', Mediana='median', Anos='count')
        resumen_mes['Frecuencia_subida'] = df_mes.groupby('Mes_num')['Cambio'].apply(lambda x: (x > 0).mean() * 100)
        resumen_mes = resumen_mes.reindex(range(1, 13))
        resumen_mes['Mes'] = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        graf_mes = px.bar(resumen_mes, x='Mes', y='Media', color='Media',
            color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
            title=f'VariaciÃ³n mensual promedio del {metal_mes.lower()}')
        graf_mes.add_hline(y=0, line_color='gray')
        graf_mes.update_layout(height=380, coloraxis_showscale=False,
            yaxis_title='Cambio del precio respecto al mes anterior (%)')
        st.plotly_chart(graf_mes, use_container_width=True)
        st.caption('Cada barra muestra el cambio promedio del precio desde el mes anterior. No representa una ganancia garantizada.')
        st.dataframe(resumen_mes[['Mes', 'Media', 'Mediana', 'Frecuencia_subida', 'Anos']].rename(columns={
            'Media': 'Cambio medio (%)', 'Mediana': 'Cambio mediano (%)',
            'Frecuencia_subida': 'AÃ±os con subida (%)', 'Anos': 'Observaciones'
        }).style.format({'Cambio medio (%)': '{:+.2f}%', 'Cambio mediano (%)': '{:+.2f}%',
                         'AÃ±os con subida (%)': '{:.0f}%'}), hide_index=True, use_container_width=True)
        vigilar_mes = resumen_mes[(resumen_mes['Anos'] >= 5) & (resumen_mes['Media'] > 0) &
            (resumen_mes['Mediana'] > 0) & (resumen_mes['Frecuencia_subida'] > 50)]
        if len(vigilar_mes):
            st.info('Meses que merecen seguimiento segÃºn los datos histÃ³ricos: **' +
                ', '.join(vigilar_mes['Mes']) + '**. Tuvieron media y mediana positivas y subieron en mÃ¡s de la mitad de los aÃ±os observados.')
        else:
            st.info('No se encontraron meses con seÃ±ales histÃ³ricas positivas consistentes bajo estos criterios.')
        st.warning('Un patrÃ³n histÃ³rico no indica cuÃ¡ndo comprar con seguridad. Hay que considerar precio actual, costos, riesgo y contexto econÃ³mico; estos patrones no se han validado como estrategia de inversiÃ³n.')
    else:
        st.info('No hay datos suficientes para analizar patrones mensuales.')
    st.subheader('ðŸ§® Simula una inversiÃ³n')
    a,b,c=st.columns(3)
    capital=a.number_input('Capital en dÃ³lares (US$)',min_value=100.0,max_value=100000000.0,value=10000.0,step=500.0)
    elegido=b.selectbox('Metal para simular',list(tabla['Metal']))
    plazo=c.selectbox('Plazo', [3,6,12],index=1,format_func=lambda n:f'{n} meses')
    variacion=float(tabla.set_index('Metal').loc[elegido,f'{plazo} meses (%)'])
    escenario=st.radio('Â¿QuÃ© escenario quieres observar?',
        ['PronÃ³stico del modelo','Si el precio sube 5%','Si el precio baja 5%'],horizontal=True)
    tasa={'PronÃ³stico del modelo':variacion,'Si el precio sube 5%':5.0,'Si el precio baja 5%':-5.0}[escenario]
    valor=capital*(1+tasa/100)
    x,y,z=st.columns(3)
    x.metric('Capital inicial',f'US$ {capital:,.2f}')
    y.metric('Valor estimado',f'US$ {valor:,.2f}')
    z.metric('Cambio estimado',f'US$ {valor-capital:+,.2f}',f'{tasa:+.2f}%',delta_color='off')
    if escenario=='PronÃ³stico del modelo':
        st.info(f'El modelo {tabla.set_index("Metal").loc[elegido,"Modelo"]} estima un cambio de precio de {variacion:+.2f}% para {elegido.lower()} en {plazo} meses. Es una simulaciÃ³n, no una rentabilidad garantizada.')
    else:
        st.info('Este es un escenario hipotÃ©tico de sensibilidad, no un pronÃ³stico estadÃ­stico.')
    st.caption('El cÃ¡lculo supone que el valor de la inversiÃ³n cambia en la misma proporciÃ³n que el precio del metal. No considera la forma de inversiÃ³n ni sus costos.')

elif pagina == 'ðŸŽ¯ Nuestra recomendaciÃ³n':
    st.title('ðŸŽ¯ Â¿QuÃ© alternativa merece una evaluaciÃ³n mÃ¡s detallada?')
    st.write('Comparamos crecimiento proyectado y fluctuaciones histÃ³ricas para apoyar una decisiÃ³n, no para prometer resultados.')
    tabla=comparacion_general(datos)
    if tabla.empty:
        st.warning('No hay informaciÃ³n suficiente para comparar.')
        st.stop()
    plazo=st.radio('Horizonte que desea evaluar', [3,6,12],index=1,horizontal=True,format_func=lambda n:f'{n} meses')
    mostrar_barras(tabla,plazo)
    st.subheader('ComparaciÃ³n sencilla')
    mostrar=tabla[['Metal','Modelo',f'{plazo} meses (%)','FluctuaciÃ³n mensual (%)']].copy()
    mostrar.columns=['Metal','MÃ©todo de pronÃ³stico','Cambio estimado (%)','FluctuaciÃ³n histÃ³rica mensual (%)']
    st.dataframe(mostrar.style.format({'Cambio estimado (%)':'{:+.2f}%',
        'FluctuaciÃ³n histÃ³rica mensual (%)':'{:.2f}%'}),hide_index=True,use_container_width=True)
    orden=tabla.sort_values(f'{plazo} meses (%)',ascending=False)
    primero=orden.iloc[0]
    st.info(f'**Mayor crecimiento de precio proyectado a {plazo} meses:** {primero["Metal"]} ({primero[f"{plazo} meses (%)"]:+.2f}%). Esto es solo un criterio de comparaciÃ³n, no una recomendaciÃ³n automÃ¡tica de compra.')
    st.warning('Antes de invertir, evalÃºe el riesgo de caÃ­das, el costo de comprar y vender, el tipo de instrumento, su plazo y la incertidumbre del pronÃ³stico. Los errores histÃ³ricos fueron medidos a un mes, no a todo el horizonte elegido.')
    with st.expander('Descargar comparaciÃ³n para la exposiciÃ³n'):
        st.download_button('Descargar CSV',tabla.to_csv(index=False).encode('utf-8-sig'),
            'comparacion_gerencial_metales.csv','text/csv')

else:
    st.title('ðŸŽ“ AnÃ¡lisis acadÃ©mico: Â¿cÃ³mo obtuvimos los resultados?')
    st.write('Esta secciÃ³n es para explicar los procedimientos estudiados en clase. No es necesario mostrarla al pÃºblico general.')
    st.subheader(f'DescomposiciÃ³n clÃ¡sica del {metal.lower()}')
    tipo=st.radio('Tipo de descomposiciÃ³n',['Aditiva','Multiplicativa'],horizontal=True)
    if len(s)>=24 and (tipo=='Aditiva' or (s>0).all()):
        des=seasonal_decompose(s,model='additive' if tipo=='Aditiva' else 'multiplicative',period=12,extrapolate_trend='freq')
        componentes={'Precio observado':des.observed,'Tendencia':des.trend,'Estacionalidad':des.seasonal,'Residuo':des.resid}
        componente=st.selectbox('Componente a visualizar',list(componentes))
        unidad=UNIDADES[metal] if componente in ['Precio observado','Tendencia'] or tipo=='Aditiva' else 'Factor (sin unidad)'
        st.plotly_chart(grafico_linea(componentes[componente],f'{tipo}: {componente}',unidad,len(s)),use_container_width=True)
        st.latex(r'Y_t=T_t+E_t+R_t' if tipo=='Aditiva' else r'Y_t=T_t\times E_t\times R_t')
        st.download_button('Descargar componentes',pd.DataFrame(componentes).to_csv().encode('utf-8-sig'),'componentes.csv','text/csv')
    else:
        st.warning('No hay suficientes meses o existen valores no positivos para esta descomposiciÃ³n.')
    st.subheader('Promedios mÃ³viles')
    pm1=s.rolling(ventana).mean()
    pm2=pm1.rolling(ventana).mean()
    f=go.Figure()
    for nombre,serie in [('Precio real',s),('Promedio mÃ³vil simple',pm1),('Promedio mÃ³vil doble',pm2)]:
        f.add_trace(go.Scatter(x=serie.tail(60).index,y=serie.tail(60).values,name=nombre))
    f.update_layout(yaxis_title=UNIDADES[metal],height=390)
    st.plotly_chart(f,use_container_width=True)
    st.caption('Se emplean ventanas de 12 meses. Para proyectar con el promedio mÃ³vil doble se usa la extrapolaciÃ³n de Brown.')
    st.subheader('ComparaciÃ³n de mÃ©todos estudiados')
    r=resultados_metal(s)
    if r:
        st.dataframe(r['tabla'].style.format({'MAE':'{:,.2f}','RMSE':'{:,.2f}','MAPE (%)':'{:.2f}%'}),hide_index=True,use_container_width=True)
        st.write('**MAE:** error absoluto promedio. **RMSE:** penaliza mÃ¡s los errores grandes. **MAPE:** error porcentual promedio.')
        st.caption('EvaluaciÃ³n histÃ³rica de un mes adelante, actualizando el origen de pronÃ³stico. Se elige el menor RMSE.')
    faltantes=datos[metal].loc[datos[metal].first_valid_index():datos[metal].last_valid_index()].isna().sum()
    st.caption(f'Meses internos sin precio original: {faltantes}. Si existen, se interpolan para el anÃ¡lisis.')
