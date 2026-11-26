
import streamlit as st
import pandas as pd
import joblib

from groq import Groq

# --------------------------------
# CONFIGURACION DE LA APLICACION
# --------------------------------

st.set_page_config(
    page_title="Mantenimiento Predictivo",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Sistema Predictivo de Mantenimiento Industrial")
st.write("Monitoreo inteligente de maquinaria mediante Machine Learning e IA.")

# --------------------------------
# CARGAR MODELO ENTRENADO
# --------------------------------

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_mantenimiento.pkl")

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

# --------------------------------
# CONTROLES DE LOS SENSORES
# --------------------------------

st.sidebar.header("⚙️ Lecturas de los sensores")

temperatura = st.sidebar.slider(
    "Temperatura (°C)", 40.0, 120.0, 80.0
)

vibracion = st.sidebar.slider(
    "Vibración (mm/s)", 0.0, 12.0, 4.5
)

horas = st.sidebar.slider(
    "Horas de operación", 0, 5000, 2000
)

presion = st.sidebar.slider(
    "Presión (PSI)", 70.0, 170.0, 116.0
)

# --------------------------------
# MOSTRAR LECTURAS
# --------------------------------

st.subheader("📊 Lecturas actuales")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Temperatura", f"{temperatura} °C")
col2.metric("Vibración", f"{vibracion} mm/s")
col3.metric("Operación", f"{horas} hrs")
col4.metric("Presión", f"{presion} PSI")

# --------------------------------
# PREDICCION DEL MODELO
# --------------------------------

datos_sensor = pd.DataFrame([{
    "temperatura_c": temperatura,
    "vibracion_mms": vibracion,
    "horas_operacion": horas,
    "presion_psi": presion
}])

prediccion = int(modelo.predict(datos_sensor)[0])

probabilidades = modelo.predict_proba(datos_sensor)[0]
probabilidad = float(probabilidades.max())

# --------------------------------
# RESULTADO DEL DIAGNOSTICO
# --------------------------------

st.subheader("🔎 Diagnóstico de la maquinaria")

nombres_estados = {
    0: "Operación Normal",
    1: "Mantenimiento Preventivo",
    2: "Fallo Crítico"
}

st.metric(
    "Probabilidad de la clasificación",
    f"{probabilidad * 100:.2f}%"
)

if prediccion == 0:

    st.success("✅ Estado: Operación Normal")

elif prediccion == 1:

    st.warning("⚠️ Estado: Mantenimiento Preventivo")

else:

    st.error("🚨 Estado: Fallo Crítico")

st.write("Clasificación:", nombres_estados[prediccion])

# --------------------------------
# INTEGRACION CON GROQ
# --------------------------------

def generar_orden(estado, temperatura, vibracion, horas, presion):

    try:
        api_key = st.secrets["GROQ_API_KEY"]

        client = Groq(api_key=api_key)

        # Detectar modelos disponibles
        modelos_activos = [
            m.id for m in client.models.list().data
            if not any(
                x in m.id.lower()
                for x in ["whisper", "guard", "tts", "playai", "embed"]
            )
        ]

        if not modelos_activos:
            raise Exception("No hay modelos de chat disponibles.")

        modelo_groq = next(
            (m for m in modelos_activos if "llama" in m.lower()),
            modelos_activos[0]
        )

        # Instrucciones para la IA
        instrucciones = """
        Eres un Jefe de Mantenimiento Industrial.

        Analiza las lecturas de los sensores y genera un
        diagnóstico técnico.

        Si el estado es 1 o 2, genera una Orden de Trabajo
        en formato Markdown que contenga:

        - Folio
        - Diagnóstico
        - Tres acciones técnicas
        - Equipo de Protección Personal (EPP)

        Si el estado es 2, indica que se debe detener la
        maquinaria y aplicar el protocolo de seguridad.

        No inventes mediciones adicionales.
        """

        respuesta = client.chat.completions.create(
            model=modelo_groq,
            messages=[
                {
                    "role": "system",
                    "content": instrucciones
                },
                {
                    "role": "user",
                    "content": f"""
                    Estado: {estado}
                    Temperatura: {temperatura} °C
                    Vibración: {vibracion} mm/s
                    Horas: {horas}
                    Presión: {presion} PSI

                    Genera el diagnóstico correspondiente.
                    """
                }
            ],
            temperature=0.3
        )

        return respuesta.choices[0].message.content

    except Exception as e:
        st.error(f"Error al conectar con Groq: {e}")
        return None

# --------------------------------
# GENERAR ORDEN DE TRABAJO
# --------------------------------

st.subheader("🤖 Orden de trabajo técnico")

if prediccion == 0:

    st.success(
        "La maquinaria presenta condiciones normales. "
        "Se recomienda continuar con el monitoreo preventivo."
    )

else:

    if st.button("Generar Orden de Trabajo con IA"):

        with st.spinner("Generando diagnóstico con Groq..."):

            resultado = generar_orden(
                prediccion,
                temperatura,
                vibracion,
                horas,
                presion
            )

        if resultado:
            st.markdown(resultado)
