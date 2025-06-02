import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from io import StringIO

# Configura la pagina Streamlit
st.set_page_config(
    page_title="Dashboard Previsioni Avanzate con Prophet",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Forecast Interattivo con Prophet e Regressori Multipli")

st.markdown("""
Questa app permette di caricare un file CSV con le seguenti colonne:
- **ds**: data (formato YYYY-MM-DD)
- **y_vendite**: valore storico delle vendite
- **y_clienti**: numero storico di clienti
- **meteo_temp**: temperatura (regressore continuo)
- **ads_tiktok**: spesa/engagement di advertising su TikTok (regressore continuo)
- **festivo**: variabile binaria (1 se giorno festivo, 0 altrimenti)

L'utente può selezionare quale colonna target utilizzare per la previsione (vendite o clienti),
quali regressori includere, l'intervallo di date per il training e l'orizzonte di forecast.
""")

# -------------------------------------------------------------
# 1) Caricamento del file CSV
# -------------------------------------------------------------
uploaded_file = st.file_uploader(
    label="Carica un file CSV con le colonne: ds, y_vendite, y_clienti, meteo_temp, ads_tiktok, festivo",
    type=["csv"],
    help="Il file deve contenere almeno le colonne elencate sopra."
)

if uploaded_file is not None:
    # Legge il CSV in un DataFrame Pandas
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["ds"])
    except Exception as e:
        st.error(f"Errore durante la lettura del CSV: {e}")
        st.stop()

    # Controlla che tutte le colonne richieste siano presenti
    required_cols = ["ds", "y_vendite", "y_clienti", "meteo_temp", "ads_tiktok", "festivo"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        st.error(f"Il file CSV è mancante delle colonne: {', '.join(missing_cols)}")
        st.stop()

    st.success("File CSV caricato correttamente!")
    st.write("Anteprima dei primi 5 record:", df.head())

    # -------------------------------------------------------------
    # 2) Selezione del Target (y_vendite o y_clienti)
    # -------------------------------------------------------------
    st.sidebar.header("Parametri di configurazione")
    target = st.sidebar.radio(
        "Seleziona il target per la previsione:",
        ("y_vendite", "y_clienti"),
        index=0,
        help="Scegli se vuoi prevedere le vendite o il numero di clienti"
    )
    st.sidebar.write(f"Target selezionato: **{target}**")

    # -------------------------------------------------------------
    # 3) Intervallo di date per il training
    # -------------------------------------------------------------
    min_date = df["ds"].min().date()
    max_date = df["ds"].max().date()
    start_date, end_date = st.sidebar.date_input(
        "Seleziona l'intervallo di date per il training:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if start_date > end_date:
        st.sidebar.error("La data di inizio deve essere precedente o uguale alla data di fine.")
        st.stop()

    mask = (df["ds"].dt.date >= start_date) & (df["ds"].dt.date <= end_date)
    df_train = df.loc[mask].copy()
    if df_train.empty:
        st.sidebar.error("Nessun dato disponibile nell'intervallo selezionato.")
        st.stop()

    st.write(f"Usando dati dal **{start_date}** al **{end_date}**: {len(df_train)} righe per il training.")
    st.dataframe(df_train.head(10))

    # -------------------------------------------------------------
    # 4) Scelta dei regressori
    # -------------------------------------------------------------
    st.sidebar.subheader("Seleziona i regressori da includere")
    regressors = st.sidebar.multiselect(
        "Regressori disponibili:",
        options=["meteo_temp", "ads_tiktok", "festivo"],
        default=["meteo_temp", "ads_tiktok", "festivo"],
        help="Scegli quali regressori includere nel modello Prophet"
    )
    st.sidebar.write(f"Regressori inclusi: {', '.join(regressors) if regressors else 'Nessuno'}")

    # -------------------------------------------------------------
    # 5) Orizzonte di forecast
    # -------------------------------------------------------------
    periodo_forecast = st.sidebar.slider(
        "Numero di giorni da prevedere nel futuro:",
        min_value=1, max_value=365, value=30, step=1,
        help="Scegli quanti giorni nel futuro vuoi prevedere"
    )
    st.sidebar.write(f"Orizzonte di forecast: **{periodo_forecast}** giorni")

    # -------------------------------------------------------------
    # 6) Bottone per eseguire il forecast
    # -------------------------------------------------------------
    if st.sidebar.button("Calcola Previsione"):
        # Prepara il DataFrame per Prophet
        df_prophet = df_train[["ds", target]].rename(columns={target: "y"}).copy()

        # Inizializza e configura Prophet
        m = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)

        # Aggiungi i regressori selezionati
        for reg in regressors:
            # Prophet si aspetta una colonna con lo stesso nome del regressore
            m.add_regressor(reg)

        # Fitting del modello
        with st.spinner("Addestramento del modello in corso…"):
            try:
                m.fit(pd.concat([df_prophet, df_train[regressors]], axis=1))
            except Exception as e:
                st.error(f"Errore durante il fitting di Prophet: {e}")
                st.stop()

        # Costruzione DataFrame futuro
        future = m.make_future_dataframe(periods=periodo_forecast, freq="D")
        # Unisci i regressori al dataframe futuro: per semplicità, assumiamo che l'utente abbia fornito valori futuri nel CSV          
        # Se il CSV non copre i giorni futuri, il modello non avrà i valori dei regressori; in un caso reale servirebbero previsioni dei regressori.
        future = future.merge(df[['ds'] + regressors], on='ds', how='left')

        with st.spinner("Calcolo della previsione…"):
            forecast = m.predict(future)

        st.success("Previsione completata!")

        # -------------------------------------------------------------
        # 7) Visualizzazioni interattive
        # -------------------------------------------------------------
        # a) Grafico principale: valori storici + forecast
        st.subheader("📈 Grafico Previsioni e Dati Storici")
        fig1 = m.plot(forecast)
        st.pyplot(fig1)

        # b) Grafico delle componenti (trend, stagionalità, regressori)
        st.subheader("🔍 Componenti del Modello")
        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)

        # c) Tabella risultati forecast vs storico
        st.subheader("📋 Tabella dei Risultati Forecast vs Storico")
        # Prende solo colonne ds, yhat, yhat_lower, yhat_upper, e unisce eventuale y reale
        df_display = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        df_display = df_display.merge(df_train[["ds", target]].rename(columns={target: "y"}), on="ds", how="left")
        st.dataframe(df_display.tail(periodo_forecast + 5).reset_index(drop=True))

        # d) Grafico interattivo con Streamlit (line chart)
        st.subheader("📊 Line Chart Interactive")
        df_plot = forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].copy()
        # Aggiunge la colonna storica "y" se esiste
        if "y" in df_display.columns:
            df_plot = pd.concat([df_plot, df_display.set_index("ds")["y"]], axis=1)
        st.line_chart(df_plot)

        # e) Download dei risultati
        st.subheader("💾 Scarica i Risultati Previsionali")
        csv_download = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False)
        st.download_button(
            label="Scarica risultati previsionali (CSV)",
            data=csv_download,
            file_name=f"forecast_{target}.csv",
            mime="text/csv"
        )

        # f) Se l'utente vuole salvare il grafico come immagine
        st.subheader("📷 Esporta il Grafico come Immagine")
        buffer = StringIO()
        fig1.savefig(buffer, format='png')
        st.download_button(
            label="Scarica grafico JSON (PNG)",
            data=buffer.getvalue().encode('utf-8'),
            file_name=f"forecast_plot_{target}.png",
            mime="image/png"
        )

    # -------------------------------------------------------------
    # 8) Sidebar: Opzioni aggiuntive e istruzioni
    # -------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **Istruzioni rapide:**
        1. Carica un file CSV con le colonne richieste.
        2. Seleziona il target: vendite o clienti.
        3. Imposta intervallo di date per il training.
        4. Scegli i regressori da includere.
        5. Imposta l’orizzonte di forecast (in giorni).
        6. Clicca su *Calcola Previsione* e attendi il risultato.
        """
    )

else:
    st.info("Carica un file CSV per iniziare l'analisi.")
