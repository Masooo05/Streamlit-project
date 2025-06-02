
main.py
+45
-33

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st

# 1) Carica i dati storici
data = pd.read_csv("dati_simulati.csv", parse_dates=['ds']).sort_values('ds')

# 2) Aggiungi trend, stagionalità e rumore ai target per renderli meno piatti
n = len(data)
# trend lineare
trend_liq = np.linspace(0, 1000, n)
trend_cli = np.linspace(0, 100, n)
# stagionalità settimanale artificiale (sinusoidale)
seasonal = 700 * np.sin(2 * np.pi * data['ds'].dt.dayofweek / 7)
seasonal_cli = 30 * np.sin(2 * np.pi * data['ds'].dt.dayofweek / 7)
# rumore
noise_liq = np.random.normal(0, 350, n)
noise_cli = np.random.normal(0, 20, n)

data['y_vendite'] = (
    data['y_vendite']
    + trend_liq
    + seasonal
    + noise_liq
).round(0)

data['y_clienti'] = (
    data['y_clienti']
    + trend_cli
    + seasonal_cli
    + noise_cli
).round(0)

# 3) Prepara i DataFrame per Prophet
liquidita_df = data[['ds','y_vendite','meteo_temp','ads_tiktok','festivo']].rename(columns={'y_vendite':'y'})
clienti_df  = data[['ds','y_clienti','meteo_temp','ads_tiktok','festivo']].rename(columns={'y_clienti':'y'})

# 4) Inizializza e addestra i modelli con stagionalità settimanale esplicita
model_liq = Prophet(
    weekly_seasonality=False,
    changepoint_prior_scale=0.3,
    seasonality_prior_scale=10,
)
model_liq.add_seasonality(name='weekly', period=7, fourier_order=3)
for reg in ['meteo_temp','ads_tiktok','festivo']:
    model_liq.add_regressor(reg)
model_liq.fit(liquidita_df)

model_cli = Prophet(
    weekly_seasonality=False,
    changepoint_prior_scale=0.3,
    seasonality_prior_scale=10,
)
model_cli.add_seasonality(name='weekly', period=7, fourier_order=3)
for reg in ['meteo_temp','ads_tiktok','festivo']:
    model_cli.add_regressor(reg)
model_cli.fit(clienti_df)

# 5) Crea future_dataframe per 100 giorni e costruisci regressori variabili
future_liq = model_liq.make_future_dataframe(periods=100)
future_cli = model_cli.make_future_dataframe(periods=100)

# regressori futuri: ads ciclico, meteo con stagionalità mensile, festivi casuali
days = np.arange(len(future_liq))
future_liq['ads_tiktok']  = 75 + 150 * np.abs(np.sin(2*np.pi*days/20))
future_liq['meteo_temp']  = 15 + 15 * np.sin(2*np.pi*days/90) + np.random.normal(0,2,len(days))
future_liq['festivo']     = ((days % 10) == 0).astype(int)  # ogni 10 giorni

future_cli['ads_tiktok']  = 3 * future_liq['ads_tiktok']
future_cli['meteo_temp']  = 5 * future_liq['meteo_temp']
future_cli['festivo']     = 3 * future_liq['festivo']

# 6) Genera le previsioni
fc_liq = model_liq.predict(future_liq)
fc_cli = model_cli.predict(future_cli)

# 7) Plot con Streamlit
cutoff = liquidita_df['ds'].max()
fig, ax = plt.subplots(figsize=(14, 7))

# storico
ax.plot(liquidita_df.ds, liquidita_df.y, '--', c='navy', alpha=0.6, label='Storico Liquidità')
ax.plot(clienti_df.ds, clienti_df.y, '--', c='darkgreen', alpha=0.6, label='Storico Clienti')

# forecast + incertezza
ax.plot(fc_liq.ds, fc_liq.yhat, color='navy', lw=2, label='Previsione Liquidità')
ax.fill_between(
    fc_liq.ds,
    fc_liq.yhat_lower,
    fc_liq.yhat_upper,
    color='navy', alpha=0.2
)

ax.plot(fc_cli.ds, fc_cli.yhat, color='darkgreen', lw=2, label='Previsione Clienti')
ax.fill_between(
    fc_cli.ds,
    fc_cli.yhat_lower,
    fc_cli.yhat_upper,
    color='darkgreen', alpha=0.2
)
# evidenzia futuro
ax.axvspan(cutoff, fc_liq.ds.max(), color='gray', alpha=0.1, label='Periodo di Previsione')

ax.set_title(
    "Previsione 100 giorni per Liquidità e Affluenza Clienti\n(dati simulati con trend, stagionalità e regressori variabili)"
)
ax.set_xlabel("Data")
ax.set_ylabel("Valore")
ax.legend()
ax.grid(True)
fig.tight_layout()

st.pyplot(fig)
