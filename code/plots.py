"""Figuras del proyecto (archivo compartido).

Secciones por dueño: EDA (Milca) primero; Arturo y Paula agregan sus
funciones al final del archivo.

Se usa el backend "Agg" (sin ventana) para que las figuras se generen igual
en Mac, Linux y Windows, también sin pantalla.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (después de fijar el backend)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402

import data  # noqa: E402

# Paleta categórica en orden fijo (azul, naranja, aqua, amarillo) y tintas de texto.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BULL_FILL, BEAR_FILL = "#d6efe4", "#fbe1d6"
PAGE_SIZE = (11, 8.5)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.edgecolor": MUTED,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
    }
)


# ---------------------------------------------------------------------------
# EDA (Milca): una función por figura; write_eda_pdf arma el PDF.
# ---------------------------------------------------------------------------

def new_page(title):
    """Hoja tamaño carta horizontal con el título del EDA."""
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.suptitle(title, fontsize=15, fontweight="bold", color=INK, x=0.04, ha="left")
    return fig


def add_design_note(fig, text):
    """Recuadro al pie de la hoja con la decisión de diseño que apoya la figura."""
    fig.text(
        0.04, 0.02, "Decisión de diseño que apoya: " + text,
        fontsize=10.5, color=INK, wrap=True, va="bottom",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f4f3ef", "edgecolor": GRID},
    )


def shade_regimes(ax, regimes):
    """Sombrea cada régimen (verde alcista, naranja bajista) según su retorno."""
    for row in regimes.itertuples():
        color = BULL_FILL if row.retorno_total > 0 else BEAR_FILL
        ax.axvspan(pd.Timestamp(row.inicio, tz="UTC"), pd.Timestamp(row.fin, tz="UTC") + pd.Timedelta(days=1),
                   color=color, zorder=0, lw=0)


def regime_handles():
    """Muestras de color de la leyenda para los regímenes alcista y bajista."""
    return [Patch(color=BULL_FILL, label="Régimen alcista"), Patch(color=BEAR_FILL, label="Régimen bajista")]


def legend_with_regimes(ax, loc, ncol=1):
    """Leyenda con las series del eje más los regímenes, sobre fondo blanco para que se lean."""
    ax.legend(handles=[*ax.get_legend_handles_labels()[0], *regime_handles()], loc=loc, ncol=ncol,
              frameon=True, facecolor="white", edgecolor=GRID, framealpha=1)


def style_table(table, col_widths, font_size=10, row_height=1.9):
    """Formato común de tablas: anchos de columna, tipografía y encabezado gris."""
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, row_height)
    for (row, col), cell in table.get_celld().items():
        cell.set_width(col_widths[col])
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f4f3ef")


def plot_price_regimes(fig, bars, regimes):
    """Precio de cierre en escala log con los regímenes sombreados y numerados."""
    ax = fig.add_axes([0.07, 0.25, 0.9, 0.63])
    shade_regimes(ax, regimes)
    ax.plot(bars.index, bars["close"], color=SERIES[0], lw=1.0, label="Cierre horario BTCUSDT")
    ax.set_yscale("log")
    for i, row in enumerate(regimes.itertuples(), start=1):
        middle = pd.Timestamp(row.inicio, tz="UTC") + (pd.Timestamp(row.fin, tz="UTC") - pd.Timestamp(row.inicio, tz="UTC")) / 2
        ax.text(middle, ax.get_ylim()[1] * 0.8, f"R{i}", ha="center", va="top", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel("Fecha (UTC)")
    ax.set_ylabel("Precio de cierre (USDT, escala log)")
    ax.set_title("Precio y regímenes · train del fold 4 (2017-08 a 2024-12)", loc="left")
    legend_with_regimes(ax, "lower right")


def plot_regime_table(fig, regimes, thresholds):
    """Tabla de regímenes (fechas, retorno, tendencia, volatilidad, drawdown) y sensibilidad del umbral del zigzag."""
    ax = fig.add_axes([0.04, 0.5, 0.92, 0.38])
    ax.axis("off")
    ax.set_title("Caracterización de regímenes · train del fold 4", loc="left")
    header = ["", "Régimen", "Inicio", "Fin", "Días", "Retorno\ntotal", "Tendencia\nlog anual",
              "Volatilidad\nanualizada", "Drawdown\nmáximo", "Curtosis\nen exceso"]
    cells = [
        [f"R{i}", r.regimen, r.inicio, r.fin, f"{r.dias}", f"{r.retorno_total:+.0%}",
         f"{r.tendencia_log_anual:+.2f}", f"{r.vol_anualizada:.0%}", f"{r.max_drawdown:.0%}", f"{r.curtosis_exceso:.1f}"]
        for i, r in enumerate(regimes.itertuples(), start=1)
    ]
    table = ax.table(cellText=cells, colLabels=header, loc="upper center", cellLoc="center")
    style_table(table, [0.04, 0.22, 0.09, 0.09, 0.06, 0.08, 0.09, 0.1, 0.09, 0.09], font_size=9.5, row_height=1.55)
    for col in range(len(header)):
        table[0, col].set_height(table[0, col].get_height() * 1.6)

    bottom = fig.add_axes([0.04, 0.24, 0.6, 0.27])
    bottom.axis("off")
    bottom.set_title("Umbral del zigzag sobre el cierre diario · solo train", loc="left")
    header = ["Umbral", "Giros\nfold 4", "Regímenes\nfold 4", *[f"Confirmados\nfold {k}" for k in range(1, 5)]]
    cells = [[f"{r.umbral:.0%}", r.giros_fold_4, r.regimenes_fold_4, r.confirmados_fold_1, r.confirmados_fold_2,
              r.confirmados_fold_3, r.confirmados_fold_4] for r in thresholds.itertuples()]
    table = bottom.table(cellText=cells, colLabels=header, loc="upper center", cellLoc="center")
    style_table(table, [0.1, 0.13, 0.15, 0.155, 0.155, 0.155, 0.155], font_size=9.5, row_height=1.5)
    for col in range(len(header)):
        table[0, col].set_height(table[0, col].get_height() * 1.7)
    chosen = thresholds.set_index("umbral").loc[data.REGIME_THRESHOLD]
    fig.text(0.67, 0.51, "Los regímenes son solo descriptivos: no entran al\n"
                         "estado del agente ni a ninguna decisión.\n\n"
                         f"Con {data.REGIME_THRESHOLD:.0%}, el zigzag marca {chosen.giros_fold_4} giros; la tabla\n"
                         "de arriba une a mano el rebote de 2018-02-05\n"
                         "a 2018-03-04.\n"
                         "Con 50% el ciclo 2020-21 se parte en la corrección\n"
                         "de abril a julio de 2021; con 70% se pierden la\n"
                         "recuperación de 2019 y la caída del COVID. 60%\n"
                         "separa ciclos de mercado y no correcciones\n"
                         "intermedias.\n\n"
                         "\"Confirmados\": giros que el zigzag ya confirma con\n"
                         "solo el train de ese fold (el último tramo queda\n"
                         "abierto). \"Fold 4\": train del fold 4.",
             fontsize=9.5, color=INK, va="top")


def plot_drawdown_volatility(fig, bars, regimes):
    """Drawdown desde el máximo previo y volatilidad móvil de 168 h anualizada."""
    ret = np.log(bars["close"]).diff()
    drawdown = bars["close"] / bars["close"].cummax() - 1
    vol = ret.rolling(data.LONG_WINDOW).std() * np.sqrt(data.HOURS_PER_YEAR)
    top = fig.add_axes([0.07, 0.6, 0.9, 0.28])
    bottom = fig.add_axes([0.07, 0.25, 0.9, 0.28], sharex=top)
    for ax in (top, bottom):
        shade_regimes(ax, regimes)
    top.plot(drawdown.index, drawdown * 100, color=SERIES[1], lw=1.0, label="Drawdown desde el máximo previo")
    top.set_ylabel("Drawdown (%)")
    top.set_title("Drawdown · train del fold 4", loc="left")
    top.set_ylim(-118, 5)
    legend_with_regimes(top, "lower center", ncol=3)
    bottom.plot(vol.index, vol * 100, color=SERIES[0], lw=1.0, label="Volatilidad móvil 168 h, anualizada")
    bottom.set_ylabel("Volatilidad anualizada (%)")
    bottom.set_xlabel("Fecha (UTC)")
    bottom.set_title("Agrupamiento de volatilidad · train del fold 4", loc="left")
    legend_with_regimes(bottom, "upper right")


def plot_return_distribution(fig, returns, summary):
    """Histograma del log-retorno horario contra una normal, y gráfica Q-Q."""
    r = returns.dropna()
    left = fig.add_axes([0.07, 0.25, 0.42, 0.6])
    right = fig.add_axes([0.57, 0.25, 0.4, 0.6])
    bins = np.linspace(r.quantile(0.0005), r.quantile(0.9995), 160)
    left.hist(r, bins=bins, density=True, color=SERIES[0], alpha=0.85, label="Log-retorno horario")
    grid = np.linspace(bins[0], bins[-1], 400)
    left.plot(grid, stats.norm.pdf(grid, r.mean(), r.std()), color=SERIES[1], lw=2, label="Normal con misma media y desviación")
    left.set_yscale("log")
    left.set_ylim(bottom=1e-2)
    left.set_xlabel("Log-retorno horario (recortado a los percentiles 0.05 y 99.95)")
    left.set_ylabel("Densidad (escala log)")
    left.set_title(f"Distribución · curtosis en exceso = {summary['curtosis_exceso']:.1f} · train fold 4", loc="left")
    left.legend(loc="upper left", fontsize=9.5)
    (theoretical, ordered), _ = stats.probplot((r - r.mean()) / r.std(), dist="norm")
    right.plot(theoretical, ordered, ".", color=SERIES[0], ms=3, label="Cuantiles observados")
    right.plot(theoretical, theoretical, color=SERIES[1], lw=2, label="Referencia normal")
    right.set_xlabel("Cuantil teórico normal (desviaciones)")
    right.set_ylabel("Cuantil observado (desviaciones)")
    right.set_title("Gráfica Q-Q contra la normal · train fold 4", loc="left")
    right.legend(loc="upper left")


def plot_autocorrelation(fig, acf_summary):
    """ACF del retorno y del retorno absoluto hasta 168 rezagos con banda de 95%."""
    band = acf_summary["banda_95"]
    panels = (
        ("acf_ret", "Retorno r", SERIES[0], [0.07, 0.6, 0.9, 0.27], acf_summary["rezagos_signif_ret"]),
        ("acf_abs", "Retorno absoluto |r|", SERIES[1], [0.07, 0.24, 0.9, 0.27], acf_summary["rezagos_signif_abs"]),
    )
    for key, label, color, rect, n_signif in panels:
        ax = fig.add_axes(rect)
        values = acf_summary[key][1:]
        lags = np.arange(1, len(values) + 1)
        ax.vlines(lags, 0, values, color=color, lw=1.6, label=f"ACF de {label}")
        ax.axhspan(-band, band, color=GRID, alpha=0.9, label=f"Banda de 95% (±{band:.4f})")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlim(0, len(values) + 1)
        ax.set_ylabel("Autocorrelación")
        ax.set_title(f"ACF de {label} · {n_signif} de {len(values)} rezagos fuera de la banda · train fold 4", loc="left")
        ax.legend(loc="upper right")
    ax.set_xlabel("Rezago (horas)")


def plot_stationarity_table(fig, table):
    """Tabla de p-valores ADF y KPSS por serie y fold (train)."""
    ax = fig.add_axes([0.04, 0.25, 0.92, 0.6])
    ax.axis("off")
    ax.set_title("Pruebas de estacionariedad · train de cada fold", loc="left")
    names = {"precio": "Precio", "log_precio": "Log-precio", "retorno_1h": "Retorno 1 h", "volatilidad_24h": "Volatilidad 24 h"}
    header = ["Serie"] + [f"Fold {f}\nADF p / KPSS p" for f in sorted(table["fold"].unique())] + ["Conclusión"]
    cells = []
    for serie, group in table.groupby("serie", sort=False):
        pvalues = [f"{r.adf_pvalue:.3f} / {'≤0.01' if r.kpss_pvalue <= 0.01 else ('≥0.10' if r.kpss_pvalue >= 0.1 else f'{r.kpss_pvalue:.2f}')}"
                   for r in group.itertuples()]
        if group["estacionaria"].all():
            verdict = "Estacionaria"
        elif (group["adf_pvalue"] >= 0.05).all():
            verdict = "Raíz unitaria\n(no estacionaria)"
        else:
            verdict = "Memoria larga\n(ADF y KPSS rechazan)"
        cells.append([names[serie]] + pvalues + [verdict])
    tbl = ax.table(cellText=cells, colLabels=header, loc="upper center", cellLoc="center")
    style_table(tbl, [0.14, 0.15, 0.15, 0.15, 0.15, 0.22], row_height=3.2)
    ax.text(0, 0.05, "ADF: H0 = raíz unitaria. KPSS: H0 = estacionaria. statsmodels tabula el p-valor KPSS solo entre 0.01 y 0.10.",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)


def plot_intraday_seasonality(fig, bars):
    """Retorno absoluto medio por hora del día y por día de la semana (UTC)."""
    abs_ret = np.log(bars["close"]).diff().abs() * 100
    by_hour = abs_ret.groupby(bars.index.hour).mean()
    by_day = abs_ret.groupby(bars.index.day_of_week).mean()
    left = fig.add_axes([0.07, 0.25, 0.55, 0.6])
    right = fig.add_axes([0.7, 0.25, 0.27, 0.6])
    left.bar(by_hour.index, by_hour.to_numpy(), color=SERIES[0], width=0.8, label="|r| medio por hora")
    left.axhline(abs_ret.mean(), color=SERIES[1], lw=2, label="Promedio general")
    left.set_xlabel("Hora de apertura de la barra (UTC)")
    left.set_ylabel("Retorno absoluto medio (%)")
    left.set_title("Estacionalidad intradía · train fold 4", loc="left")
    left.set_ylim(0, by_hour.max() * 1.3)
    left.legend(loc="upper left")
    days = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    right.bar(days, by_day.to_numpy(), color=SERIES[0], width=0.8, label="|r| medio por día")
    right.axhline(abs_ret.mean(), color=SERIES[1], lw=2, label="Promedio general")
    right.set_xlabel("Día de la semana (UTC)")
    right.set_ylabel("Retorno absoluto medio (%)")
    right.set_title("Por día de la semana · train fold 4", loc="left")
    right.set_ylim(0, by_day.max() * 1.3)
    right.legend(loc="upper left", fontsize=9.5)
    return by_hour, by_day


def plot_feature_information(fig, analysis):
    """Información mutua máxima por familia con el objetivo a 1, 4 y 24 h (train de un fold).

    Sobre cada barra, una marca negra con la MI de esa misma variable contra
    el objetivo barajado por bloques de 30 días (lo que se obtiene solo por
    identificar la época).
    """
    ax = fig.add_axes([0.07, 0.3, 0.9, 0.55])
    families = list(dict.fromkeys(analysis["familia"]))
    width = 0.26
    x = np.arange(len(families))
    for i, h in enumerate(data.HORIZONS):
        best = analysis.loc[analysis.groupby("familia", sort=False)[f"mi_h{h}"].idxmax()].set_index("familia").reindex(families)
        ax.bar(x + (i - 1) * width, best[f"mi_h{h}"].to_numpy(), width=width * 0.92, color=SERIES[i], label=f"MI máxima, objetivo h = {h} h")
        ax.scatter(x + (i - 1) * width, best[f"mi_null_bloques_h{h}"].to_numpy(), marker="_", s=180, linewidths=2.2,
                   color=INK, zorder=3, label="MI de esa variable con objetivo barajado por bloques de 30 días" if i == 0 else None)
    noise = analysis[[f"mi_null_h{h}" for h in data.HORIZONS]].to_numpy().max()
    ax.axhline(noise, color=INK, lw=1.2, ls="--", label=f"MI máxima con objetivo barajado hora por hora (ruido) = {noise:.3f}")
    ax.set_ylim(0, analysis[[f"mi_h{h}" for h in data.HORIZONS]].to_numpy().max() * 1.4)
    ax.set_xticks(x, [f.replace("_", " / ").replace("rechazada / nivel", "rechazada\n(nivel)") for f in families])
    ax.set_xlabel("Familia de variables")
    ax.set_ylabel("Información mutua (nats)")
    fold = int(analysis["fold"].iloc[0])
    ax.set_title(f"Información mutua con el retorno futuro · train del fold {fold}", loc="left")
    ax.legend(loc="upper left")


def plot_data_quality(fig, summary, gaps):
    """Tablas de limpieza (archivo completo) y de huecos por fold y periodo.

    Solo cuenta barras y huecos; no usa precios ni retornos de valid/test.
    """
    top = fig.add_axes([0.04, 0.6, 0.45, 0.28])
    top.axis("off")
    top.set_title("Limpieza del archivo de Canvas (archivo completo)", loc="left")
    items = [
        ("Filas del archivo", f"{summary['filas_crudas']:,}"),
        ("Filas repetidas eliminadas", f"{summary['duplicados_eliminados']:,}"),
        ("Barras redondeadas a la hora (2018)", f"{summary['barras_redondeadas_a_la_hora']:,}"),
        ("Barras únicas", f"{summary['barras_unicas']:,}"),
        ("Horas faltantes (malla completa)", f"{summary['horas_faltantes']:,}"),
        ("Barras con hueco previo", f"{summary['barras_con_gap']:,}"),
        ("Barras con volumen cero", f"{summary['barras_volumen_cero']:,}"),
    ]
    table = top.table(cellText=[list(item) for item in items], colLabels=["Concepto", "Valor"], loc="upper left", cellLoc="left")
    style_table(table, [0.75, 0.25], font_size=9.5, row_height=1.25)

    bottom = fig.add_axes([0.04, 0.17, 0.92, 0.38])
    bottom.axis("off")
    bottom.set_title("Huecos por fold y periodo (conteo de barras)", loc="left")
    header = ["Fold", "Periodo", "Barras", "Horas\nfaltantes", "Huecos\n1 h", "Huecos\n2-3 h", "Huecos\n> 3 h",
              "Hueco más\nlargo (h)", "Filas con hueco\nen ventana 168"]
    cells = [[r.fold, r.periodo, f"{r.barras:,}", r.horas_faltantes, r.huecos_1h, r.huecos_2a3h, r.huecos_mas_3h,
              r.hueco_mas_largo_h, f"{r.filas_hueco_en_ventana_168:,} ({r.filas_hueco_en_ventana_168 / r.barras:.0%})"]
             for r in gaps.itertuples()]
    table = bottom.table(cellText=cells, colLabels=header, loc="upper center", cellLoc="center")
    style_table(table, [0.06, 0.08, 0.09, 0.1, 0.09, 0.09, 0.09, 0.11, 0.17], font_size=9.5, row_height=1.3)
    for col in range(len(header)):
        table[0, col].set_height(table[0, col].get_height() * 2)


def format_pvalue(p):
    """Texto de un p-valor; los que no se distinguen de cero se reportan como "< 1e-300"."""
    return "< 1e-300" if p < 1e-300 else f"= {p:.1g}"


def write_eda_pdf(bars, path=data.ARTIFACTS_DIR / "btcusdt_eda.pdf", raw=None):
    """Escribe artifacts/btcusdt_eda.pdf con datos SOLO de entrenamiento.

    EDA general sobre el train del fold 4 (2017-08 a 2024-12, contiene a los
    demás trains); la tabla de estacionariedad cubre el train de los 4 folds.
    Cada hoja termina con la decisión de diseño que sustenta. La última hoja
    (calidad de datos) solo cuenta barras y huecos por periodo. ``raw`` es el
    CSV crudo (``data.read_raw()``); si no se pasa, se lee del repo. Sin fecha
    de creación en los metadatos, para que repetir la corrida dé el mismo archivo.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    train = data.eda_train(bars)
    returns = np.log(train["close"]).diff()
    regimes = data.characterize_regimes(train)
    distribution = data.return_distribution_stats(returns)
    acf_summary = data.autocorrelation_summary(returns)
    stationarity = data.stationarity_table(bars)
    analysis = data.feature_analysis(bars, data.get_folds()[data.EDA_FOLD - 1])
    summary = data.cleaning_summary(data.read_raw() if raw is None else raw, bars)
    gaps = data.gap_report(bars)

    with PdfPages(path, metadata={"CreationDate": None}) as pdf:
        fig = new_page("1. Regímenes de BTCUSDT")
        plot_price_regimes(fig, train, regimes)
        add_design_note(fig, f"el precio pasa de {train['close'].iloc[0]:,.0f} a {train['close'].iloc[-1]:,.0f} USDT "
                             f"(×{train['close'].iloc[-1] / train['close'].iloc[0]:.0f}); un mismo nivel de precio no significa lo mismo "
                             "en distintos regímenes, por eso el estado usa retornos, razones y distancias en log, nunca precios en nivel.")
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page("2. Caracterización de regímenes")
        plot_regime_table(fig, regimes, data.regime_threshold_table(bars))
        add_design_note(fig, f"la volatilidad anualizada va de {regimes['vol_anualizada'].min():.0%} a {regimes['vol_anualizada'].max():.0%} "
                             f"y el drawdown llega a {regimes['max_drawdown'].min():.0%}; ningún régimen representa a todo el train. "
                             "Por eso el estado incluye volatilidad y la creencia de tendencia con su incertidumbre, y cada fold se normaliza solo con su train.")
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page("3. Drawdown y agrupamiento de volatilidad")
        plot_drawdown_volatility(fig, train, regimes)
        vol = returns.rolling(data.LONG_WINDOW).std() * np.sqrt(data.HOURS_PER_YEAR)
        add_design_note(fig, f"la volatilidad de 168 h anualizada oscila entre {vol.quantile(0.05):.0%} (p5) y {vol.quantile(0.95):.0%} (p95) "
                             "y llega en episodios; las variables de volatilidad (24 y 168 h) informan al agente del riesgo actual "
                             "y un stop-loss fijo en % significa riesgos muy distintos según el episodio.")
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page("4. Distribución de retornos: colas gordas")
        plot_return_distribution(fig, returns, distribution)
        add_design_note(fig, f"curtosis en exceso {distribution['curtosis_exceso']:.1f} y Jarque-Bera p {format_pvalue(distribution['jarque_bera_p'])}; "
                             f"{distribution['frac_mas_de_4_sd']:.2%} de las horas superan 4 desviaciones contra {distribution['frac_normal_mas_de_4_sd']:.4%} "
                             "en una normal. Por eso el normalizador usa mediana e IQR con recorte a ±5 en lugar de media y desviación.")
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page("5. Autocorrelación: dirección contra magnitud")
        plot_autocorrelation(fig, acf_summary)
        max_ret = np.abs(acf_summary["acf_ret"][1:]).max()
        add_design_note(fig, f"la ACF de r es casi nula (máximo |ACF| = {max_ret:.3f}) mientras la de |r| sigue fuera de la banda en "
                             f"{acf_summary['rezagos_signif_abs']} de 168 rezagos (ACF rezago 1 = {acf_summary['acf_abs'][1]:.2f}). Se rechaza una "
                             "ventana larga de retornos crudos y se usan pocos retornos resumidos (1, 4, 24, 168 h) más variables de volatilidad.")
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page("6. Estacionariedad por fold")
        plot_stationarity_table(fig, stationarity)
        level = stationarity[stationarity["serie"].isin(["precio", "log_precio"])]
        add_design_note(fig, f"precio y log-precio no rechazan raíz unitaria en ningún fold (ADF p entre {level['adf_pvalue'].min():.2f} y "
                             f"{level['adf_pvalue'].max():.2f}): es la alternativa rechazada para el estado. El retorno es estacionario en los 4 folds; "
                             "la volatilidad tiene memoria larga, se usa pero con normalización robusta por fold.")
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page("7. Estacionalidad por hora y día")
        by_hour, by_day = plot_intraday_seasonality(fig, train)
        add_design_note(fig, f"el |r| medio va de {by_hour.min():.3f}% a {by_hour.max():.3f}% según la hora (×{by_hour.max() / by_hour.min():.1f}) "
                             f"y de {by_day.min():.3f}% a {by_day.max():.3f}% según el día; la hora y el día (seno y coseno) entran como familia "
                             "tiempo para que el agente distinga horas activas de horas tranquilas.")
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page("8. Información de cada familia sobre el retorno futuro")
        plot_feature_information(fig, analysis)
        rows = analysis.set_index("variable")
        level_share = rows.loc["log_close", "mi_null_bloques_h24"] / rows.loc["log_close", "mi_h24"]
        ema_share = rows.loc["dist_ema_168", "mi_null_bloques_h24"] / rows.loc["dist_ema_168", "mi_h24"]
        add_design_note(fig, "volatilidad y creencia aportan la mayor MI (sobre todo en la magnitud del movimiento); tiempo y volumen quedan "
                             f"cerca del ruido. El log-precio en nivel tiene MI alta a 24 h ({rows.loc['log_close', 'mi_h24']:.3f}), pero no es "
                             f"estacionario y el objetivo barajado por bloques de 30 días conserva {level_share:.0%} de esa MI (contra {ema_share:.0%} "
                             "en dist_ema_168): sobre todo identifica la época, no una regla. Por eso se rechazan las variables en nivel.")
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page("9. Calidad de datos: limpieza y huecos")
        plot_data_quality(fig, summary, gaps)
        worst = gaps.loc[gaps["filas_hueco_en_ventana_168"].div(gaps["barras"]).idxmax()]
        add_design_note(fig, "los huecos NO se rellenan: la columna gap_hours los marca y las ventanas móviles cuentan barras observadas. "
                             f"El fold {worst['fold']} ({worst['periodo']}) tiene un hueco dentro de la ventana de 168 barras en "
                             f"{worst['filas_hueco_en_ventana_168'] / worst['barras']:.0%} de sus filas; 2025 y 2026 deben "
                             "leerse con esa salvedad al comparar folds, y el simulador debe distinguir huecos de 1 h de huecos largos.")
        pdf.savefig(fig)
        plt.close(fig)
