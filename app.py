from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import plotly.express as px
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Tab global de energía – sieLAC",
    page_icon="🌎",
    layout="wide",
)

BASE_PATH = Path(__file__).resolve().parent
SUPPLY_PATH = BASE_PATH / "Oferta y demanda " / " Todos " / "Oferta total de energía.csv"
CONSUMPTION_PATH = BASE_PATH / "Oferta y demanda " / " Todos " / "ConsumoFinal.csv"
RENEWABLES_PATH = BASE_PATH / "Oferta y demanda " / " Renovables " / "Renovables-Consumo final.csv"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _strip_quotes(value: str) -> str:
    return value.strip().strip("'\"")


def _to_number(value: str | float | int | None) -> float | None:
    if value is None:
        return None

    text = str(value)
    text = _strip_quotes(text)
    text = text.replace("\u202f", "").replace("\ufeff", "")
    text = text.replace(" ", "")

    if text == "":
        return None

    # Si solo hay una coma y no hay punto, asumimos coma decimal.
    if text.count(",") == 1 and text.count(".") == 0:
        text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return None


def _clean_columns(columns: Iterable[str]) -> List[str]:
    return [_strip_quotes(col) for col in columns]


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

@st.cache_data
def load_energy_supply() -> pd.DataFrame:
    df = pd.read_csv(SUPPLY_PATH)
    df.columns = _clean_columns(df.columns)

    rename_map: Dict[str, str] = {
        "Carbón mineral (Cantidad 10⁶ tep)": "Carbón mineral",
        "Gas natural (Cantidad 10⁶ tep)": "Gas natural",
        "Hidroenergía (Cantidad 10⁶ tep)": "Hidroenergía",
        "Nuclear (Cantidad 10⁶ tep)": "Nuclear",
        "Petróleo (Cantidad 10⁶ tep)": "Petróleo",
        "Otras renovables (Cantidad 10⁶ tep)": "Otras renovables",
    }
    df = df.rename(columns=rename_map)

    numeric_cols = [col for col in df.columns if col != "Region"]
    for col in numeric_cols:
        df[col] = df[col].apply(_to_number)

    df["Oferta total"] = df[numeric_cols].sum(axis=1)
    return df


@st.cache_data
def load_final_consumption() -> pd.DataFrame:
    df = pd.read_csv(CONSUMPTION_PATH)
    df.columns = _clean_columns(df.columns)
    df = df.rename(columns={"Energía (Cantidad 10⁶ tep)": "Energía"})
    df["Año"] = df["Año"].apply(_strip_quotes)
    df["Region"] = df["Region"].apply(_strip_quotes)
    df["Energía"] = df["Energía"].apply(_to_number)
    return df


@st.cache_data
def load_renewables_consumption() -> pd.DataFrame:
    df = pd.read_csv(RENEWABLES_PATH)
    df.columns = _clean_columns(df.columns)
    rename_map = {
        "Otros (Consumo 10³ bep)": "Otros",
        "Otra biomasa (Consumo 10³ bep)": "Otra biomasa",
    }
    df = df.rename(columns=rename_map)
    df["Año"] = df["Año"].apply(_strip_quotes)
    df["Region"] = df["Region"].apply(_strip_quotes)

    numeric_cols = [col for col in df.columns if col not in {"Año", "Region"}]
    for col in numeric_cols:
        df[col] = df[col].apply(_to_number)

    return df


# ---------------------------------------------------------------------------
# Componentes visuales
# ---------------------------------------------------------------------------

def render_world_tab(supply_df: pd.DataFrame, consumption_df: pd.DataFrame) -> None:
    st.subheader("Panorama mundial (2023)")

    world_row = supply_df.loc[supply_df["Region"] == "Mundo"]
    world_consumption = consumption_df.loc[
        consumption_df["Region"] == "Mundo", "Energía"
    ].iloc[0]

    supply_cols = [col for col in supply_df.columns if col not in {"Region", "Oferta total"}]
    total_supply = world_row[supply_cols].sum(axis=1).iloc[0]
    renewables_supply = world_row[["Hidroenergía", "Otras renovables"]].sum(axis=1).iloc[0]
    renewables_share = (renewables_supply / total_supply) * 100 if total_supply else 0

    metric_cols = st.columns(3)
    metric_cols[0].metric("Oferta total", f"{total_supply:,.1f} 10⁶ tep")
    metric_cols[1].metric("Consumo final", f"{world_consumption:,.1f} 10⁶ tep")
    metric_cols[2].metric("Participación hidro + renovables", f"{renewables_share:,.1f}%")

    melted = (
        world_row.melt(
            id_vars="Region",
            value_vars=supply_cols,
            var_name="Fuente",
            value_name="Energía (10⁶ tep)",
        )
        .sort_values("Energía (10⁶ tep)", ascending=False)
    )

    st.plotly_chart(
        px.bar(
            melted,
            x="Fuente",
            y="Energía (10⁶ tep)",
            color="Fuente",
            title="Distribución de la oferta mundial por fuente",
            text_auto=True,
        ).update_layout(showlegend=False),
        use_container_width=True,
    )

    supply_long = supply_df.melt(
        id_vars=["Region"],
        value_vars=supply_cols,
        var_name="Fuente",
        value_name="Energía (10⁶ tep)",
    )

    top_regions = supply_df.nlargest(5, "Oferta total")["Region"].tolist()
    st.plotly_chart(
        px.bar(
            supply_long[supply_long["Region"].isin(top_regions)],
            x="Region",
            y="Energía (10⁶ tep)",
            color="Fuente",
            title="Oferta por fuente en las 5 regiones con mayor aporte",
            barmode="stack",
        ),
        use_container_width=True,
    )

    st.dataframe(
        world_row.set_index("Region")[supply_cols + ["Oferta total"]].T,
        use_container_width=True,
        height=260,
    )


def render_region_tab(supply_df: pd.DataFrame, consumption_df: pd.DataFrame) -> None:
    st.subheader("Comparativo regional")

    regions = supply_df["Region"].unique().tolist()
    default_regions = [r for r in regions if r not in {"Mundo", "OCDE"}][:5]
    selected_regions = st.multiselect(
        "Selecciona regiones para explorar",
        options=regions,
        default=default_regions,
    )

    supply_cols = [col for col in supply_df.columns if col not in {"Region", "Oferta total"}]
    supply_long = supply_df.melt(
        id_vars=["Region"],
        value_vars=supply_cols,
        var_name="Fuente",
        value_name="Energía (10⁶ tep)",
    )

    if selected_regions:
        filtered_supply = supply_long[supply_long["Region"].isin(selected_regions)]
        st.plotly_chart(
            px.bar(
                filtered_supply,
                x="Region",
                y="Energía (10⁶ tep)",
                color="Fuente",
                title="Oferta energética por región y fuente",
                barmode="stack",
            ),
            use_container_width=True,
        )

        filtered_consumption = consumption_df[
            consumption_df["Region"].isin(selected_regions)
        ]
        st.plotly_chart(
            px.scatter(
                filtered_consumption,
                x="Region",
                y="Energía",
                size="Energía",
                color="Region",
                title="Consumo final por región",
            ),
            use_container_width=True,
        )

    detail_cols = st.columns((2, 1))
    detail_cols[0].markdown("#### Datos crudos de oferta")
    detail_cols[0].dataframe(
        supply_df.loc[supply_df["Region"].isin(selected_regions)]
        if selected_regions
        else supply_df,
        use_container_width=True,
    )

    detail_cols[1].markdown("#### Consumo final")
    detail_cols[1].dataframe(
        consumption_df.loc[consumption_df["Region"].isin(selected_regions)]
        if selected_regions
        else consumption_df,
        use_container_width=True,
        height=320,
    )


def render_renewables_tab(renewables_df: pd.DataFrame) -> None:
    st.subheader("Renovables y biomasa")

    st.caption(
        "Las cifras están en miles de barriles equivalentes de petróleo (10³ bep)."
    )

    regions = renewables_df["Region"].unique().tolist()
    selected = st.multiselect(
        "Regiones a comparar",
        options=regions,
        default=["Mundo", "América Latina y el Caribe", "Asia & Australia"],
    )

    filtered = renewables_df[renewables_df["Region"].isin(selected)]
    renew_long = filtered.melt(
        id_vars=["Año", "Region"],
        var_name="Categoría",
        value_name="Consumo (10³ bep)",
    )

    st.plotly_chart(
        px.bar(
            renew_long,
            x="Region",
            y="Consumo (10³ bep)",
            color="Categoría",
            title="Consumo renovable y de biomasa",
            barmode="group",
        ),
        use_container_width=True,
    )

    world_slice = renewables_df[renewables_df["Region"] == "Mundo"]
    st.dataframe(world_slice, use_container_width=True, height=200)


# ---------------------------------------------------------------------------
# UI principal
# ---------------------------------------------------------------------------

st.title("Datos globales de energía")
st.caption(
    "Tablero interactivo construido a partir de los archivos provistos en el dataset"
    " sieLAC. Los valores corresponden a unidades energéticas comunes para"
    " facilitar la comparación entre regiones y fuentes."
)

supply_df = load_energy_supply()
consumption_df = load_final_consumption()
renewables_df = load_renewables_consumption()

world_tab, regions_tab, renewables_tab = st.tabs(
    ["Mundo", "Comparación regional", "Renovables"]
)

with world_tab:
    render_world_tab(supply_df, consumption_df)

with regions_tab:
    render_region_tab(supply_df, consumption_df)

with renewables_tab:
    render_renewables_tab(renewables_df)
