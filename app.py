from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="sieLAC Global Energy Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_PATH = Path(__file__).resolve().parent

import glob as glob_module

def _find_csv(pattern: str) -> str:
    """Find CSV file using glob pattern to handle encoding issues."""
    matches = glob_module.glob(str(BASE_PATH / pattern))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No file matching pattern: {pattern}")

DATA_PATHS = {
    "supply": lambda: _find_csv("Oferta y demanda */*/Oferta total de energ*.csv"),
    "consumption": lambda: _find_csv("Oferta y demanda */*Todos*/ConsumoFinal.csv"),
    "renewables_consumption": lambda: _find_csv("Oferta y demanda */*Renovables*/Renovables-Consumo final.csv"),
    "biofuel_production": lambda: _find_csv("Oferta y demanda */*Renovables*/Renovables-Producci*n de biocombustibles.csv"),
    "biomass_production": lambda: _find_csv("Oferta y demanda */*Renovables*/Renovables-Producci*n de biomasa.csv"),
    "coal_consumption": lambda: _find_csv("Oferta y demanda */*Carb*n mineral*/CosumoFinal.csv"),
    "coal_production": lambda: _find_csv("Oferta y demanda */*Carb*n mineral*/Producci*.csv"),
    "electricity_consumption": lambda: _find_csv("Oferta y demanda */*El*ctrico*/Consumo Final.csv"),
    "electricity_generation": lambda: _find_csv("Oferta y demanda */*El*ctrico*/Generaci*n el*ctrica por fuente.csv"),
    "hydrocarbon_consumption": lambda: _find_csv("Oferta y demanda */*Hidrocarburos*/Hidrocarburos-Consumo final.csv"),
    "hydrocarbon_production": lambda: _find_csv("Oferta y demanda */*Hidrocarburos*/Hidrocarburos-Producci*.csv"),
    "electrical_capacity": lambda: _find_csv("Infraestructura */*El*ctrico*/Capacidad instalada por fuente.csv"),
    "refining_capacity": lambda: _find_csv("Infraestructura */*Hidrocarburos*/Consolidated Refining Capacity*.csv"),
}

def get_data_path(key: str) -> str:
    """Get the actual path for a data file."""
    path_func = DATA_PATHS.get(key)
    if path_func:
        return path_func()
    raise KeyError(f"Unknown data path key: {key}")

DATA_SOURCES = {
    "primary": "sieLAC-OLADE (Sistema de Información Energética de Latinoamérica y el Caribe)",
    "secondary": "BP p.l.c. Statistical Review of World Energy",
    "note_lac": "Para América Latina y el Caribe a partir del 2000, los datos corresponden a recopilaciones directas de OLADE",
    "last_update": "Julio de 2025",
    "coverage": "1970-2023 (datos históricos), 2023 (datos actuales)",
}

UNIT_DESCRIPTIONS = {
    "10⁶ tep": "Millones de toneladas equivalentes de petróleo",
    "10³ bep": "Miles de barriles equivalentes de petróleo",
    "10³ t": "Miles de toneladas",
    "GWh": "Gigawatts-hora",
    "MW": "Megawatts",
    "10⁶ m³": "Millones de metros cúbicos",
    "10³ bbl": "Miles de barriles",
    "10³ bbl/d": "Miles de barriles por día",
}


def _strip_quotes(value: str | int | float) -> str:
    text = str(value)
    return text.strip().strip("'\"")


def _to_number(value: str | float | int | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    text = _strip_quotes(text)
    text = text.replace("\u202f", "").replace("\ufeff", "").replace(" ", "")
    if text == "":
        return None
    if text.count(",") == 1 and text.count(".") == 0:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _clean_columns(columns: Iterable[str]) -> List[str]:
    return [_strip_quotes(col) for col in columns]


def _clean_dataframe(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df.columns = _clean_columns(df.columns)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: _strip_quotes(x) if pd.notna(x) else x)
    if numeric_cols:
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(_to_number)
    return df


@st.cache_data
def load_energy_supply() -> pd.DataFrame:
    df = pd.read_csv(get_data_path("supply"))
    df = _clean_dataframe(df)
    rename_map = {
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
    df = pd.read_csv(get_data_path("consumption"))
    df = _clean_dataframe(df)
    df = df.rename(columns={"Energía (Cantidad 10⁶ tep)": "Energía"})
    df["Energía"] = df["Energía"].apply(_to_number)
    return df


@st.cache_data
def load_renewables_consumption() -> pd.DataFrame:
    df = pd.read_csv(get_data_path("renewables_consumption"))
    df = _clean_dataframe(df)
    rename_map = {
        "Otros (Consumo 10³ bep)": "Otros",
        "Otra biomasa (Consumo 10³ bep)": "Otra biomasa",
    }
    df = df.rename(columns=rename_map)
    numeric_cols = [col for col in df.columns if col not in {"Año", "Region"}]
    for col in numeric_cols:
        df[col] = df[col].apply(_to_number)
    return df


@st.cache_data
def load_biofuel_production() -> pd.DataFrame:
    df = pd.read_csv(get_data_path("biofuel_production"))
    df = _clean_dataframe(df)
    df = df.rename(columns={"Biocombustibles (Cantidad 10³ bep)": "Biocombustibles"})
    df["Biocombustibles"] = df["Biocombustibles"].apply(_to_number)
    return df


@st.cache_data
def load_biomass_production() -> pd.DataFrame:
    df = pd.read_csv(get_data_path("biomass_production"))
    df = _clean_dataframe(df)
    df = df.rename(columns={"Otra Biomasa (Cantidad 10³ bep)": "Biomasa"})
    df["Biomasa"] = df["Biomasa"].apply(_to_number)
    return df


@st.cache_data
def load_coal_consumption() -> pd.DataFrame:
    df = pd.read_csv(get_data_path("coal_consumption"))
    df = _clean_dataframe(df)
    df = df.rename(columns={"Carbón mineral (Consumo 10³ t)": "Consumo"})
    df["Consumo"] = df["Consumo"].apply(_to_number)
    return df


@st.cache_data
def load_coal_production() -> pd.DataFrame:
    df = pd.read_csv(get_data_path("coal_production"))
    df = _clean_dataframe(df)
    df = df.rename(columns={"Carbón mineral  (Cantidad 10³ t)": "Producción"})
    df["Producción"] = df["Producción"].apply(_to_number)
    return df


@st.cache_data
def load_electricity_consumption() -> pd.DataFrame:
    df = pd.read_csv(get_data_path("electricity_consumption"))
    df = _clean_dataframe(df)
    df["Electricidad"] = df["Electricidad"].apply(_to_number)
    return df


@st.cache_data
def load_electricity_generation() -> pd.DataFrame:
    df = pd.read_csv(get_data_path("electricity_generation"))
    df = _clean_dataframe(df)
    region_cols = [c for c in df.columns if c not in {"Año", "Descripción", "Unidad"}]
    for col in region_cols:
        df[col] = df[col].apply(_to_number)
    return df


@st.cache_data
def load_hydrocarbon_consumption() -> pd.DataFrame:
    df = pd.read_csv(get_data_path("hydrocarbon_consumption"))
    df = _clean_dataframe(df)
    rename_map = {
        "Gas natural (Consumo 10⁶ m³)": "Gas natural",
        "Fuel oil (Consumo 10³ bbl)": "Fuel oil",
        "Gasolina/alcohol (Consumo 10³ bbl)": "Gasolina/alcohol",
        "Petróleo (Consumo 10³ bbl)": "Petróleo",
        "Otros (Consumo 10³ bep)": "Otros",
        "Destilados medios (Consumo 10³ bep)": "Destilados medios",
    }
    df = df.rename(columns=rename_map)
    numeric_cols = [c for c in df.columns if c not in {"Año", "Region"}]
    for col in numeric_cols:
        df[col] = df[col].apply(_to_number)
    return df


@st.cache_data
def load_hydrocarbon_production() -> pd.DataFrame:
    df = pd.read_csv(get_data_path("hydrocarbon_production"))
    df = _clean_dataframe(df)
    rename_map = {
        "Gas natural (Cantidad 10⁶ m³)": "Gas natural",
        "Petróleo (Cantidad 10³ bbl)": "Petróleo",
    }
    df = df.rename(columns=rename_map)
    numeric_cols = [c for c in df.columns if c not in {"Año", "Region"}]
    for col in numeric_cols:
        df[col] = df[col].apply(_to_number)
    return df


@st.cache_data
def load_electrical_capacity() -> pd.DataFrame:
    df = pd.read_csv(get_data_path("electrical_capacity"))
    df = _clean_dataframe(df)
    region_cols = [c for c in df.columns if c not in {"Año", "Descripción", "Unidad"}]
    for col in region_cols:
        df[col] = df[col].apply(_to_number)
    return df


@st.cache_data
def load_refining_capacity() -> pd.DataFrame:
    df = pd.read_csv(get_data_path("refining_capacity"))
    df = _clean_dataframe(df)
    df["Refining Capacity (10^3 bbl/d)"] = df["Refining Capacity (10^3 bbl/d)"].apply(_to_number)
    return df


def render_sidebar():
    with st.sidebar:
        st.header("Fuentes y Metodología")
        st.markdown("---")
        st.subheader("Fuentes de Datos")
        st.markdown(f"**Principal:** {DATA_SOURCES['primary']}")
        st.markdown(f"**Secundaria:** {DATA_SOURCES['secondary']}")
        st.markdown("---")
        st.subheader("Notas Metodológicas")
        st.info(DATA_SOURCES['note_lac'])
        st.markdown("---")
        st.subheader("Cobertura Temporal")
        st.markdown(f"**{DATA_SOURCES['coverage']}**")
        st.markdown(f"*Última actualización: {DATA_SOURCES['last_update']}*")
        st.markdown("---")
        st.subheader("Unidades de Medida")
        for unit, desc in UNIT_DESCRIPTIONS.items():
            st.markdown(f"**{unit}:** {desc}")
        st.markdown("---")
        st.subheader("Regiones Cubiertas")
        regions = [
            "África", "América Latina y el Caribe", "Asia & Australia",
            "Europa", "Ex URSS", "Medio Oriente", "Norte América",
            "Mundo", "OCDE", "China", "EE.UU."
        ]
        for r in regions:
            st.markdown(f"• {r}")


def render_executive_summary(supply_df, consumption_df, elec_gen_df, refining_df):
    st.subheader("Resumen Ejecutivo Global (2023)")
    
    world_supply = supply_df[supply_df["Region"] == "Mundo"]
    world_consumption = consumption_df[consumption_df["Region"] == "Mundo"]["Energía"].iloc[0]
    
    supply_cols = [c for c in supply_df.columns if c not in {"Region", "Oferta total"}]
    total_supply = world_supply[supply_cols].sum(axis=1).iloc[0]
    renewables = world_supply[["Hidroenergía", "Otras renovables"]].sum(axis=1).iloc[0]
    fossil_fuels = world_supply[["Carbón mineral", "Gas natural", "Petróleo"]].sum(axis=1).iloc[0]
    nuclear = world_supply["Nuclear"].iloc[0]
    
    renewables_pct = (renewables / total_supply * 100) if total_supply else 0
    fossil_pct = (fossil_fuels / total_supply * 100) if total_supply else 0
    nuclear_pct = (nuclear / total_supply * 100) if total_supply else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Oferta Total Mundial", f"{total_supply:,.0f} 10⁶ tep", 
                help="Oferta total de energía primaria a nivel mundial")
    col2.metric("Consumo Final Mundial", f"{world_consumption:,.0f} 10⁶ tep",
                help="Consumo final de energía a nivel mundial")
    col3.metric("Participación Renovables", f"{renewables_pct:.1f}%",
                help="Hidroenergía + Otras renovables")
    col4.metric("Participación Fósiles", f"{fossil_pct:.1f}%",
                help="Carbón + Gas natural + Petróleo")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        supply_data = world_supply[supply_cols].T.reset_index()
        supply_data.columns = ["Fuente", "Valor"]
        supply_data["Valor"] = supply_data["Valor"].apply(lambda x: x if pd.notna(x) else 0)
        
        fig = px.pie(
            supply_data,
            values="Valor",
            names="Fuente",
            title="Matriz Energética Mundial 2023",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        regional_supply = supply_df[~supply_df["Region"].isin(["Mundo", "OCDE", "China", "EE.UU."])]
        regional_supply = regional_supply.sort_values("Oferta total", ascending=True)
        
        fig = px.bar(
            regional_supply,
            x="Oferta total",
            y="Region",
            orientation='h',
            title="Oferta Energética por Región (10⁶ tep)",
            color="Oferta total",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Análisis Comparativo Regional")
    
    supply_long = supply_df[~supply_df["Region"].isin(["Mundo", "OCDE"])].melt(
        id_vars=["Region"],
        value_vars=supply_cols,
        var_name="Fuente",
        value_name="Energía"
    )
    
    fig = px.bar(
        supply_long,
        x="Region",
        y="Energía",
        color="Fuente",
        title="Composición de la Oferta Energética por Región y Fuente (10⁶ tep)",
        barmode="stack",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def render_historical_trends(refining_df):
    st.subheader("Tendencias Históricas: Capacidad de Refinación (1970-2023)")
    
    st.info("📊 **54 años de datos históricos** - Esta serie temporal muestra la evolución de la capacidad de refinación petrolera mundial desde 1970.")
    
    regions = refining_df["Region"].unique().tolist()
    default_regions = ["Latin America & Caribbean", "North America", "Europe", "Asia & Australia", "Middle East"]
    default_regions = [r for r in default_regions if r in regions]
    
    selected_regions = st.multiselect(
        "Selecciona regiones para analizar",
        options=regions,
        default=default_regions[:5],
        key="refining_regions"
    )
    
    if selected_regions:
        filtered = refining_df[refining_df["Region"].isin(selected_regions)]
        
        fig = px.line(
            filtered,
            x="Year",
            y="Refining Capacity (10^3 bbl/d)",
            color="Region",
            title="Evolución de la Capacidad de Refinación por Región (10³ bbl/día)",
            markers=True
        )
        fig.update_layout(
            xaxis_title="Año",
            yaxis_title="Capacidad (10³ bbl/día)",
            legend_title="Región",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            latest = filtered[filtered["Year"] == 2023].sort_values(
                "Refining Capacity (10^3 bbl/d)", ascending=False
            )
            fig = px.bar(
                latest,
                x="Region",
                y="Refining Capacity (10^3 bbl/d)",
                title="Capacidad de Refinación 2023 por Región",
                color="Refining Capacity (10^3 bbl/d)",
                color_continuous_scale="Blues"
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            start_year = filtered[filtered["Year"] == 1970].set_index("Region")["Refining Capacity (10^3 bbl/d)"]
            end_year = filtered[filtered["Year"] == 2023].set_index("Region")["Refining Capacity (10^3 bbl/d)"]
            
            growth = ((end_year - start_year) / start_year * 100).reset_index()
            growth.columns = ["Region", "Crecimiento (%)"]
            growth = growth.dropna().sort_values("Crecimiento (%)", ascending=False)
            
            fig = px.bar(
                growth,
                x="Region",
                y="Crecimiento (%)",
                title="Crecimiento de Capacidad 1970-2023 (%)",
                color="Crecimiento (%)",
                color_continuous_scale="RdYlGn"
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Datos Detallados")
        pivot = filtered.pivot(index="Year", columns="Region", values="Refining Capacity (10^3 bbl/d)")
        st.dataframe(pivot, use_container_width=True, height=300)


def render_electricity_tab(elec_gen_df, elec_cons_df, elec_cap_df):
    st.subheader("Sector Eléctrico: Generación, Consumo e Infraestructura")
    
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Generación por Fuente", "Consumo Regional", "Capacidad Instalada"])
    
    with sub_tab1:
        st.markdown("### Generación Eléctrica por Fuente (2023)")
        
        region_cols = [c for c in elec_gen_df.columns if c not in {"Año", "Descripción", "Unidad"}]
        gen_data = elec_gen_df[elec_gen_df["Descripción"] != "Total"].copy()
        
        selected_region = st.selectbox(
            "Selecciona una región para análisis detallado",
            options=region_cols,
            index=region_cols.index("Mundo") if "Mundo" in region_cols else 0
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            region_data = gen_data[["Descripción", selected_region]].copy()
            region_data.columns = ["Fuente", "GWh"]
            region_data = region_data.dropna()
            region_data = region_data[region_data["GWh"] > 0]
            
            fig = px.pie(
                region_data,
                values="GWh",
                names="Fuente",
                title=f"Mix de Generación: {selected_region}",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            region_data_sorted = region_data.sort_values("GWh", ascending=True)
            fig = px.bar(
                region_data_sorted,
                x="GWh",
                y="Fuente",
                orientation='h',
                title=f"Generación por Fuente: {selected_region} (GWh)",
                color="GWh",
                color_continuous_scale="Greens"
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Comparación Multi-Regional")
        
        compare_regions = st.multiselect(
            "Selecciona regiones para comparar",
            options=region_cols,
            default=["América Latina y el Caribe", "Europa", "Norte América", "Asia & Australia"][:4],
            key="elec_compare"
        )
        
        if compare_regions:
            compare_data = gen_data[["Descripción"] + compare_regions].copy()
            compare_melted = compare_data.melt(
                id_vars=["Descripción"],
                value_vars=compare_regions,
                var_name="Región",
                value_name="GWh"
            )
            compare_melted = compare_melted.dropna()
            
            fig = px.bar(
                compare_melted,
                x="Región",
                y="GWh",
                color="Descripción",
                title="Generación Eléctrica por Fuente y Región (GWh)",
                barmode="stack"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with sub_tab2:
        st.markdown("### Consumo Eléctrico por Región (2023)")
        
        elec_cons_sorted = elec_cons_df.sort_values("Electricidad", ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            regional = elec_cons_sorted[~elec_cons_sorted["Region"].isin(["Mundo", "OCDE"])]
            fig = px.bar(
                regional,
                x="Region",
                y="Electricidad",
                title="Consumo Eléctrico por Región (GWh)",
                color="Electricidad",
                color_continuous_scale="Oranges"
            )
            fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            total_world = elec_cons_df[elec_cons_df["Region"] == "Mundo"]["Electricidad"].iloc[0]
            regional_share = elec_cons_sorted[~elec_cons_sorted["Region"].isin(["Mundo", "OCDE", "China", "EE.UU."])].copy()
            regional_share["Participación (%)"] = regional_share["Electricidad"] / total_world * 100
            
            fig = px.pie(
                regional_share,
                values="Participación (%)",
                names="Region",
                title="Participación en Consumo Mundial (%)",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Datos Completos")
        st.dataframe(elec_cons_df, use_container_width=True)
    
    with sub_tab3:
        st.markdown("### Capacidad Instalada por Fuente (2023)")
        
        cap_data = elec_cap_df[elec_cap_df["Descripción"] != "Total"].copy()
        region_cols = [c for c in cap_data.columns if c not in {"Año", "Descripción", "Unidad"}]
        
        for region in region_cols:
            cap_data[region] = cap_data[region].apply(_to_number)
        
        cap_melted = cap_data.melt(
            id_vars=["Descripción"],
            value_vars=region_cols,
            var_name="Región",
            value_name="MW"
        )
        cap_melted = cap_melted.dropna()
        
        fig = px.bar(
            cap_melted,
            x="Descripción",
            y="MW",
            color="Región",
            title="Capacidad Instalada por Tipo de Fuente (MW)",
            barmode="group"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Datos de Capacidad")
        st.dataframe(elec_cap_df, use_container_width=True)


def render_fossil_fuels_tab(coal_prod_df, coal_cons_df, hydro_prod_df, hydro_cons_df):
    st.subheader("Combustibles Fósiles: Carbón e Hidrocarburos")
    
    sub_tab1, sub_tab2 = st.tabs(["Carbón Mineral", "Hidrocarburos"])
    
    with sub_tab1:
        st.markdown("### Análisis del Carbón Mineral (2023)")
        
        coal_merged = coal_prod_df.merge(coal_cons_df, on=["Año", "Region"], how="outer")
        coal_merged["Balance"] = coal_merged["Producción"].fillna(0) - coal_merged["Consumo"].fillna(0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            regional = coal_merged[~coal_merged["Region"].isin(["Mundo", "OCDE"])].copy()
            regional = regional.sort_values("Producción", ascending=False)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Producción",
                x=regional["Region"],
                y=regional["Producción"],
                marker_color="darkblue"
            ))
            fig.add_trace(go.Bar(
                name="Consumo",
                x=regional["Region"],
                y=regional["Consumo"],
                marker_color="darkred"
            ))
            fig.update_layout(
                title="Producción vs Consumo de Carbón (10³ t)",
                barmode="group",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            balance_data = regional[["Region", "Balance"]].sort_values("Balance", ascending=False)
            colors = ["green" if x > 0 else "red" for x in balance_data["Balance"]]
            
            fig = px.bar(
                balance_data,
                x="Region",
                y="Balance",
                title="Balance Neto (Producción - Consumo)",
                color="Balance",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Top Productores y Consumidores")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 5 Productores**")
            top_prod = coal_prod_df.nlargest(5, "Producción")[["Region", "Producción"]]
            st.dataframe(top_prod, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Top 5 Consumidores**")
            top_cons = coal_cons_df.nlargest(5, "Consumo")[["Region", "Consumo"]]
            st.dataframe(top_cons, use_container_width=True, hide_index=True)
    
    with sub_tab2:
        st.markdown("### Análisis de Hidrocarburos (2023)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Producción de Petróleo (10³ bbl)")
            oil_prod = hydro_prod_df[~hydro_prod_df["Region"].isin(["Mundo", "OCDE"])].copy()
            oil_prod = oil_prod.sort_values("Petróleo", ascending=False)
            
            fig = px.bar(
                oil_prod,
                x="Region",
                y="Petróleo",
                title="Producción de Petróleo por Región",
                color="Petróleo",
                color_continuous_scale="YlOrRd"
            )
            fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Producción de Gas Natural (10⁶ m³)")
            gas_prod = hydro_prod_df[~hydro_prod_df["Region"].isin(["Mundo", "OCDE"])].copy()
            gas_prod = gas_prod.sort_values("Gas natural", ascending=False)
            
            fig = px.bar(
                gas_prod,
                x="Region",
                y="Gas natural",
                title="Producción de Gas Natural por Región",
                color="Gas natural",
                color_continuous_scale="Blues"
            )
            fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Consumo de Hidrocarburos por Tipo")
        hydro_regional = hydro_cons_df[~hydro_cons_df["Region"].isin(["Mundo", "OCDE"])].copy()
        
        fuel_cols = ["Gas natural", "Petróleo", "Fuel oil", "Gasolina/alcohol", "Otros", "Destilados medios"]
        available_cols = [c for c in fuel_cols if c in hydro_cons_df.columns]
        
        hydro_melted = hydro_regional.melt(
            id_vars=["Region"],
            value_vars=available_cols,
            var_name="Tipo",
            value_name="Consumo"
        )
        hydro_melted = hydro_melted.dropna()
        hydro_melted = hydro_melted[hydro_melted["Consumo"] > 0]
        
        fig = px.bar(
            hydro_melted,
            x="Region",
            y="Consumo",
            color="Tipo",
            title="Consumo de Hidrocarburos por Tipo y Región",
            barmode="stack"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


def render_renewables_tab(renewables_df, biofuel_df, biomass_df):
    st.subheader("Energías Renovables y Biocombustibles")
    
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Consumo Renovable", "Producción de Biocombustibles", "Producción de Biomasa"])
    
    with sub_tab1:
        st.markdown("### Consumo de Energías Renovables (2023)")
        st.caption("Cifras en miles de barriles equivalentes de petróleo (10³ bep)")
        
        regions = renewables_df["Region"].unique().tolist()
        default_regions = [r for r in ["Mundo", "América Latina y el Caribe"] if r in regions]
        
        if renewables_df.shape[0] > 0:
            renew_long = renewables_df.melt(
                id_vars=["Año", "Region"],
                value_vars=["Otros", "Otra biomasa"],
                var_name="Categoría",
                value_name="Consumo"
            )
            renew_long = renew_long.dropna()
            
            fig = px.bar(
                renew_long,
                x="Region",
                y="Consumo",
                color="Categoría",
                title="Consumo Renovable por Categoría",
                barmode="group",
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(renewables_df, use_container_width=True)
    
    with sub_tab2:
        st.markdown("### Producción de Biocombustibles por Región (2023)")
        st.caption("Cifras en miles de barriles equivalentes de petróleo (10³ bep)")
        
        biofuel_regional = biofuel_df[~biofuel_df["Region"].isin(["Mundo", "OCDE"])].copy()
        biofuel_regional = biofuel_regional.sort_values("Biocombustibles", ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                biofuel_regional,
                x="Region",
                y="Biocombustibles",
                title="Producción de Biocombustibles por Región",
                color="Biocombustibles",
                color_continuous_scale="Greens"
            )
            fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            total_world = biofuel_df[biofuel_df["Region"] == "Mundo"]["Biocombustibles"].iloc[0]
            biofuel_regional["Participación (%)"] = biofuel_regional["Biocombustibles"] / total_world * 100
            
            fig = px.pie(
                biofuel_regional,
                values="Participación (%)",
                names="Region",
                title="Participación en Producción Mundial",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Ranking de Productores")
        ranking = biofuel_df.sort_values("Biocombustibles", ascending=False)[["Region", "Biocombustibles"]].copy()
        ranking["Ranking"] = range(1, len(ranking) + 1)
        ranking = ranking[["Ranking", "Region", "Biocombustibles"]]
        st.dataframe(ranking, use_container_width=True, hide_index=True)
    
    with sub_tab3:
        st.markdown("### Producción de Biomasa")
        st.caption("Cifras en miles de barriles equivalentes de petróleo (10³ bep)")
        
        st.info("📊 **Nota:** Los datos de producción de biomasa corresponden al año 2012, siendo los más recientes disponibles para esta categoría.")
        
        if biomass_df.shape[0] > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    biomass_df,
                    x="Region",
                    y="Biomasa",
                    title="Producción de Biomasa por Región (2012)",
                    color="Biomasa",
                    color_continuous_scale="YlGn"
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if "Mundo" in biomass_df["Region"].values:
                    total_world = biomass_df[biomass_df["Region"] == "Mundo"]["Biomasa"].iloc[0]
                    st.metric(
                        "Producción Mundial de Biomasa", 
                        f"{total_world:,.0f} 10³ bep",
                        help="Total mundial de producción de biomasa en miles de barriles equivalentes de petróleo"
                    )
                
                st.markdown("#### Datos de Producción")
                st.dataframe(biomass_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No hay datos de producción de biomasa disponibles.")


def render_regional_comparison(supply_df, consumption_df):
    st.subheader("Comparativo Regional Detallado")
    
    regions = supply_df["Region"].unique().tolist()
    default_regions = [r for r in regions if r not in {"Mundo", "OCDE", "China", "EE.UU."}][:5]
    
    selected_regions = st.multiselect(
        "Selecciona regiones para comparar",
        options=regions,
        default=default_regions,
        key="regional_compare"
    )
    
    if not selected_regions:
        st.warning("Selecciona al menos una región para visualizar datos.")
        return
    
    supply_cols = [c for c in supply_df.columns if c not in {"Region", "Oferta total"}]
    
    col1, col2 = st.columns(2)
    
    with col1:
        supply_filtered = supply_df[supply_df["Region"].isin(selected_regions)]
        supply_long = supply_filtered.melt(
            id_vars=["Region"],
            value_vars=supply_cols,
            var_name="Fuente",
            value_name="Energía"
        )
        
        fig = px.bar(
            supply_long,
            x="Region",
            y="Energía",
            color="Fuente",
            title="Oferta Energética por Fuente (10⁶ tep)",
            barmode="stack"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cons_filtered = consumption_df[consumption_df["Region"].isin(selected_regions)]
        
        fig = px.bar(
            cons_filtered.sort_values("Energía", ascending=True),
            x="Energía",
            y="Region",
            orientation='h',
            title="Consumo Final (10⁶ tep)",
            color="Energía",
            color_continuous_scale="Reds"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Análisis de Matriz Energética")
    
    matrix_data = supply_df[supply_df["Region"].isin(selected_regions)].copy()
    for col in supply_cols:
        matrix_data[f"{col} (%)"] = matrix_data[col] / matrix_data["Oferta total"] * 100
    
    pct_cols = [f"{c} (%)" for c in supply_cols]
    matrix_pct = matrix_data[["Region"] + pct_cols].copy()
    
    fig = px.imshow(
        matrix_pct.set_index("Region")[pct_cols].T,
        labels=dict(x="Región", y="Fuente", color="Participación (%)"),
        title="Matriz de Participación por Fuente (%)",
        color_continuous_scale="YlGnBu",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Datos Detallados")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Oferta por Fuente**")
        st.dataframe(supply_df[supply_df["Region"].isin(selected_regions)], use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Consumo Final**")
        st.dataframe(consumption_df[consumption_df["Region"].isin(selected_regions)], use_container_width=True, hide_index=True)


def main():
    st.title("⚡ sieLAC Global Energy Intelligence Dashboard")
    st.markdown(
        """
        **Panel de inteligencia energética global** construido con datos del Sistema de Información 
        Energética de Latinoamérica y el Caribe (sieLAC-OLADE) y BP Statistical Review of World Energy.
        
        📊 **13 conjuntos de datos** | 📅 **54 años de series históricas (1970-2023)** | 🌍 **11 regiones mundiales**
        """
    )
    
    render_sidebar()
    
    with st.spinner("Cargando datos energéticos globales..."):
        supply_df = load_energy_supply()
        consumption_df = load_final_consumption()
        renewables_df = load_renewables_consumption()
        biofuel_df = load_biofuel_production()
        biomass_df = load_biomass_production()
        coal_cons_df = load_coal_consumption()
        coal_prod_df = load_coal_production()
        elec_cons_df = load_electricity_consumption()
        elec_gen_df = load_electricity_generation()
        hydro_cons_df = load_hydrocarbon_consumption()
        hydro_prod_df = load_hydrocarbon_production()
        elec_cap_df = load_electrical_capacity()
        refining_df = load_refining_capacity()
    
    tabs = st.tabs([
        "📊 Resumen Ejecutivo",
        "📈 Tendencias Históricas",
        "⚡ Sector Eléctrico",
        "🛢️ Combustibles Fósiles",
        "🌿 Renovables",
        "🌍 Comparativo Regional"
    ])
    
    with tabs[0]:
        render_executive_summary(supply_df, consumption_df, elec_gen_df, refining_df)
    
    with tabs[1]:
        render_historical_trends(refining_df)
    
    with tabs[2]:
        render_electricity_tab(elec_gen_df, elec_cons_df, elec_cap_df)
    
    with tabs[3]:
        render_fossil_fuels_tab(coal_prod_df, coal_cons_df, hydro_prod_df, hydro_cons_df)
    
    with tabs[4]:
        render_renewables_tab(renewables_df, biofuel_df, biomass_df)
    
    with tabs[5]:
        render_regional_comparison(supply_df, consumption_df)
    
    st.markdown("---")
    st.caption(
        f"**Fuentes:** {DATA_SOURCES['primary']} | {DATA_SOURCES['secondary']} | "
        f"Última actualización: {DATA_SOURCES['last_update']}"
    )


if __name__ == "__main__":
    main()
