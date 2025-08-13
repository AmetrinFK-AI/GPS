# streamlit_app.py
# Только OpenStreetMap. После загрузки файл проигрывается ДИНАМИЧЕСКИ:
# маршрут растёт "от точки к точке" с нумерацией и чёткими маркерами.
# CSV из твоего файла: ['timestamp','latlng','odometer'] — latlng -> lat, lon.

import io
import re
import json
import math
import time
import hashlib
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import AntPath, PolyLineTextPath
from streamlit_folium import st_folium

# ---------------------------
# Вспомогательные функции
# ---------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def _clean_quotes(s: str) -> str:
    return re.sub(r'["\']', '', str(s)).strip()

def _parse_timestamp_series(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() >= max(3, int(0.5 * len(s))):
        med = float(num.dropna().median())
        unit = "s" if med < 1e12 else "ms"
        return pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if ts.notna().any() and (ts.dt.year.dropna().median() < 1980) and num.notna().any():
        med = float(num.dropna().median())
        unit = "s" if med < 1e12 else "ms"
        ts = pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
    return ts

def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    cols = {c.lower().strip(): c for c in df.columns}
    lat = next((cols[c] for c in ["lat", "latitude", "y"] if c in cols), None)
    lon = next((cols[c] for c in ["lon", "lng", "long", "longitude", "x"] if c in cols), None)
    latlng_col = cols.get("latlng")
    time_col = next((cols[c] for c in ["timestamp", "time", "datetime", "date"] if c in cols), None)
    elevation_col = next((cols[c] for c in ["elevation", "alt", "altitude", "ele", "height"] if c in cols), None)
    odo_col = next((cols[c] for c in ["odometer", "odo", "distance"] if c in cols), None)
    return lat, lon, latlng_col, time_col, elevation_col, odo_col

def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    df = pd.read_csv(buf, sep=None, engine="python")
    lat, lon, latlng_col, time_col, elevation_col, odo_col = detect_columns(df)

    # latlng -> lat/lon
    if latlng_col is not None and (lat is None or lon is None):
        latlon = df[latlng_col].astype(str).str.split(",", n=1, expand=True)
        df["__lat"] = pd.to_numeric(latlon[0].map(_clean_quotes), errors="coerce")
        df["__lon"] = pd.to_numeric(latlon[1].map(_clean_quotes), errors="coerce")
        lat, lon = "__lat", "__lon"

    if lat is None or lon is None:
        raise ValueError("Не найдены координаты. Нужны lat/lon или одна колонка latlng.")

    out = pd.DataFrame({
        "lat": pd.to_numeric(df[lat], errors="coerce"),
        "lon": pd.to_numeric(df[lon], errors="coerce"),
    })

    if time_col is not None:
        out["timestamp"] = _parse_timestamp_series(df[time_col])
    else:
        out["timestamp"] = pd.NaT

    out["elevation"] = pd.to_numeric(df[elevation_col], errors="coerce") if elevation_col else np.nan
    out["odometer"]  = pd.to_numeric(df[odo_col], errors="coerce") if odo_col else np.nan
    return out

def parse_gpx(file_bytes: bytes) -> pd.DataFrame:
    import gpxpy
    gpx = gpxpy.parse(io.StringIO(file_bytes.decode("utf-8", errors="ignore")))
    rows = []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                rows.append({
                    "lat": p.latitude, "lon": p.longitude,
                    "timestamp": pd.to_datetime(p.time, utc=True) if p.time else pd.NaT,
                    "elevation": p.elevation if p.elevation is not None else np.nan,
                    "odometer": np.nan,
                })
    for w in gpx.waypoints:
        rows.append({
            "lat": w.latitude, "lon": w.longitude,
            "timestamp": pd.to_datetime(getattr(w, "time", None), utc=True, errors="coerce") if getattr(w, "time", None) else pd.NaT,
            "elevation": getattr(w, "elevation", np.nan) if getattr(w, "elevation", None) is not None else np.nan,
            "odometer": np.nan,
        })
    if not rows:
        raise ValueError("В GPX не найдено точек.")
    return pd.DataFrame(rows)

def parse_geojson(file_bytes: bytes) -> pd.DataFrame:
    geo = json.loads(file_bytes.decode("utf-8", errors="ignore"))
    rows = []
    def add_point(coord, props):
        lon, lat = coord[0], coord[1]
        t = pd.NaT; elev = np.nan
        if isinstance(props, dict):
            if "time" in props: t = pd.to_datetime(props["time"], errors="coerce", utc=True)
            if "timestamp" in props: t = pd.to_datetime(props["timestamp"], errors="coerce", utc=True)
            if "elevation" in props: elev = pd.to_numeric(props["elevation"], errors="coerce")
        rows.append({"lat": lat, "lon": lon, "timestamp": t, "elevation": elev, "odometer": np.nan})
    def handle_geom(geom, props):
        if geom["type"] == "Point":
            add_point(geom["coordinates"], props)
        elif geom["type"] == "LineString":
            for c in geom["coordinates"]: add_point(c, props)
        elif geom["type"] == "MultiLineString":
            for line in geom["coordinates"]:
                for c in line: add_point(c, props)
    if geo.get("type") == "FeatureCollection":
        for f in geo["features"]: handle_geom(f["geometry"], f.get("properties", {}))
    elif geo.get("type") in ("Feature",):
        handle_geom(geo["geometry"], geo.get("properties", {}))
    else:
        handle_geom(geo, {})
    if not rows: raise ValueError("В GeoJSON не получилось извлечь точки.")
    return pd.DataFrame(rows)

def preprocess_track(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))].copy()
    if df["timestamp"].notna().any():
        df = df.sort_values(by=["timestamp"], kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    return df

def compute_metrics(df: pd.DataFrame):
    n = len(df)
    seg_dist_km = np.zeros(n); seg_dt_h = np.zeros(n)
    for i in range(1, n):
        seg_dist_km[i] = haversine_km(df.at[i-1,"lat"], df.at[i-1,"lon"], df.at[i,"lat"], df.at[i,"lon"])
        t1, t2 = df.at[i-1,"timestamp"], df.at[i,"timestamp"]
        if pd.notna(t1) and pd.notna(t2):
            seg_dt_h[i] = max((t2 - t1).total_seconds(), 0) / 3600.0
    df["seg_dist_km"] = seg_dist_km
    df["seg_dt_h"] = seg_dt_h
    with np.errstate(divide="ignore", invalid="ignore"):
        df["speed_kmh"] = np.where(seg_dt_h > 0, seg_dist_km / seg_dt_h, np.nan)
    total_km = float(np.nansum(seg_dist_km))
    duration_h = float(np.nansum(seg_dt_h)) if np.isfinite(np.nansum(seg_dt_h)) else np.nan
    avg_speed = float(total_km / duration_h) if duration_h and duration_h > 0 else np.nan
    max_speed = float(np.nanmax(df["speed_kmh"])) if np.isfinite(np.nanmax(df["speed_kmh"])) else np.nan
    elev_gain = elev_loss = np.nan
    if df["elevation"].notna().any():
        de = np.diff(df["elevation"].astype(float).to_numpy())
        elev_gain = float(np.sum(de[de > 0])) if de.size else np.nan
        elev_loss = float(-np.sum(de[de < 0])) if de.size else np.nan
    return {
        "points": int(n),
        "total_km": total_km,
        "duration_h": duration_h,
        "avg_speed_kmh": avg_speed,
        "max_speed_kmh": max_speed,
        "elev_gain": elev_gain,
        "elev_loss": elev_loss,
    }, df

# Номерованные маркеры (ярко для уже «пройденных», приглушённо — для будущих)
def numbered_marker(lat, lon, num, active=True, is_start=False, is_end=False, ts=None):
    bg = "#2ecc71" if is_start else ("#e74c3c" if is_end else "#1976d2")
    if not active:
        bg = "#9e9e9e"
    html = f"""
    <div style="
        background:{bg};
        color:#fff;
        border:2px solid #000;
        border-radius:50%;
        width:26px;height:26px;
        line-height:22px;
        text-align:center;
        font-weight:700;
        font-size:13px;
        box-shadow:0 0 0 2px rgba(255,255,255,0.95);
    ">{num}</div>
    """
    tip = f"#{num}"
    if ts is not None and pd.notna(ts):
        tip += f" • {pd.to_datetime(ts).isoformat()}"
    return folium.Marker(
        [lat, lon],
        icon=folium.DivIcon(html=html, icon_size=(26,26), icon_anchor=(13,13)),
        tooltip=tip
    )

# ---------------------------
# Состояние проигрывания
# ---------------------------
def reset_state(df: pd.DataFrame, file_sig: str):
    st.session_state.file_sig = file_sig
    st.session_state.df = df
    st.session_state.N = len(df)
    st.session_state.k = 1  # начнём с первой точки
    st.session_state.play = True  # автостарт
    st.session_state.step_points = st.session_state.get("step_points", 1)
    st.session_state.interval_ms = st.session_state.get("interval_ms", 200)

st.set_page_config(page_title="Маршрут по точкам — динамическая прорисовка", layout="wide")
st.title("🗺️ Маршрут по точкам — динамическая прорисовка (OpenStreetMap)")

with st.sidebar:
    st.header("Файл")
    file = st.file_uploader("Загрузите CSV/GPX/GeoJSON", type=["csv","gpx","geojson","json"])
    st.header("Проигрывание")
    step_points = st.number_input("Точек за тик", 1, 100, st.session_state.get("step_points", 1))
    interval_ms = st.slider("Интервал между тиками, мс", 50, 1500, st.session_state.get("interval_ms", 200), 10)
    colb1, colb2, colb3 = st.columns(3)
    if colb1.button("⏵ Пуск"):
        st.session_state.play = True
    if colb2.button("⏸ Пауза"):
        st.session_state.play = False
    if colb3.button("⏮ Сброс") and "df" in st.session_state:
        st.session_state.k = 1
        st.session_state.play = True
    st.session_state.step_points = int(step_points)
    st.session_state.interval_ms = int(interval_ms)

if not file and "df" not in st.session_state:
    st.info("Загрузите файл — маршрут будет прорисовываться постепенно от точки к точке.")
    st.stop()

# Парсинг файла и инициализация состояния
if file:
    raw = file.read()
    file_sig = hashlib.md5(raw + file.name.encode("utf-8")).hexdigest()
    if st.session_state.get("file_sig") != file_sig:
        ext = file.name.lower().split(".")[-1]
        with st.spinner("Разбор файла…"):
            if ext == "csv":
                df = parse_csv(raw)
            elif ext == "gpx":
                df = parse_gpx(raw)
            elif ext in ("geojson", "json"):
                df = parse_geojson(raw)
            else:
                df = parse_csv(raw)
            df = preprocess_track(df)
            if len(df) < 2:
                st.error("Недостаточно точек.")
                st.stop()
            reset_state(df, file_sig)

# Данные из состояния
df = st.session_state.df
N  = st.session_state.N
k  = st.session_state.k

# Текущая подвыборка (то, что уже «пройдено»)
df_now = df.iloc[:max(1, k)].copy()
metrics, df_now = compute_metrics(df_now)

# Панель метрик с прогрессом
progress = k / N
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Точек показано", f"{k}/{N}")
c2.metric("Пройдено", f"{metrics['total_km']:.2f} км")
if not math.isnan(metrics["duration_h"]):
    h = int(metrics["duration_h"]); m = int(round((metrics["duration_h"] - h)*60))
    c3.metric("В пути", f"{h} ч {m} м")
else:
    c3.metric("В пути", "—")
c4.metric("Ср. скорость", "—" if math.isnan(metrics["avg_speed_kmh"]) else f"{metrics['avg_speed_kmh']:.1f} км/ч")
c5.progress(progress)

# Карта (только OpenStreetMap)
center = [float(df_now["lat"].mean()), float(df_now["lon"].mean())]
m = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=13, control_scale=True)

coords_all = df[["lat","lon"]].to_numpy().tolist()
coords_now = df_now[["lat","lon"]].to_numpy().tolist()

# Линия уже пройденного участка (яркая) + направление
if len(coords_now) >= 2:
    poly = folium.PolyLine(coords_now, weight=6, opacity=0.95, color="#1976d2", tooltip="Маршрут (текущее)")
    poly.add_to(m)
    PolyLineTextPath(poly, "▶", repeat=True, offset=8,
                     attributes={"fill": "#1976d2", "font-weight": "bold", "font-size": "16"}).add_to(m)

# Оставшийся маршрут (тонкая серо-пунктирная подсказка)
if k < N:
    folium.PolyLine(coords_all[k-1:], weight=3, opacity=0.6, color="#9e9e9e", dash_array="6,8",
                    tooltip="Маршрут (дальше)").add_to(m)

# Пронумерованные маркеры: активные до k — яркие, после — приглушённые
for i, (lat, lon) in enumerate(coords_all, start=1):
    ts = df.at[i-1, "timestamp"]
    numbered_marker(
        lat, lon, i,
        active=(i <= k),
        is_start=(i == 1),
        is_end=(i == N),
        ts=ts
    ).add_to(m)

# Анимация линии (поверх «текущей» части) — опционально
if len(coords_now) >= 2:
    AntPath(coords_now, delay=600, dash_array=[10, 16]).add_to(m)

st_folium(m, height=660, returned_objects=[])

# Такт проигрывания: если play=True и не дошли до конца — увеличиваем k и перерисовываем
if st.session_state.play and st.session_state.k < N:
    st.session_state.k = min(N, st.session_state.k + st.session_state.step_points)
    time.sleep(st.session_state.interval_ms / 1000.0)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

with st.expander("Первые строки данных (сортированы по времени, если есть)"):
    st.dataframe(df.head(50))
