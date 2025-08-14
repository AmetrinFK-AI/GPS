# streamlit_app.py
# Только OpenStreetMap. ВСЕ точки на карте и ЧЁТКАЯ НУМЕРАЦИЯ.
# CSV из твоего файла: ['timestamp', 'latlng', 'odometer'] — latlng -> lat, lon.

import io
import re
import json
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import AntPath, TimestampedGeoJson, PolyLineTextPath
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

def _format_hms(hours: float) -> str:
    if not isinstance(hours, (int, float)) or not np.isfinite(hours):
        return "—"
    total_seconds = int(round(hours * 3600))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:d}:{m:02d}:{s:02d}"

def compute_metrics(df: pd.DataFrame, prefer_source: str = "auto"):
    """
    Расчитывает:
      - seg_dist_km, seg_dt_h, speed_kmh
      - cum_km
      - total_km, duration_h, avg_speed_kmh, max_speed_kmh
    Источник расстояния:
      auto: по одометру, если есть адекватные данные, иначе GPS
      gps: только haversine
      odo: только одометр (с авто-масштабом м→км по медиане)
    """
    n = len(df)
    seg_dt_h = np.zeros(n)
    # время сегмента
    for i in range(1, n):
        t1, t2 = df.at[i-1, "timestamp"], df.at[i, "timestamp"]
        if pd.notna(t1) and pd.notna(t2):
            seg_dt_h[i] = max((t2 - t1).total_seconds(), 0) / 3600.0

    # GPS расстояния
    gps_dist = np.zeros(n)
    for i in range(1, n):
        gps_dist[i] = haversine_km(df.at[i-1,"lat"], df.at[i-1,"lon"], df.at[i,"lat"], df.at[i,"lon"])

    # Одометрические расстояния (как есть)
    odo_dist_raw = np.full(n, np.nan)
    if "odometer" in df.columns and df["odometer"].notna().sum() >= 2:
        od = df["odometer"].astype(float).to_numpy()
        d = np.diff(od, prepend=np.nan)
        # первый сегмент нет значения, отрицательные/слишком большие скачки -> NaN
        d[0] = np.nan
        d[d < 0] = np.nan  # сброс одометра
        odo_dist_raw = d

    # Выбор источника и нормировка одометра (м или км)
    seg_dist_km = gps_dist.copy()
    used_source = "gps"
    if prefer_source in ("odo", "auto") and np.nanmax(odo_dist_raw) > 0 and np.nanmean(odo_dist_raw) > 0:
        # попытка понять единицы по отношению к GPS (медиана по валидным сегментам)
        common_mask = (~np.isnan(odo_dist_raw)) & (gps_dist > 0)
        scale = np.nan
        if np.any(common_mask):
            med_odo = float(np.nanmedian(odo_dist_raw[common_mask]))
            med_gps = float(np.nanmedian(gps_dist[common_mask]))
            if med_gps > 0:
                scale = med_odo / med_gps
        # если похоже на метры (около *1000), делим на 1000
        if np.isfinite(scale) and 100 <= scale <= 2000:
            odo_dist_km = odo_dist_raw / 1000.0
        else:
            odo_dist_km = odo_dist_raw

        if prefer_source == "odo":
            seg_dist_km = np.nan_to_num(odo_dist_km, nan=0.0)
            used_source = "odom"
        else:  # auto: одометр, если доля валидных сегментов приличная
            valid_ratio = np.isfinite(odo_dist_km).sum() / max(1, n)
            if valid_ratio >= 0.5:  # достаточно половины сегментов
                seg_dist_km = np.nan_to_num(odo_dist_km, nan=0.0)
                used_source = "odom"
            else:
                seg_dist_km = gps_dist
                used_source = "gps"
    elif prefer_source == "gps":
        seg_dist_km = gps_dist
        used_source = "gps"

    with np.errstate(divide="ignore", invalid="ignore"):
        speed_kmh = np.where(seg_dt_h > 0, seg_dist_km / seg_dt_h, np.nan)

    df["seg_dist_km"] = seg_dist_km
    df["seg_dt_h"] = seg_dt_h
    df["speed_kmh"] = speed_kmh
    df["cum_km"] = np.cumsum(seg_dist_km)
    df["seg_dt_s"] = (seg_dt_h * 3600).round().astype("int64")

    total_km = float(np.nansum(seg_dist_km))
    duration_h = float(np.nansum(seg_dt_h)) if np.isfinite(np.nansum(seg_dt_h)) else np.nan
    avg_speed = float(total_km / duration_h) if duration_h and duration_h > 0 else np.nan
    max_speed = float(np.nanmax(speed_kmh)) if np.isfinite(np.nanmax(speed_kmh)) else np.nan

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
        "used_source": used_source,
    }, df

def to_geojson_points(df: pd.DataFrame) -> dict:
    feats = []
    for _, r in df.iterrows():
        props = {}
        if pd.notna(r.get("timestamp", pd.NaT)): props["time"] = pd.to_datetime(r["timestamp"]).isoformat()
        if pd.notna(r.get("elevation", np.nan)): props["elevation"] = float(r["elevation"])
        feats.append({"type":"Feature","geometry":{"type":"Point","coordinates":[float(r["lon"]), float(r["lat"])]},"properties":props})
    return {"type":"FeatureCollection","features":feats}

# ---------------------------
# UI (только OpenStreetMap; нумерация и повышенная заметность точек)
# ---------------------------

st.set_page_config(page_title="Маршрут по точкам", layout="wide")
st.title("🗺️ Маршрут по точкам — OpenStreetMap (нумерация точек)")

with st.sidebar:
    st.header("Файл")
    file = st.file_uploader("Загрузите CSV/GPX/GeoJSON", type=["csv","gpx","geojson","json"])
    st.header("Опции")
    dist_source = st.radio(
        "Источник расстояния",
        ["Авто", "GPS", "Одометр"],
        index=0,
        help="Авто: берём одометр, если он полон, иначе вычисляем по GPS (haversine)."
    )
    animate = st.checkbox("Анимация линии (AntPath)", True)
    time_slider = st.checkbox("Тайм-слайдер (если есть время)", True)

if not file:
    st.info("Для CSV колонка **latlng** автоматически разбивается на **lat, lon**. Все точки будут пронумерованы. Также считаются **время, километраж и скорость** (если есть timestamp).")
    st.stop()

with st.spinner("Разбор файла…"):
    raw = file.read()
    ext = file.name.lower().split(".")[-1]
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

    prefer = {"Авто":"auto","GPS":"gps","Одометр":"odo"}[dist_source]
    metrics, df = compute_metrics(df, prefer_source=prefer)

# Карта (только OpenStreetMap)
center = [float(df["lat"].mean()), float(df["lon"].mean())]
m = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=13, control_scale=True)

coords = df[["lat","lon"]].to_numpy().tolist()

# Линия маршрута + стрелки направления
poly = folium.PolyLine(coords, weight=6, opacity=0.95, color="#1976d2", tooltip="Маршрут")
poly.add_to(m)
PolyLineTextPath(
    poly, "▶", repeat=True, offset=10,
    attributes={"fill": "#1976d2", "font-weight": "bold", "font-size": "16"}
).add_to(m)

# Нумерация: яркие круглые бейджи с номером
def numbered_marker(lat, lon, num, is_last=False, color_bg="#1e90ff"):
    if num == 1:
        color_bg = "#2ecc71"  # старт: зелёный
    if is_last:
        color_bg = "#e74c3c"  # финиш: красный
    ts = df.at[num-1, "timestamp"]
    tip_time = f" • {pd.to_datetime(ts).isoformat()}" if pd.notna(ts) else ""
    seg_d = df.at[num-1, "seg_dist_km"] if num-1 >= 0 else np.nan
    seg_v = df.at[num-1, "speed_kmh"] if num-1 >= 0 else np.nan
    cum_d = df.at[num-1, "cum_km"] if num-1 >= 0 else np.nan
    tooltip = f"#{num}{tip_time}"
    if np.isfinite(seg_d):
        tooltip += f" • Δ {seg_d:.3f} км"
    if np.isfinite(seg_v):
        tooltip += f" • v {seg_v:.1f} км/ч"
    if np.isfinite(cum_d):
        tooltip += f" • Σ {cum_d:.3f} км"

    html = f"""
    <div style="
        background:{color_bg};
        color:#fff;
        border:2px solid #000;
        border-radius:50%;
        width:26px;height:26px;
        line-height:22px;
        text-align:center;
        font-weight:700;
        font-size:13px;
        box-shadow:0 0 0 2px rgba(255,255,255,0.9);
    ">{num}</div>
    """
    return folium.Marker(
        [lat, lon],
        icon=folium.DivIcon(html=html, icon_size=(26,26), icon_anchor=(13,13)),
        tooltip=tooltip
    )

# Добавляем все пронумерованные точки
N = len(coords)
for i, (lat, lon) in enumerate(coords, start=1):
    is_last = (i == N)
    numbered_marker(lat, lon, i, is_last=is_last).add_to(m)

# Отдельно пометки старт/финиш (иконки + подсказки)
folium.Marker(coords[0], tooltip="Старт (№1)", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(coords[-1], tooltip=f"Финиш (№{N})", icon=folium.Icon(color="red")).add_to(m)

# Анимация и тайм-слайдер
if animate:
    AntPath(coords, delay=800, dash_array=[10,20]).add_to(m)
if time_slider and df["timestamp"].notna().any():
    ts_points = to_geojson_points(df)
    TimestampedGeoJson(
        ts_points,
        period="PT1S",
        add_last_point=True,
        duration="PT5S",
        transition_time=200,
        loop=False,
        auto_play=False
    ).add_to(m)

st_folium(m, width=1350, height=800, returned_objects=[])

# ---- Итоги
st.subheader("Итоги")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Километраж", f"{metrics['total_km']:.3f} км")
col2.metric("Время", _format_hms(metrics["duration_h"]))
col3.metric("Средняя скорость", f"{metrics['avg_speed_kmh']:.1f} км/ч" if np.isfinite(metrics["avg_speed_kmh"]) else "—")
col4.metric("Макс. скорость", f"{metrics['max_speed_kmh']:.1f} км/ч" if np.isfinite(metrics["max_speed_kmh"]) else "—")
col5.metric("Источник расстояния", "Одометр" if metrics["used_source"]=="odom" else "GPS")

if not df["timestamp"].notna().any():
    st.warning("Во входных данных нет корректного времени — длительность и скорость посегментно не вычисляются.")

with st.expander("Первые строки очищенных данных и метрик (до 50)"):
    show_cols = ["timestamp","lat","lon","elevation","odometer","seg_dist_km","seg_dt_s","speed_kmh","cum_km"]
    st.dataframe(df[show_cols].head(50))

# Скачать результат
csv_out = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Скачать обогащённый CSV", data=csv_out, file_name="track_with_metrics.csv", mime="text/csv")
