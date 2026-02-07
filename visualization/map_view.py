# visualization/map_view.py

import folium
from folium.plugins import HeatMap


def visualize_gaza_dashboard(df):
    """
    Create an interactive Gaza reconstruction map using Folium.
    """

    # -------------------------------------------------
    # Gaza center
    # -------------------------------------------------
    GAZA_LAT = 31.5204
    GAZA_LON = 34.4536

    m = folium.Map(
        location=[GAZA_LAT, GAZA_LON],
        zoom_start=13,
        tiles="OpenStreetMap"
    )

    # -------------------------------------------------
    # Feature Groups
    # -------------------------------------------------
    roads_layer = folium.FeatureGroup("Reconstruction Decisions")
    heat_layer = folium.FeatureGroup("Damage Heatmap")
    hospitals_layer = folium.FeatureGroup("Hospitals")
    aid_layer = folium.FeatureGroup("Aid Routes")

    # -------------------------------------------------
    # Roads (Selected vs Deferred)
    # -------------------------------------------------
    for _, r in df.iterrows():

        color = "green" if r["selected"] == 1 else "red"

        popup_text = f"""
        <b>Road ID:</b> {r['id']}<br>
        <b>Status:</b> {"Rebuild" if r['selected'] == 1 else "Deferred"}<br>
        <b>Impact:</b> {r['impact']:.3f}<br>
        <b>Cost:</b> {r['final_cost']:.2f}<br>
        <b>Population:</b> {r['population']}
        """

        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=9,
            color=color,
            fill=True,
            fill_opacity=0.85,
            popup=popup_text
        ).add_to(roads_layer)

    # -------------------------------------------------
    # Damage Heatmap
    # -------------------------------------------------
    heat_data = [
        [r["lat"], r["lon"], r["damage"]]
        for _, r in df.iterrows()
    ]

    HeatMap(
        heat_data,
        radius=25,
        blur=15,
        min_opacity=0.5
    ).add_to(heat_layer)

    # -------------------------------------------------
    # Hospitals (Realistic Gaza examples)
    # -------------------------------------------------
    hospitals = [
        ("Al-Shifa Hospital", 31.5283, 34.4607),
        ("Al-Quds Hospital", 31.5070, 34.4465),
        ("Indonesian Hospital", 31.5631, 34.5209),
    ]

    for name, lat, lon in hospitals:
        folium.Marker(
            location=[lat, lon],
            popup=f"🏥 {name}",
            icon=folium.Icon(color="blue", icon="plus-sign")
        ).add_to(hospitals_layer)

    # -------------------------------------------------
    # Aid Routes (Example corridors)
    # -------------------------------------------------
    aid_routes = [
        [(31.515, 34.440), (31.525, 34.460)],
        [(31.500, 34.455), (31.540, 34.470)]
    ]

    for route in aid_routes:
        folium.PolyLine(
            route,
            color="orange",
            weight=4,
            opacity=0.9
        ).add_to(aid_layer)

    # -------------------------------------------------
    # Add Layers to Map
    # -------------------------------------------------
    roads_layer.add_to(m)
    heat_layer.add_to(m)
    hospitals_layer.add_to(m)
    aid_layer.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)

    return m