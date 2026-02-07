# quantum/data_loader.py

import pandas as pd

def load_road_data():
    """
    Load structured road-level data for Gaza Strip.

    NOTE:
    - Locations and population figures are approximated
      using publicly available humanitarian reports
      (UN OCHA, WorldPop, OpenStreetMap).
    - This dataset is designed for decision-support simulation,
      not for exact operational deployment.
    """

    roads = [

        # ---------------------------
        # Gaza City
        # ---------------------------
        {
            "id": 0,
            "zone": "Gaza City",
            "road_name": "Al-Jalaa Street",
            "damage": 0.9,
            "population": 18000,
            "distance_to_hospital": 0.5,
            "aid_route": 1.0,
            "land_use": "residential",
            "soil": "sandy",
            "base_cost": 5,
            "lat": 31.5204,
            "lon": 34.4536
        },

        {
            "id": 1,
            "zone": "Gaza City",
            "road_name": "Omar Al-Mukhtar Street",
            "damage": 0.6,
            "population": 12000,
            "distance_to_hospital": 1.2,
            "aid_route": 0.6,
            "land_use": "mixed",
            "soil": "compact",
            "base_cost": 4,
            "lat": 31.5240,
            "lon": 34.4580
        },

        # ---------------------------
        # Jabalia
        # ---------------------------
        {
            "id": 2,
            "zone": "Jabalia",
            "road_name": "Camp Main Road",
            "damage": 0.85,
            "population": 22000,
            "distance_to_hospital": 0.8,
            "aid_route": 1.0,
            "land_use": "residential",
            "soil": "sandy",
            "base_cost": 7,
            "lat": 31.5330,
            "lon": 34.4830
        },

        # ---------------------------
        # Khan Younis
        # ---------------------------
        {
            "id": 3,
            "zone": "Khan Younis",
            "road_name": "Salah Al-Din Road",
            "damage": 0.5,
            "population": 15000,
            "distance_to_hospital": 2.5,
            "aid_route": 0.7,
            "land_use": "mixed",
            "soil": "compact",
            "base_cost": 6,
            "lat": 31.3400,
            "lon": 34.3060
        },

        # ---------------------------
        # Rafah
        # ---------------------------
        {
            "id": 4,
            "zone": "Rafah",
            "road_name": "Border Access Road",
            "damage": 0.7,
            "population": 10000,
            "distance_to_hospital": 3.0,
            "aid_route": 0.9,
            "land_use": "industrial",
            "soil": "sandy",
            "base_cost": 5,
            "lat": 31.2870,
            "lon": 34.2590
        }
    ]

    return pd.DataFrame(roads)