from flask import Flask, render_template, request
import pandas as pd
import folium
import math
import numpy as np
from queue import PriorityQueue

app = Flask(__name__)

# -----------------------------------------
# Load Data
# -----------------------------------------
data = pd.read_csv(r"D:\MCA\SEM1\python\safeRoute\data.csv")
places = data["place"].tolist()

# -----------------------------------------
# Haversine Distance
# -----------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

# -----------------------------------------
# Build Neighbors
# -----------------------------------------
coords = data[["latitude", "longitude"]].to_numpy()
neighbor_dict = {}
RADIUS_KM = 10

for i, place in enumerate(places):
    lat1, lon1 = coords[i]
    neighbors = []
    for j, (lat2, lon2) in enumerate(coords):
        if i != j:
            dist = haversine(lat1, lon1, lat2, lon2)
            if dist <= RADIUS_KM:
                neighbors.append((places[j], dist))
    if len(neighbors) < 3:
        dists = np.array([
            haversine(lat1, lon1, lat2, lon2) if j != i else np.inf
            for j, (lat2, lon2) in enumerate(coords)
        ])
        nearest_idx = np.argsort(dists)[:3]
        neighbors = [(places[j], dists[j]) for j in nearest_idx]
    neighbor_dict[place] = neighbors

data["neighbors"] = data["place"].apply(lambda x: [n for n, _ in neighbor_dict[x]])

# -----------------------------------------
# A* Pathfinding (Improved Distance + Time)
# -----------------------------------------
def a_star(start, end, alpha=0.5, base_speed_kmh=40):
    pq = PriorityQueue()
    pq.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    place_index = {p: i for i, p in enumerate(places)}

    while not pq.empty():
        _, current = pq.get()
        if current == end:
            break

        curr_row = data.iloc[place_index[current]]
        curr_lat, curr_lon = curr_row["latitude"], curr_row["longitude"]

        for neighbor, dist in neighbor_dict[current]:
            n_row = data.iloc[place_index[neighbor]]
            safety = n_row["safety_score"]

            safety_factor = 1 - (safety / 10)
            cost = alpha * (dist / 10) + (1 - alpha) * safety_factor
            new_cost = cost_so_far[current] + cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                goal = data.iloc[place_index[end]]
                h_dist = haversine(n_row["latitude"], n_row["longitude"],
                                   goal["latitude"], goal["longitude"])
                heuristic = alpha * (h_dist / 10)
                pq.put((new_cost + heuristic, neighbor))
                came_from[neighbor] = current

    # reconstruct path
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()

    # ----------------------------------------
    # Accurate Distance & Time
    # ----------------------------------------
    total_road_distance = 0
    total_safety = 0
    total_time_hr = 0

    for i in range(len(path) - 1):
        r1 = data.iloc[place_index[path[i]]]
        r2 = data.iloc[place_index[path[i + 1]]]
        safety = (r1["safety_score"] + r2["safety_score"]) / 2

        # Step 1: Realistic road distance (slight detour)
        straight_dist = haversine(r1["latitude"], r1["longitude"], r2["latitude"], r2["longitude"])
        road_dist = straight_dist * 1.15
        total_road_distance += road_dist

        # Step 2: Adjust speed based on safety + type of route
        if road_dist <= 2:
            speed = 5  # walking
        elif road_dist <= 10:
            speed = 25  # city traffic
        else:
            speed = 50  # highways or intercity

        # Safety impact on speed
        if safety < 4:
            speed *= 0.8
        elif safety >= 7:
            speed *= 1.1

        # Step 3: Add segment time
        total_time_hr += road_dist / speed
        total_safety += safety

    total_safety += data.iloc[place_index[path[-1]]]["safety_score"]
    avg_safety = total_safety / len(path)

    return path, total_road_distance, avg_safety, total_time_hr

# -----------------------------------------
# Flask Route
# -----------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    map_html = None
    stats = None

    if request.method == "POST":
        start = request.form["start"]
        end = request.form["end"]
        alpha = float(request.form.get("alpha", 0.5))

        path, total_dist, avg_safety, est_time_hr = a_star(start, end, alpha=alpha)

        # Convert time into minutes/hours dynamically
        if est_time_hr < 1:
            time_str = f"{round(est_time_hr * 60, 1)}"
        else:
            time_str = f"{round(est_time_hr, 2)} hrs"

        start_row = data[data["place"] == start].iloc[0]
        m = folium.Map(location=[start_row["latitude"], start_row["longitude"]], zoom_start=13)

        coordinates = []
        for place in path:
            row = data[data["place"] == place].iloc[0]
            coordinates.append([row["latitude"], row["longitude"]])
            folium.Marker(
                [row["latitude"], row["longitude"]],
                popup=f"<b>{place}</b><br>Safety: {row['safety_score']}",
                icon=folium.Icon(
                    color="green" if place == start else "red" if place == end else "blue"
                ),
            ).add_to(m)

        route_color = "green" if avg_safety >= 7 else "orange" if avg_safety >= 4 else "red"

        folium.PolyLine(
            coordinates,
            color=route_color,
            weight=5,
            opacity=0.9,
            tooltip=f"Route: {' → '.join(path)}",
        ).add_to(m)

        legend_html = '''
             <div style="position: fixed; 
             bottom: 50px; left: 50px; width: 210px; height: 120px; 
             background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
             box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
             padding: 10px;">
             <b>🚦 Safety Level Legend</b><br>
             <i style="background:green; width:15px; height:15px; float:left; margin-right:8px;"></i> Safe (≥ 7)<br>
             <i style="background:orange; width:15px; height:15px; float:left; margin-right:8px;"></i> Moderate (4–7)<br>
             <i style="background:red; width:15px; height:15px; float:left; margin-right:8px;"></i> Unsafe (&lt; 4)
             </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        map_html = m._repr_html_()
        stats = {
            "distance": round(total_dist, 2),
            "safety": round(avg_safety, 2),
            "time": time_str
        }

    return render_template("index.html", map=map_html, places=places, stats=stats)

# -----------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
