def traffic_density(vehicle_count):
    if vehicle_count < 10:
        return "Low"
    elif vehicle_count < 25:
        return "Medium"
    else:
        return "High"
