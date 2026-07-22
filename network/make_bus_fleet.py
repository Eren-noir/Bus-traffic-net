"""
Phase 1b: Take the randomly generated trips and tag a subset of vehicles as
'bus' (the info-sharing agents). Everything else stays a regular car.
Produces routes.rou.xml with two vTypes: car, bus.
"""
import xml.etree.ElementTree as ET
import random

random.seed(42)
BUS_FRACTION = 0.15  # ~15% of vehicles are buses (roughly matches real fleet mix)

tree = ET.parse("trips.trips.xml")
root = tree.getroot()

# Insert vType definitions at the top
vtype_car = ET.Element("vType", {
    "id": "car", "length": "5", "maxSpeed": "20", "color": "1,1,1",
})
vtype_bus = ET.Element("vType", {
    "id": "bus", "length": "12", "maxSpeed": "15", "color": "1,0.6,0",
    "vClass": "bus",
})
root.insert(0, vtype_bus)
root.insert(0, vtype_car)

n_bus = 0
for trip in root.findall("trip"):
    is_bus = random.random() < BUS_FRACTION
    trip.set("type", "bus" if is_bus else "car")
    if is_bus:
        n_bus += 1

tree.write("routes.rou.xml", xml_declaration=True, encoding="UTF-8")
total = len(root.findall("trip"))
print(f"Total vehicles: {total}, buses: {n_bus} ({n_bus/total:.1%})")
