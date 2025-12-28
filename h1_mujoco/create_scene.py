import xml.etree.ElementTree as ET
import os

assets_dir = os.path.join(os.path.dirname(__file__), "assets")
h1_path = os.path.join(assets_dir, "h1.xml")
scene_path = os.path.join(assets_dir, "h1_scene.xml")

def create_scene():
    print(f"Parsing {h1_path}...")
    tree = ET.parse(h1_path)
    root = tree.getroot()

    # 1. Update Option
    # Attempt to set collision="all" if supported, but user reported error.
    # We will rely on default collision pairing.
    # Ensure timestep/gravity are set.
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.002")
    option.set("gravity", "0 0 -9.81")
    
    # 2. Add Assets (Skybox, Ground Plane)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    
    # Check if we already added them (idempotency)
    if asset.find(".//texture[@name='skybox']") is None:
        # Skybox
        ET.SubElement(asset, "texture", {
            "type": "skybox", "builtin": "gradient", 
            "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", 
            "width": "512", "height": "3072"
        })
        # Floor Texture
        ET.SubElement(asset, "texture", {
            "type": "2d", "name": "groundplane", "builtin": "checker", 
            "mark": "edge", "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", 
            "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300"
        })
        # Floor Material
        ET.SubElement(asset, "material", {
            "name": "groundplane", "texture": "groundplane", 
            "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2"
        })

    # 3. Add Visual Defaults
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
        
    if visual.find("headlight") is None:
        ET.SubElement(visual, "headlight", {
            "diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"
        })
    if visual.find("rgba") is None:
        ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    if visual.find("global") is None:
        ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})

    # 4. Modify Worldbody
    worldbody = root.find("worldbody")
    
    # Remove old floor geoms (detected by their size/pos or just first items if we know order)
    # We previously saw they are the first children.
    # Let's verify by iterating and removing logic.
    to_remove = []
    for child in worldbody:
        if child.tag == "geom" and child.get("type") == "box" and child.get("size") == "10 10 0.05":
            to_remove.append(child)
    
    for item in to_remove:
        print("Removing old floor geom")
        worldbody.remove(item)
        
    # COLLISION FIX 2.0 (STABILITY)
    # The simulation is exploding (1-step episodes). This is likely due to overlapping collision primitives.
    # We will DISABLE SELF-COLLISION but ENABLE FLOOR COLLISION.
    # Logic:
    # - Robot Geoms: contype=1, conaffinity=0
    # - Floor Geom:  contype=1, conaffinity=1
    # Collision check: (A.contype & B.conaffinity) | (B.contype & A.conaffinity)
    # Robot vs Robot: (1 & 0) | (1 & 0) = 0 (No Collision)
    # Robot vs Floor: (1 & 1) | (1 & 0) = 1 (Collision)
    
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        # Floor is set later, but if it exists...
        if name == "floor":
             continue
             
        # Robot Parts
        gtype = geom.get("type", "unknown")
        
        if gtype == "mesh":
            # Visuals: No collision at all (cleaner)
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("group", "1")
        else:
            # Collision Primitives: Collide with World, not Self
            geom.set("contype", "1")
            geom.set("conaffinity", "0") 
            geom.set("rgba", "0.6 0.6 0.6 0.0") # Invisible
            geom.set("group", "0")

    # Insert New Floor and Light
    light = ET.Element("light", {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"})
    
    # Floor: 1, 1 (Accepts collisions from Robot)
    floor = ET.Element("geom", {
        "name": "floor", "size": "0 0 0.05", "pos": "0 0 0", "type": "plane", 
        "material": "groundplane", "contype": "1", "conaffinity": "1"
    })
    
    worldbody.insert(0, floor)
    worldbody.insert(0, light)
    
    # 6. Add Actuators
    # We need to find all joints and create a motor for them (except floating base)
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
        
    for joint in root.findall(".//joint"):
        jname = joint.get("name")
        if jname == "floating_base_joint":
            continue
            
        # Get limits if available
        # XML stores this as actuatorfrcrange="-200 200"
        frc_range = joint.get("actuatorfrcrange")
        
        motor = ET.SubElement(actuator, "motor")
        motor.set("name", jname + "_motor")
        motor.set("joint", jname)
        motor.set("gear", "1")
        motor.set("ctrllimited", "true")
        
        if frc_range:
            motor.set("ctrlrange", frc_range)
        else:
            # Default fallback
            motor.set("ctrlrange", "-50 50")
            
        print(f"Added motor for {jname} range={frc_range}")

    # 5. Save
    print(f"Saving to {scene_path}...")
    tree.write(scene_path)
    print("Done.")

if __name__ == "__main__":
    create_scene()
