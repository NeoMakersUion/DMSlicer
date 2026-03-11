def adjust_unit(property_name:str):
    if "density" in property_name.lower():
        return ["kg/m3","g/cm3"]
    elif "modulus" in property_name.lower():
        return ["Pa","KPa","MPa"]
    elif "ratio" in property_name.lower():
        return ["-"]
    elif "temp" in property_name.lower():
        return ["C","F","K"]
    elif "color" in property_name.lower():
        return ["RGB","HSL","HEX"]
    else:
        return ["kg/m3","Pa","-","C"]