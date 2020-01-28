# Доступ к ноде внутри материала при написании скрипта: bpy.context.view_layer.objects.active.material_slots[0].material.node_tree.nodes[3].name


import bpy
import os


def find_group(material: bpy.types.Material, name: str) -> bpy.types.ShaderNodeGroup:
    for node in material.node_tree.nodes:
        if type(node) is bpy.types.ShaderNodeGroup and node.node_tree.name == name:
            return node
    return None


def write_material(material_name: str, result_file_path: str):
    material: bpy.types.Material = bpy.data.materials[material_name]
    
    group: bpy.types.ShaderNodeGroup

    group = find_group(material, "Unlit")
    if group is not None:
        write_unlit_material(group, result_file_path)
        return

    group = find_group(material, "LitSolid")
    if group is not None:
        write_litsolid_material(group, result_file_path)
        return

    raise Exception("Unknown material type")


def write_unlit_material(group: bpy.types.ShaderNodeGroup, result_file_path:str):
    diff_rgb: bpy.types.Color = group.inputs['MatDiffColor.rgb (float3)'].default_value
    diff_a: float = group.inputs['MatDiffColor.a (float)'].default_value
    diffmap: bool = group.inputs['DIFFMAP (bool)'].default_value

    diff_map_name: str = "Unknown"
    if group.inputs['DiffMap.rgb (float3) (unit="diffuse")'].is_linked:
        diff_map_name = bpy.context.scene.urho_exportsettings.texturesPath + "/" + group.inputs['DiffMap.rgb (float3) (unit="diffuse")'].links[0].from_node.image.name

    vertexcolor: bool = group.inputs['VERTEXCOLOR (bool)'].default_value
    alphamask: bool = group.inputs['ALPHAMASK (bool)'].default_value
    alpha_blending: bool = group.inputs['ALPHA BLENDING (bool)'].default_value
    two_sided: bool = group.inputs['TWO SIDED (bool)'].default_value

    technique: str
    if diffmap:
        technique = "Diff"
    else:
        technique = "NoTexture"
    
    if vertexcolor:
        technique += "VCol"
    
    technique += "Unlit"
    
    if alpha_blending:
        technique += "Alpha"

    result: str = (
        '<material>\n'
        f'    <technique name="Techniques/{technique}.xml" />\n'
    )

    if diff_rgb[0] != 1.0 or diff_rgb[1] != 1.0 or diff_rgb[2] != 1.0 or diff_a != 1.0:
        result += f'    <parameter name="MatDiffColor" value="{diff_rgb[0]} {diff_rgb[1]} {diff_rgb[2]} {diff_a}" />\n'

    if diffmap:
        result += f'    <texture unit="diffuse" name="{diff_map_name}" />\n'
    
    if alphamask:
        result += '    <shader psdefines="ALPHAMASK" />\n'

    if two_sided:
        result += (
            '    <cull value="none" />\n'
            '    <shadowcull value="none" />\n'
        )

    result += '</material>'
    
    with open(result_file_path, "w") as text_file:
        text_file.write(result)


def write_litsolid_material(group: bpy.types.ShaderNodeGroup, result_file_path:str):
    diff_rgb: bpy.types.Color = group.inputs['MatDiffColor.rgb (float3)'].default_value
    diff_a: float = group.inputs['MatDiffColor.a (float)'].default_value
    diffmap: bool = group.inputs['DIFFMAP (bool)'].default_value

    diff_map_name: str = "Unknown"
    if group.inputs['DiffMap.rgb (float3) (unit="diffuse")'].is_linked:
        diff_map_name = bpy.context.scene.urho_exportsettings.texturesPath + "/" + group.inputs['DiffMap.rgb (float3) (unit="diffuse")'].links[0].from_node.image.name

    vertexcolor: bool = group.inputs['VERTEXCOLOR (bool)'].default_value
    spec_rgb: bpy.types.Color = group.inputs['MatSpecColor.rgb (float3)'].default_value
    spec_a: float = group.inputs['MatSpecColor.a (float) (specularPower)'].default_value
    specmap: bool = group.inputs['SPECMAP (bool)'].default_value

    spec_map_name: str = "Unknown"
    if group.inputs['SpecMap.rgb (float3) (unit="specular")'].is_linked:
        spec_map_name = bpy.context.scene.urho_exportsettings.texturesPath + "/" + group.inputs['SpecMap.rgb (float3) (unit="specular")'].links[0].from_node.image.name

    emis_rgb: bpy.types.Color = group.inputs['MatEmissiveColor.rgb (float3)'].default_value
    emissivemap: bool = group.inputs['EMISSIVEMAP (bool)'].default_value

    emis_map_name: str = "Unknown"
    if group.inputs['EmissiveMap.rgb (float3) (unit="emissive")'].is_linked:
        emis_map_name = bpy.context.scene.urho_exportsettings.texturesPath + "/" + group.inputs['EmissiveMap.rgb (float3) (unit="emissive")'].links[0].from_node.image.name

    normalmap: bool = group.inputs['NORMALMAP (bool)'].default_value

    normal_map_name: str = "Unknown"
    if group.inputs['NormalMap.rgb (float3) (unit="normal")'].is_linked:
        normal_map_name = bpy.context.scene.urho_exportsettings.texturesPath + "/" + group.inputs['NormalMap.rgb (float3) (unit="normal")'].links[0].from_node.image.name

    translucent: bool = group.inputs['TRANSLUCENT (bool)'].default_value
    alphamask: bool = group.inputs['ALPHAMASK (bool)'].default_value
    alpha_blending: bool = group.inputs['ALPHA BLENDING (bool)'].default_value
    two_sided: bool = group.inputs['TWO SIDED (bool)'].default_value
    
    technique: str
    if diffmap:
        technique = "Diff"
    else:
        technique = "NoTexture"

    if vertexcolor:
        technique += "VCol"

    if normalmap:
        technique += "Normal"

    if specmap:
        technique += "Spec"

    if emissivemap:
        technique += "Emissive"
    
    if alpha_blending:
        technique += "Alpha"

    if translucent:
        technique += "Translucent"

    result: str = (
        '<material>\n'
        f'    <technique name="Techniques/{technique}.xml" />\n'
    )

    if diff_rgb[0] != 1.0 or diff_rgb[1] != 1.0 or diff_rgb[2] != 1.0 or diff_a != 1.0:
        result += f'    <parameter name="MatDiffColor" value="{diff_rgb[0]} {diff_rgb[1]} {diff_rgb[2]} {diff_a}" />\n'
    
    if diffmap:
        result += f'    <texture unit="diffuse" name="{diff_map_name}" />\n'

    if spec_rgb[0] != 0.0 or spec_rgb[1] != 0.0 or spec_rgb[2] != 0.0 or spec_a != 1.0:
        result += f'    <parameter name="MatSpecColor" value="{spec_rgb[0]} {spec_rgb[1]} {spec_rgb[2]} {spec_a}" />\n'

    if specmap:
        result += f'    <texture unit="specular" name="{spec_map_name}" />\n'

    if emis_rgb[0] != 0.0 or emis_rgb[1] != 0.0 or emis_rgb[2] != 0.0:
        result += f'    <parameter name="MatEmissiveColor" value="{emis_rgb[0]} {emis_rgb[1]} {emis_rgb[2]}" />\n'

    if emissivemap:
        result += f'    <texture unit="emissive" name="{emis_map_name}" />\n'

    if normalmap:
        result += f'    <texture unit="normal" name="{normal_map_name}" />\n'
    
    if alphamask:
        result += '    <shader psdefines="ALPHAMASK" />\n'

    if two_sided:
        result += (
            '    <cull value="none" />\n'
            '    <shadowcull value="none" />\n'
        )

    result += '</material>'
    
    with open(result_file_path, "w") as text_file:
        text_file.write(result)

