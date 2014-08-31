
#
# This script is licensed as public domain.
#

from .utils       import ComposePath
from .export_urho import FloatToString, BinaryFileWriter, XmlToPrettyString
from xml.etree    import ElementTree as ET
import os
import logging

log = logging.getLogger("ExportLogger")

# Options for scene and nodes export
class SOptions:
    def __init__(self):
        self.doIndividualPrefab = False
        self.doCollectivePrefab = False
        self.doScenePrefab = False
        self.doPhysics = False
        self.mergeObjects = False


# Write individual prefabs
def WriteIndividualPrefabs(model, sceneName, physics, filename, useStandardDirs):

    # Set first node ID
    nodeID = 0x1000000

    # Check for Static or Animated Model
    modelType = "StaticModel"
    if len(model.bones) > 0:
        modelType = "AnimatedModel"

    # Create node name
    nodeName = ""
    if not useStandardDirs:
        nodeName = "Models/" + sceneName + "/" + model.name + os.path.extsep + "mdl"
    else: 
        nodeName = "Models/" + model.name + os.path.extsep + "mdl"

    # Gather materials
    materials = ""
    for uGeometry in model.geometries:			
        name = uGeometry.uMaterialName
        if name:
            if not useStandardDirs:
                name = "Models/" + sceneName + "/Materials/" + name + os.path.extsep + "xml"
            else: 
                name = "Materials/" + name + os.path.extsep + "xml"
            materials = materials + ";" + name

    # Generate xml prefab content
    rootNodeElem = ET.Element('node')
    rootNodeElem.set("id", "{:d}".format(nodeID))

    modelNameElem = ET.SubElement(rootNodeElem, "attribute")
    modelNameElem.set("name", "Name")
    modelNameElem.set("value", model.name)

    typeElem = ET.SubElement(rootNodeElem, "component")
    typeElem.set("type", modelType)
    typeElem.set("id", "{:d}".format(nodeID))

    modelElem = ET.SubElement(typeElem, "attribute")
    modelElem.set("name", "Model")
    modelElem.set("value", "Model;" + nodeName)

    materialElem = ET.SubElement(typeElem, "attribute")
    materialElem.set("name", "Material")
    materialElem.set("value", "Material" + materials)

    if physics:
        bodyElem = ET.SubElement(rootNodeElem, "component")
        bodyElem.set("type", "RigidBody")
        bodyElem.set("id", "{:d}".format(nodeID+1))

        collisionLayerElem = ET.SubElement(bodyElem, "attribute")
        collisionLayerElem.set("name", "Collision Layer")
        collisionLayerElem.set("value", FloatToString(2))

        gravityElem = ET.SubElement(bodyElem, "attribute")
        gravityElem.set("name", "Use Gravity")
        gravityElem.set("value", "false")

        shapeElem = ET.SubElement(rootNodeElem, "component")
        shapeElem.set("type", "CollisionShape")
        shapeElem.set("id", "{:d}".format(nodeID+2))

        shapeTypeElem = ET.SubElement(shapeElem, "attribute")
        shapeTypeElem.set("name", "Shape Type")
        shapeTypeElem.set("value", "TriangleMesh")

        shapeTypeElem = ET.SubElement(shapeElem, "attribute")
        shapeTypeElem.set("name", "Offset Position")
        shapeTypeElem.set("value", "0 1.02 0")

        physicsModelElem = ET.SubElement(shapeElem, "attribute")
        physicsModelElem.set("name", "Model")
        physicsModelElem.set("value", "Model;" + nodeName)

    # Write xml prefab file using Ascii xml for size and readability matters #TODO: full binary
    fw = BinaryFileWriter()
    try:
        fw.open(filename)
    except Exception as e:
        log.error("Cannot open file {:s} {:s}".format(filename, e))
        return
    fw.writeAsciiStr(XmlToPrettyString(rootNodeElem)) #fw.writeAsciiStr(XmlToPrettyString(content))
    fw.close()


# Write prefab xml file
def WritePrefab(file, content, overwrite):
    if not os.path.exists(file) or overwrite:
        log.info( "Creating/Updating Prefab file {:s}".format(file) )
        fw = BinaryFileWriter()
        try:
            fw.open(file)
        except Exception as e:
            log.error("Cannot open prefab file {:s} {:s}".format(file, e))
            return
        fw.writeAsciiStr(XmlToPrettyString(content))
        fw.close()
    else:
        log.error( "Prefab file already exists {:s}".format(file) )


# Export scene and nodes
def UrhoExportScene(context, uModelList, settings, sOptions):

    # Create folders and filenames
    sceneName = context.scene.name
        
    if sOptions.doCollectivePrefab or sOptions.doIndividualPrefab:
        objectsPath = ComposePath(settings.outputPath, "Objects", settings.useStandardDirs)

    # Create "TestScenes" folder an use scene name for export
    if sOptions.doScenePrefab:
        scenesPath = ComposePath(settings.outputPath, "Scenes", settings.useStandardDirs)
        sceneFullFilename = os.path.join(scenesPath, sceneName + os.path.extsep + "xml")

    # Use scene name for the export in "TestObjects" folder
    if sOptions.doCollectivePrefab: 
        sceneFilename = os.path.join(objectsPath, sceneName + os.path.extsep + "xml")

    # For Individual prefabs (sOptions.doIndividualPrefab) we will set filename later as it is specific to each exported object

    a = {}
    k = 0x1000000   # node ID
    compoID = k     # component ID
    m = 0           # internal counter

    # Create scene components
    if sOptions.doScenePrefab:
        sceneRoot = ET.Element('scene')
        sceneRoot.set("id", "1")

        a["{:d}".format(m)] = ET.SubElement(sceneRoot, "component")
        a["{:d}".format(m)].set("type", "Octree")
        a["{:d}".format(m)].set("id", "1")

        a["{:d}".format(m+1)] = ET.SubElement(sceneRoot, "component")
        a["{:d}".format(m+1)].set("type", "DebugRenderer")
        a["{:d}".format(m+1)].set("id", "2")

        a["{:d}".format(m+2)] = ET.SubElement(sceneRoot, "component")
        a["{:d}".format(m+2)].set("type", "Light")
        a["{:d}".format(m+2)].set("id", "3")

        a["{:d}".format(m+3)] = ET.SubElement(a["{:d}".format(m+2)], "attribute")
        a["{:d}".format(m+3)].set("name", "Light Type")
        a["{:d}".format(m+3)].set("value", "Directional")
        m += 4

        if sOptions.doPhysics:
            a["{:d}".format(m)] = ET.SubElement(sceneRoot, "component")
            a["{:d}".format(m)].set("type", "PhysicsWorld")
            a["{:d}".format(m)].set("id", "4")
            m += 1

        # Create Root node
        root = ET.SubElement(sceneRoot, "node")
    else: 
        # Root node
        root = ET.Element('node') 
        
    root.set("id", "{:d}".format(k))
    a["{:d}".format(m)] = ET.SubElement(root, "attribute")
    a["{:d}".format(m)].set("name", "Name")
    a["{:d}".format(m)].set("value", sceneName)
    
    # Create physics stuff for the root node
    if sOptions.doPhysics:
        if not settings.useStandardDirs:
            physicsModel = "Models/" + sceneName + "/" + "Physics" + os.path.extsep + "mdl"
        else: 
            physicsModel = "Models/" + "Physics" + os.path.extsep + "mdl"

        a["{:d}".format(m)] = ET.SubElement(root, "component")
        a["{:d}".format(m)].set("type", "RigidBody")
        a["{:d}".format(m)].set("id", "{:d}".format(compoID))

        a["{:d}".format(m+1)] = ET.SubElement(a["{:d}".format(m)] , "attribute")
        a["{:d}".format(m+1)].set("name", "Collision Layer")
        a["{:d}".format(m+1)].set("value", "2")

        a["{:d}".format(m+2)] = ET.SubElement(a["{:d}".format(m)], "attribute")
        a["{:d}".format(m+2)].set("name", "Use Gravity")
        a["{:d}".format(m+2)].set("value", "false")

        a["{:d}".format(m+3)] = ET.SubElement(root, "component")
        a["{:d}".format(m+3)].set("type", "CollisionShape")
        a["{:d}".format(m+3)].set("id", "{:d}".format(compoID+1))
        m += 3

        a["{:d}".format(m+1)] = ET.SubElement(a["{:d}".format(m)], "attribute")
        a["{:d}".format(m+1)].set("name", "Shape Type")
        a["{:d}".format(m+1)].set("value", "TriangleMesh")

        a["{:d}".format(m+2)] = ET.SubElement(a["{:d}".format(m)], "attribute")
        a["{:d}".format(m+2)].set("name", "Model")
        a["{:d}".format(m+2)].set("value", "Model;" + physicsModel)
        m += 2
        compoID += 2

    # Export each decomposed object
    for uModel in uModelList:

        # Check for Static or Animated Model
        modelType = "StaticModel"
        if len(uModel.bones) > 0:
            modelType = "AnimatedModel"

        # Create node name
        if not settings.useStandardDirs:
            nodeName = "Models/" + sceneName + "/" + uModel.name + os.path.extsep + "mdl"
        else: 
            nodeName = "Models/" + uModel.name + os.path.extsep + "mdl"

        # Gather materials
        materials = ""
        for uGeometry in uModel.geometries:			
            name = uGeometry.uMaterialName
            if name:
                if not settings.useStandardDirs:
                    name = "Models/" + sceneName + "/Materials/" + name + os.path.extsep + "xml"
                else: 
                    name = "Materials/" + name + os.path.extsep + "xml"
                materials = materials + ";" + name

        # Generate XML Content
        k += 1
        modelNode = uModel.name
        ## !!FIXME
        #if modelType == "StaticModel" and uModel.parentName:
        #    a[modelNode] = ET.SubElement(a[uModel.parentName], "node") #If child node, parent to parent object instead of root
        #else: 
        #    a[modelNode] = ET.SubElement(root, "node")
        a[modelNode] = ET.SubElement(root, "node")
        ## !!FIXME
            
        a[modelNode].set("id", "{:d}".format(k))

        a["{:d}".format(m)] = ET.SubElement(a[modelNode], "attribute")
        a["{:d}".format(m)].set("name", "Name")
        a["{:d}".format(m)].set("value", uModel.name)
        m += 1

        a["{:d}".format(m)] = ET.SubElement(a[modelNode], "component")
        a["{:d}".format(m)].set("type", modelType)
        a["{:d}".format(m)].set("id", "{:d}".format(compoID))
        m += 1

        a["{:d}".format(m)] = ET.SubElement(a["{:d}".format(m-1)], "attribute")
        a["{:d}".format(m)].set("name", "Model")
        a["{:d}".format(m)].set("value", "Model;" + nodeName)
        m += 1

        a["{:d}".format(m)] = ET.SubElement(a["{:d}".format(m-2)], "attribute")
        a["{:d}".format(m)].set("name", "Material")
        a["{:d}".format(m)].set("value", "Material" + materials)
        m += 1
        compoID += 1

        # Write individual prefabs
        if sOptions.doIndividualPrefab:
            filename2 = os.path.join(objectsPath, uModel.name + os.path.extsep + "xml")

            if not os.path.exists(filename2) or settings.fileOverwrite:
                log.info( "Creating Prefab file {:s}".format(filename2) )
                WriteIndividualPrefabs(uModel, sceneName, sOptions.doPhysics, filename2, settings.useStandardDirs)
            else:
                log.error( "File already exists {:s}".format(filename2) )

        # Merging objects equates to an individual export. And collective equates to individual, so we can skip collective
        if sOptions.mergeObjects and sOptions.doScenePrefab: 
            WritePrefab(sceneFullFilename, sceneRoot, settings.fileOverwrite)

    # Write collective and scene prefab files
    if not sOptions.mergeObjects:

        if sOptions.doCollectivePrefab:
            WritePrefab(sceneFilename, root, settings.fileOverwrite)

        if sOptions.doScenePrefab:
            WritePrefab(sceneFullFilename, sceneRoot, settings.fileOverwrite)


