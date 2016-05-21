
# LOD: be sure all have the same material or a new geometry is created

#
# This script is licensed as public domain.
#

# http://www.blender.org/documentation/blender_python_api_2_57_release/bpy.props.html
# http://www.blender.org/documentation/blender_python_api_2_59_0/bpy.props.html
# http://www.blender.org/documentation/blender_python_api_2_66_4/bpy.props.html
# http://www.blender.org/documentation/blender_python_api_2_57_release/bpy.types.Panel.html
# http://www.blender.org/documentation/blender_python_api_2_57_release/bpy.types.PropertyGroup.html
# http://www.blender.org/documentation/blender_python_api_2_66_4/bpy.types.WindowManager.html
# http://wiki.blender.org/index.php/Dev:2.5/Py/Scripts/Guidelines/Layouts
# http://wiki.blender.org/index.php/Dev:2.5/Py/Scripts/Cookbook/Code_snippets/Properties
# http://wiki.blender.org/index.php/Dev:2.5/Py/Scripts/Cookbook/Code_snippets/Interface
# http://wiki.blender.org/index.php/Dev:IT/2.5/Py/Scripts/Cookbook/Code_snippets/Interface
# http://wiki.blender.org/index.php/Dev:IT/2.5/Py/Scripts/Cookbook/Code_snippets/Multi-File_packages
# http://wiki.blender.org/index.php/Doc:2.6/Manual/Extensions/Python/Properties
# http://www.blender.org/documentation/blender_python_api_2_66_4/info_tutorial_addon.html

DEBUG = 0
if DEBUG: print("Urho export init")

bl_info = {
    "name": "Urho3D export",
    "description": "Urho3D export",
    "author": "reattiva",
    "version": (0, 6),
    "blender": (2, 66, 0),
    "location": "Properties > Render > Urho export",
    "warning": "big bugs, use at your own risk",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Import-Export"}

if "decompose" in locals():
    import imp
    imp.reload(decompose)
    imp.reload(export_urho)
    imp.reload(export_scene)
    imp.reload(utils)
    if DEBUG and "testing" in locals(): imp.reload(testing)

from .decompose import TOptions, Scan
from .export_urho import UrhoExportData, UrhoExportOptions, UrhoWriteModel, UrhoWriteAnimation, \
                         UrhoWriteTriggers, UrhoExport
from .export_scene import SOptions, UrhoScene, UrhoExportScene, UrhoWriteMaterial, UrhoWriteMaterialsList
from .utils import PathType, FOptions, GetFilepath, CheckFilepath, ErrorsMem
if DEBUG: from .testing import PrintUrhoData, PrintAll

import os
import time
import sys
import shutil
import logging

import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty, FloatProperty, IntProperty
from bpy.app.handlers import persistent
from mathutils import Quaternion
from math import radians

#--------------------
# Loggers
#--------------------

# A list to save export messages
logList = []

# Create a logger
log = logging.getLogger("ExportLogger")
log.setLevel(logging.DEBUG)

# Formatter for the logger
FORMAT = '%(levelname)s:%(message)s'
formatter = logging.Formatter(FORMAT)

# Console filter: no more than 3 identical messages 
consoleFilterMsg = None
consoleFilterCount = 0
class ConsoleFilter(logging.Filter):
    def filter(self, record):
        global consoleFilterMsg
        global consoleFilterCount
        if consoleFilterMsg == record.msg:
            consoleFilterCount += 1
            if consoleFilterCount > 2:
                return False
        else:
            consoleFilterCount = 0
            consoleFilterMsg = record.msg
        return True
consoleFilter = ConsoleFilter()

# Logger handler which saves unique messages in the list
logMaxCount = 500
class ExportLoggerHandler(logging.StreamHandler):
    def emit(self, record):
        global logList
        try:
            if len(logList) < logMaxCount:
                msg = self.format(record)
                if not msg in logList:
                    logList.append(msg)
            #self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

# Delete old handlers
for handler in reversed(log.handlers):
    log.removeHandler(handler)

# Create a logger handler for the list
listHandler = ExportLoggerHandler()
listHandler.setFormatter(formatter)
log.addHandler(listHandler)
    
# Create a logger handler for the console
consoleHandler = logging.StreamHandler()
consoleHandler.addFilter(consoleFilter)
log.addHandler(consoleHandler)



#--------------------
# Blender UI
#--------------------

# Addon preferences, they are visible in the Users Preferences Addons page,
# under the Urho exporter addon row
class UrhoAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    outputPath = StringProperty(
            name = "Default export path",
            description = "Default path where to export",
            default = "", 
            maxlen = 1024,
            subtype = "DIR_PATH")

    modelsPath = StringProperty(
            name = "Default Models subpath",
            description = "Models subpath (relative to output)",
            default = "Models")
    animationsPath = StringProperty(
            name = "Default Animations subpath",
            description = "Animations subpath (relative to output)",
            default = "Models")
    materialsPath = StringProperty(
            name = "Default Materials subpath",
            description = "Materials subpath (relative to output)",
            default = "Materials")
    techniquesPath = StringProperty(
            name = "Default Techniques subpath",
            description = "Techniques subpath (relative to output)",
            default = "Techniques")
    texturesPath = StringProperty(
            name = "Default Textures subpath",
            description = "Textures subpath (relative to output)",
            default = "Textures")
    objectsPath = StringProperty(
            name = "Default Objects subpath",
            description = "Objects subpath (relative to output)",
            default = "Objects")
    scenesPath = StringProperty(
            name = "Default Scenes subpath",
            description = "Scenes subpath (relative to output)",
            default = "Scenes")

    bonesPerGeometry = IntProperty(
            name = "Per geometry",
            description = "Max numbers of bones per geometry",
            min = 64, max = 2048,
            default = 64)
    bonesPerVertex = IntProperty(
            name = "Per vertex",
            description = "Max numbers of bones per vertex",
            min = 4, max = 256,
            default = 4)

    reportWidth = IntProperty(
            name = "Window width",
            description = "Width of the report window",
            default = 500)

    maxMessagesCount = IntProperty(
            name = "Max number of messages",
            description = "Max number of messages in the report window",
            default = 500)
            
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "outputPath")
        layout.prop(self, "modelsPath")
        layout.prop(self, "animationsPath")
        layout.prop(self, "materialsPath")
        layout.prop(self, "techniquesPath")
        layout.prop(self, "texturesPath")
        layout.prop(self, "objectsPath")
        layout.prop(self, "scenesPath")
        row = layout.row()
        row.label("Max number of bones:")
        row.prop(self, "bonesPerGeometry")
        row.prop(self, "bonesPerVertex")
        row = layout.row()
        row.label("Report window:")
        row.prop(self, "reportWidth")
        row.prop(self, "maxMessagesCount")


# Here we define all the UI objects we'll add in the export panel
class UrhoExportSettings(bpy.types.PropertyGroup):

    # This is called each time a property (created with the parameter 'update')
    # changes its value
    def update_func(self, context):
        # Avoid infinite recursion
        if self.updatingProperties:
            return
        self.updatingProperties = True

        # Save preferred output path
        addonPrefs = context.user_preferences.addons[__name__].preferences
        if self.outputPath:
            addonPrefs.outputPath = self.outputPath
        # Skeleton implies weights    
        if self.skeletons:
            self.geometryWei = True
        else:
            self.geometryWei = False
            self.animations = False
        # Morphs need geometries    
        if not self.geometries:
            self.morphs = False
        # Tangent needs position, normal and UV
        if not self.geometryPos or not self.geometryNor or not self.geometryUV:
            self.geometryTan = False
        # Morph normal needs geometry normal
        if not self.geometryNor:
            self.morphNor = False
        # Morph tangent needs geometry tangent
        if not self.geometryTan:
            self.morphTan = False
        # Morph tangent needs normal
        if not self.morphNor:
            self.morphTan = False
        # Select errors and merge are incompatible
        if self.selectErrors:
            self.merge = False
            
        self.updatingProperties = False

    def update_func2(self, context):
        if self.updatingProperties:
            return
        self.updatingProperties = True
        # Select errors and merge are incompatible
        if self.merge:
            self.selectErrors = False

        self.updatingProperties = False

    def errors_update_func(self, context):
        if self.updatingProperties:
            return
        self.updatingProperties = True
        errorName = self.errorsEnum
        self.errorsEnum = 'NONE'
        self.updatingProperties = False
        selectErrors(context, self.errorsMem, errorName)
        
    def errors_items_func(self, context):
        items = [('NONE', "", ""),
                 ('ALL',  "all", "")]
        for error in self.errorsMem.Names():
            items.append( (error, error, "") )
        return items

    # Revert all the export settings back to their default values
    def reset(self, context): 
        
        addonPrefs = context.user_preferences.addons[__name__].preferences

        self.minimize = False
        self.onlyErrors = False
        self.showDirs = False

        self.useSubDirs = True
        self.fileOverwrite = False

        self.source = 'ONLY_SELECTED'
        self.scale = 1.0
        self.modifiers = False
        self.modifiersRes = 'PREVIEW'
        self.origin = 'LOCAL'
        self.selectErrors = True
        self.forceElements = False
        self.merge = False
        self.mergeNotMaterials = False
        self.geometrySplit = False
        self.lods = False
        self.strictLods = True
        self.optimizeIndices = False

        self.skeletons = False
        self.onlyKeyedBones = False
        self.onlyDeformBones = False
        self.onlyVisibleBones = False
        self.parentBoneSkinning = False
        self.derigify = False

        self.animations = False
        self.animationSource = 'USED_ACTIONS'
        self.animationZero = True
        self.animationTriggers = False
        self.animationRatioTriggers = False
        self.animationPos = True
        self.animationRot = True
        self.animationSca = False

        self.geometries = True
        self.geometryPos = True
        self.geometryNor = True
        self.geometryCol = False
        self.geometryColAlpha = False
        self.geometryUV = False
        self.geometryUV2 = False
        self.geometryTan = False
        self.geometryWei = False

        self.morphs = False
        self.morphNor = True
        self.morphTan = False

        self.materials = False
        self.materialsList = False
        self.textures = False

        self.prefabs = True
        self.individualPrefab = False
        self.collectivePrefab = False
        self.scenePrefab = False
        self.physics = 'INDIVIDUAL'
        self.shape = 'TRIANGLEMESH'

    # Revert the output paths back to their default values
    def reset_paths(self, context, forced):

        addonPrefs = context.user_preferences.addons[__name__].preferences

        if forced or (not self.outputPath and addonPrefs.outputPath):
            self.outputPath = addonPrefs.outputPath

        if forced or (not self.modelsPath and addonPrefs.modelsPath):
            self.modelsPath = addonPrefs.modelsPath
            self.animationsPath = addonPrefs.animationsPath
            self.materialsPath = addonPrefs.materialsPath
            self.techniquesPath = addonPrefs.techniquesPath
            self.texturesPath = addonPrefs.texturesPath
            self.objectsPath = addonPrefs.objectsPath
            self.scenesPath = addonPrefs.scenesPath

    # --- Accessory ---

    updatingProperties = BoolProperty(default = False)

    minimize = BoolProperty(
            name = "Minimize",
            description = "Minimize the export panel",
            default = False)

    onlyErrors = BoolProperty(
            name = "Log errors",
            description = "Show only warnings and errors in the log",
            default = False)

    showDirs = BoolProperty(
            name = "Show dirs",
            description = "Show the dirs list",
            default = False)

    # --- Output settings ---
    
    outputPath = StringProperty(
            name = "",
            description = "Path where to export",
            default = "", 
            maxlen = 1024,
            subtype = "DIR_PATH",
            update = update_func)   

    useSubDirs = BoolProperty(
            name = "Use sub folders",
            description = "Use sub folders inside the output folder (Materials, Models, Textures ...)",
            default = True)

    modelsPath = StringProperty(
            name = "Models",
            description = "Models subpath (relative to output)")
    animationsPath = StringProperty(
            name = "Animations",
            description = "Animations subpath (relative to output)")
    materialsPath = StringProperty(
            name = "Materials",
            description = "Materials subpath (relative to output)")
    techniquesPath = StringProperty(
            name = "Techniques",
            description = "Techniques subpath (relative to output)")
    texturesPath = StringProperty(
            name = "Textures",
            description = "Textures subpath (relative to output)")
    objectsPath = StringProperty(
            name = "Objects",
            description = "Objects subpath (relative to output)")
    scenesPath = StringProperty(
            name = "Scenes",
            description = "Scenes subpath (relative to output)")

    fileOverwrite = BoolProperty(
            name = "Files overwrite",
            description = "If enabled existing files are overwritten without warnings",
            default = False)

    # --- Source settings ---
            
    source = EnumProperty(
            name = "Source",
            description = "Objects to be exported",
            items=(('ALL', "All", "all the objects in the scene"),
                   ('ONLY_SELECTED', "Only selected", "only the selected objects in visible layers")),
            default='ONLY_SELECTED')

    orientation = EnumProperty(
            name = "Front view",
            description = "Front view of the model",
            items = (('X_MINUS', "Left (--X +Z)", ""),
                     ('X_PLUS',  "Right (+X +Z)", ""),
                     ('Y_MINUS', "Front (--Y +Z)", ""),
                     ('Y_PLUS',  "Back (+Y +Z) *", ""),
                     ('Z_MINUS', "Bottom (--Z --Y)", ""),
                     ('Z_PLUS',  "Top (+Z +Y)", "")),
            default = 'X_PLUS')

    scale = FloatProperty(
            name = "Scale", 
            description = "Scale to apply on the exported objects", 
            default = 1.0,
            min = 0.0, 
            max = 1000.0,
            step = 10,
            precision = 1)

    modifiers = BoolProperty(
            name = "Apply modifiers",
            description = "Apply the object modifiers before exporting",
            default = False)

    modifiersRes = EnumProperty(
            name = "Modifiers setting",
            description = "Resolution setting to use while applying modifiers",
            items = (('PREVIEW', "Preview", "use the Preview resolution setting"),
                     ('RENDER', "Render", "use the Render resolution setting")),
            default = 'RENDER')

    origin = EnumProperty(
            name = "Mesh origin",
            description = "Origin for the position of vertices/bones",
            items=(('GLOBAL', "Global", "Blender's global origin"),
                   ('LOCAL', "Local", "object's local origin (orange dot)")),
            default = 'LOCAL')

    selectErrors = BoolProperty(
            name = "Select vertices with errors",
            description = "If a vertex has errors (e.g. invalid UV, missing UV or color or weights) select it",
            default = True,
            update = update_func)

    errorsMem = ErrorsMem()
    errorsEnum = EnumProperty(
            name = "",
            description = "List of errors",
            items = errors_items_func,
            update = errors_update_func)

    forceElements = BoolProperty(
            name = "Force missing elements",
            description = "If a vertex element (UV, color, weights) is missing add it with a zero value",
            default = False)

    merge = BoolProperty(
            name = "Merge objects",
            description = ("Merge all the objects in a single file, one common geometry for each material. "
                           "It uses the current object name."),
            default = False,
            update = update_func2)

    mergeNotMaterials = BoolProperty(
            name = "Don't merge materials",
            description = "Create a different geometry for each material of each object",
            default = False)

    geometrySplit = BoolProperty(
            name = "One vertex buffer per object",
            description = "Split each object into its own vertex buffer",
            default = False)

    lods = BoolProperty(
            name = "Use LODs",
            description = "Search for the LOD distance if the object name, objects with the same name are added as LODs",
            default = False)

    strictLods = BoolProperty(
            name = "Strict LODs",
            description = "Add a new vertex if the LOD0 does not contain a vertex with the exact same position, normal and UV",
            default = True)
            
    optimizeIndices = BoolProperty(
            name = "Optimize indices (slow)",
            description = "Linear-Speed vertex cache optimisation",
            default = True)

    # --- Components settings ---

    skeletons = BoolProperty(
            name = "Skeletons",
            description = "Export model armature bones",
            default = False,
            update = update_func)

    onlyKeyedBones = BoolProperty(
            name = "Only keyed bones",
            description = "In animinations export only bones with keys",
            default = False)

    onlyDeformBones = BoolProperty(
            name = "Only deform bones",
            description = "Don't export bones without Deform and its children",
            default = False,
            update = update_func2)
            
    onlyVisibleBones = BoolProperty(
            name = "Only visible bones",
            description = "Don't export bones not visible and its children",
            default = False,
            update = update_func2)

    derigify = BoolProperty(
            name = "Derigify",
            description = "Remove extra bones from Rigify armature",
            default = False,
            update = update_func)

    parentBoneSkinning = BoolProperty(
            name = "Use skinning for parent bones",
            description = "If an object has a parent of type BONE use a 100% skinning on its vertices "
                          "(use this only for a quick prototype)",
            default = False,
            update = update_func)

    animations = BoolProperty(
            name = "Animations",
            description = "Export animations (Skeletons needed)",
            default = False)

    animationSource = EnumProperty(
            name = "",
            items = (('ALL_ACTIONS', "All Actions", "Export all the actions in memory"),
                    ('USED_ACTIONS', "Actions used in tracks", "Export only the actions used in NLA tracks"),
                    ('SELECTED_ACTIONS', "Selected Strips' Actions", "Export the actions of the current selected NLA strips"),
                    ('SELECTED_STRIPS', "Selected Strips", "Export the current selected NLA strips"),
                    ('SELECTED_TRACKS', "Selected Tracks", "Export the current selected NLA tracks"),
                    ('ALL_STRIPS', "All Strips", "Export all NLA strips"),
                    ('ALL_TRACKS', "All Tracks (not muted)", "Export all NLA tracks"),
                    ('TIMELINE', "Timelime", "Export the timeline (NLA tracks sum)")),
            default = 'USED_ACTIONS',
            update = update_func)

    animationZero = BoolProperty(
            name = "Start at frame zero",
            description = "Force frame zero as the start of Actions, Tracks and Timeline. Otherwise use the first keyframe "
                          "for Actions or the playback start for Tracks and Timeline (Strips can only use their start)",
            default = True)

    animationTriggers = BoolProperty(
            name = "Export markers as triggers",
            description = "Export action pose markers (for actions, strips and tracks) or scene markers (for timeline) "
                          "as triggers, the time is expressed in seconds",
            default = False)

    animationRatioTriggers = BoolProperty(
            name = "Normalize time",
            description = "Export the time of triggers as a number from 0 (start) to 1 (end)",
            default = False)

    #---------------------------------

    animationPos = BoolProperty(
            name = "Position",
            description = "Within animations export bone positions",
            default = True)

    animationRot = BoolProperty(
            name = "Rotation",
            description = "Within animations export bone rotations",
            default = True)

    animationSca = BoolProperty(
            name = "Scale",
            description = "Within animations export bone scales",
            default = False)

    geometries = BoolProperty(
            name = "Geometries",
            description = "Export vertex buffers, index buffers, geometries, lods",
            default = True,
            update = update_func)

    geometryPos = BoolProperty(
            name = "Position",
            description = "Within geometry export vertex position",
            default = True,
            update = update_func)

    geometryNor = BoolProperty(
            name = "Normal",
            description = "Within geometry export vertex normal (enable 'Auto Smooth' to export custom normals)",
            default = True,
            update = update_func)

    geometryCol = BoolProperty(
            name = "Color",
            description = "Within geometry export vertex color",
            default = False)

    geometryColAlpha = BoolProperty(
            name = "Alpha",
            description = "Within geometry export vertex alpha (append _ALPHA to the color layer name)",
            default = False)

    geometryUV = BoolProperty(
            name = "UV",
            description = "Within geometry export vertex UV",
            default = False,
            update = update_func)

    geometryUV2 = BoolProperty(
            name = "UV2",
            description = "Within geometry export vertex UV2 (append _UV2 to the texture name)",
            default = False,
            update = update_func)

    geometryTan = BoolProperty(
            name = "Tangent",
            description = "Within geometry export vertex tangent (Position, Normal, UV needed)",
            default = False,
            update = update_func)

    geometryWei = BoolProperty(
            name = "Weights",
            description = "Within geometry export vertex bones weights (Skeletons needed)",
            default = False)

    morphs = BoolProperty(
            name = "Morphs (shape keys)",
            description = "Export vertex morphs (Geometries needed)",
            default = False)

    morphNor = BoolProperty(
            name = "Normal",
            description = "Within morph export vertex normal (Geometry Normal needed)",
            default = True,
            update = update_func)

    morphTan = BoolProperty(
            name = "Tangent",
            description = "Within morph export vertex tangent (Morph Normal, Geometry Tangent needed)",
            default = False,
            update = update_func)

    materials = BoolProperty(
            name = "Export materials",
            description = "Export XML materials",
            default = False,
            update = update_func)

    materialsList = BoolProperty(
            name = "Materials text list",
            description = "Write a txt file with the list of materials filenames",
            default = False)

    textures = BoolProperty(
            name = "Copy textures",
            description = "Copy diffuse textures",
            default = False,
            update = update_func)            

    prefabs = BoolProperty(
            name = "Export Urho Prefabs",
            description = "Export Urho3D XML objects (prefabs)",
            default = False,
            update = update_func)

    individualPrefab = BoolProperty(
            name = "Individual Prefabs",
            description = "Create one prefab per exported object (so if \"Merge objects\" option is checked, export one prefab for the merged object only)",
            default = False,
            update = update_func)

    collectivePrefab = BoolProperty(
            name = "One Collective",
            description = "Create one unic/global prefab containing every exported objects. An empty root node holds the objects.",
            default = False,
            update = update_func)

    scenePrefab = BoolProperty(
            name = "Scene Prefab",
            description = "Same content as 'Collective', but outputs a Urho3D xml scene (with Octree, PhysicsWorld and DebugRenderer)",
            default = False,
            update = update_func)

    physics = EnumProperty(
            name = "Physics",
            description = "Generate physics RigidBody(s) & Shape(s)",
            items = (('DISABLE', "No physics", "Do not create physics stuff"),
                        ('GLOBAL', "Global", "Create a unic RigidBody + Shape at the root. Expects a 'Physics.mdl' model as TriangleMesh."),
                        ('INDIVIDUAL', "Individual", "Create individual physics RigidBodies and Shapes")),
            default = 'INDIVIDUAL',
            update = update_func2)

    shapeItems = [ ('BOX', "Box", ""), ('CAPSULE', "Capsule", ""), ('CONE', "Cone", ""), \
                ('CONVEXHULL', "ConvexHull", ""), ('CYLINDER', "Cylinder", ""), ('SPHERE', "Sphere", ""), \
                ('STATICPLANE', "StaticPlane", ""), ('TRIANGLEMESH', "TriangleMesh", "") ]
    shape = EnumProperty(
            name = "CollisionShape",
            description = "CollisionShape type. Discarded if 'Collision Bounds' is checked in Physics panel.",
            items = shapeItems,
            default = 'TRIANGLEMESH',
            update = update_func2)

    bonesGlobalOrigin = BoolProperty(name = "Bones global origin", default = False)
    actionsGlobalOrigin = BoolProperty(name = "Actions global origin", default = False)
    

# Reset settings button
class UrhoExportResetOperator(bpy.types.Operator):
    """ Reset export settings """

    bl_idname = "urho.exportreset"
    bl_label = "Revert settings to default"

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_confirm(self, event)

    def execute(self, context):
        context.scene.urho_exportsettings.reset(context)
        return {'FINISHED'}


# Reset output paths button
class UrhoExportResetPathsOperator(bpy.types.Operator):
    """ Reset paths """

    bl_idname = "urho.exportresetpaths"
    bl_label = "Revert paths to default"

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_confirm(self, event)

    def execute(self, context):
        context.scene.urho_exportsettings.reset_paths(context, True)
        return {'FINISHED'}


# View log button
class UrhoReportDialog(bpy.types.Operator):
    """ View export log """
    
    bl_idname = "urho.report"
    bl_label = "Urho export report"
 
    def execute(self, context):
        return {'FINISHED'}
 
    def invoke(self, context, event):
        global logMaxCount
        wm = context.window_manager
        addonPrefs = context.user_preferences.addons[__name__].preferences
        logMaxCount = addonPrefs.maxMessagesCount
        return wm.invoke_props_dialog(self, width = addonPrefs.reportWidth)
        #return wm.invoke_popup(self, width = addonPrefs.reportWidth)
     
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        for line in logList:
            lines = line.split(":", 1)
            if lines[0] == 'CRITICAL':
                lineicon = 'RADIO'
            elif lines[0] == 'ERROR':
                lineicon = 'CANCEL'
            elif lines[0] == 'WARNING':
                lineicon = 'ERROR'
            elif lines[0] == 'INFO':
                lineicon = 'INFO'
            else:
                lineicon = 'TEXT'
            layout.label(text = lines[1], icon = lineicon)


# Export button
class UrhoExportOperator(bpy.types.Operator):
    """ Start exporting """
    
    bl_idname = "urho.export"
    bl_label = "Export"
  
    def execute(self, context):
        ExecuteAddon(context)
        return {'FINISHED'}
 
    def invoke(self, context, event):
        return self.execute(context)


# The export panel, here we draw the panel using properties we have created earlier
class UrhoExportRenderPanel(bpy.types.Panel):
    
    bl_idname = "urho.exportrenderpanel"
    bl_label = "Urho export"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    #bl_options = {'DEFAULT_CLOSED'}
    
    # Draw the export panel
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.urho_exportsettings

        row = layout.row()
        #row=layout.row(align=True)
        minimizeIcon = 'ZOOMIN' if settings.minimize else 'ZOOMOUT'
        row.prop(settings, "minimize", text="", icon=minimizeIcon, toggle=False)
        row.operator("urho.export", icon='EXPORT')
        #split = layout.split(percentage=0.1)
        if sys.platform.startswith('win'):
            row.operator("wm.console_toggle", text="", icon='CONSOLE')
        row.prop(settings, "onlyErrors", text="", icon='FORCE_WIND')
        row.operator("urho.report", text="", icon='TEXT')
        if settings.minimize:
            return

        row = layout.row()
        row.label("Output:")
        row.operator("urho.exportresetpaths", text="", icon='LIBRARY_DATA_DIRECT')

        box = layout.box()

        box.label("Output folder:")
        box.prop(settings, "outputPath")
        box.prop(settings, "fileOverwrite")
        row = box.row()
        row.prop(settings, "useSubDirs")
        showDirsIcon = 'ZOOMOUT' if settings.showDirs else 'ZOOMIN'
        row.prop(settings, "showDirs", text="", icon=showDirsIcon, toggle=False)
        if settings.showDirs:
            dbox = box.box()
            dbox.prop(settings, "modelsPath")
            dbox.prop(settings, "animationsPath")
            dbox.prop(settings, "materialsPath")
            dbox.prop(settings, "techniquesPath")
            dbox.prop(settings, "texturesPath")
            dbox.prop(settings, "objectsPath")
            dbox.prop(settings, "scenesPath")

        row = layout.row()
        row.label("Settings:")
        row.operator("urho.exportreset", text="", icon='LIBRARY_DATA_DIRECT')
        
        box = layout.box()

        row = box.row()
        row.label("Objects:")
        row.prop(settings, "source", expand=True)

        row = box.row()
        row.label("Origin:")
        row.prop(settings, "origin", expand=True)

        box.prop(settings, "orientation")
        box.prop(settings, "scale")
        
        box.prop(settings, "modifiers")
        if settings.modifiers:
            row = box.row()
            row.separator()
            row.prop(settings, "modifiersRes", expand=True)

        row = box.row()
        row.prop(settings, "selectErrors")
        row.prop(settings, "errorsEnum")
        
        box.prop(settings, "forceElements")
        box.prop(settings, "merge")
        if settings.merge:
            row = box.row()
            row.separator()
            row.prop(settings, "mergeNotMaterials")
        box.prop(settings, "geometrySplit")
        box.prop(settings, "optimizeIndices")
        box.prop(settings, "lods")
        if settings.lods:
            row = box.row()
            row.separator()
            row.prop(settings, "strictLods")

        box = layout.box()

        row = box.row()
        row.prop(settings, "skeletons")
        row.label("", icon='BONE_DATA')
        if settings.skeletons:
            row = box.row()
            row.separator()
            col = row.column()
            col.prop(settings, "derigify")
            #col.prop(settings, "bonesGlobalOrigin")
            #col.prop(settings, "actionsGlobalOrigin")
            col.prop(settings, "onlyDeformBones")
            col.prop(settings, "onlyVisibleBones")
            col.prop(settings, "parentBoneSkinning")

        row = box.row()
        row.enabled = settings.skeletons
        row.prop(settings, "animations")
        row.label("", icon='ANIM_DATA')
        if settings.skeletons and settings.animations:
            row = box.row()
            row.separator()
            column = row.column()
            column.prop(settings, "animationSource")
            column.prop(settings, "animationZero")
            column.prop(settings, "animationTriggers")
            if settings.animationTriggers:
                row = column.row()
                row.separator()
                row.prop(settings, "animationRatioTriggers")
            column.prop(settings, "onlyKeyedBones")
            row = column.row()
            row.prop(settings, "animationPos")
            row.prop(settings, "animationRot")
            row.prop(settings, "animationSca")
        
        row = box.row()
        row.prop(settings, "geometries")
        row.label("", icon='MESH_DATA')
        if settings.geometries:
            row = box.row()
            row.separator()
            row.prop(settings, "geometryPos")
            row.prop(settings, "geometryNor")
            
            row = box.row()
            row.separator()
            row.prop(settings, "geometryUV")
            row.prop(settings, "geometryUV2")

            row = box.row()
            row.separator()
            col = row.column()
            col.enabled = settings.geometryPos and settings.geometryNor and settings.geometryUV
            col.prop(settings, "geometryTan")
            col = row.column()
            col.enabled = settings.skeletons
            col.prop(settings, "geometryWei")
            
            row = box.row()
            row.separator()
            row.prop(settings, "geometryCol")
            row.prop(settings, "geometryColAlpha")
        
        row = box.row()
        row.enabled = settings.geometries
        row.prop(settings, "morphs")
        row.label("", icon='SHAPEKEY_DATA')
        if settings.geometries and settings.morphs:
            row = box.row()
            row.separator()
            col = row.column()
            col.enabled = settings.geometryNor
            col.prop(settings, "morphNor")
            col = row.column()
            col.enabled = settings.morphNor and settings.geometryTan
            col.prop(settings, "morphTan")

        row = box.row()
        row.prop(settings, "materials")
        row.label("", icon='MATERIAL_DATA')
        if settings.materials:
            row = box.row()
            row.separator()
            row.prop(settings, "materialsList")

        row = box.row()
        row.prop(settings, "textures")
        row.label("", icon='TEXTURE_DATA')

        row = box.row()
        row.prop(settings, "prefabs")
        row.label("", icon='MOD_OCEAN')

        if settings.prefabs:
            row = box.row()
            row.separator()
            row.prop(settings, "individualPrefab")
            row.label("", icon='MOD_BUILD')

            if not settings.merge:
                row = box.row()
                row.separator()
                row.prop(settings, "collectivePrefab")
                row.label("", icon='URL')

            row = box.row()
            row.separator()
            row.prop(settings, "scenePrefab")
            row.label("", icon='WORLD')

            row = box.row()
            row.separator()
            row.prop(settings, "physics")
            row.label("", icon='PHYSICS')

            row = box.row()
            row.separator()
            row.prop(settings, "shape")
            row.label("", icon='GROUP')


#--------------------
# Handlers
#--------------------

# Called after loading a new blend. Set the default path if the path edit box is empty.        
@persistent
def PostLoad(dummy):
    addonPrefs = bpy.context.user_preferences.addons[__name__].preferences
    settings = bpy.context.scene.urho_exportsettings
    settings.errorsMem.Clear()
    settings.updatingProperties = False
    settings.reset_paths(bpy.context, False)


#--------------------
# Register Unregister
#--------------------

# Called when the addon is enabled. Here we register out UI classes so they can be 
# used by Python scripts.
def register():
    if DEBUG: print("Urho export register")
    
    #bpy.utils.register_module(__name__)
        
    bpy.utils.register_class(UrhoAddonPreferences)
    bpy.utils.register_class(UrhoExportSettings)
    bpy.utils.register_class(UrhoExportOperator)
    bpy.utils.register_class(UrhoExportResetOperator)
    bpy.utils.register_class(UrhoExportResetPathsOperator)
    bpy.utils.register_class(UrhoExportRenderPanel)
    bpy.utils.register_class(UrhoReportDialog)
    
    bpy.types.Scene.urho_exportsettings = bpy.props.PointerProperty(type=UrhoExportSettings)
    
    bpy.context.user_preferences.filepaths.use_relative_paths = False
    
    if not PostLoad in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(PostLoad)


# Note: the script __init__.py is executed only the first time the addons is enabled. After that
# disabling or enabling the script will only call unregister() or register(). So in unregister()
# delete only objects created with register(), do not delete global objects as they will not be
# created re-enabling the addon.
# __init__.py is re-executed pressing F8 or randomly(?) enabling the addon.

# Called when the addon is disabled. Here we remove our UI classes.
def unregister():
    if DEBUG: print("Urho export unregister")
    
    #bpy.utils.unregister_module(__name__)
    
    bpy.utils.unregister_class(UrhoAddonPreferences)
    bpy.utils.unregister_class(UrhoExportSettings)
    bpy.utils.unregister_class(UrhoExportOperator)
    bpy.utils.unregister_class(UrhoExportResetOperator)
    bpy.utils.unregister_class(UrhoExportResetPathsOperator)    
    bpy.utils.unregister_class(UrhoExportRenderPanel)
    bpy.utils.unregister_class(UrhoReportDialog)
    
    del bpy.types.Scene.urho_exportsettings
    
    if PostLoad in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(PostLoad)


#--------------------
# Blender UI utility
#--------------------

# Select vertices on a object
def selectVertices(context, objectName, indicesList, deselect):

    objects = context.scene.objects
    
    try:
        obj = objects[objectName]
    except KeyError:
        log.error( "Cannot select vertices on not found object {:s}".format(objectName) )
        return

    # Set the object as current
    objects.active = obj
    # Enter Edit mode (check poll() to avoid exception)
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    # Deselect all
    if deselect and bpy.ops.mesh.select_all.poll():
        bpy.ops.mesh.select_all(action='DESELECT')
    # Save the current select mode
    sel_mode = bpy.context.tool_settings.mesh_select_mode
    # Set Vertex select mode
    bpy.context.tool_settings.mesh_select_mode = [True, False, False]
    # Exit Edit mode
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
    # Select the vertices
    mesh = obj.data
    for index in indicesList:
        try:
            mesh.vertices[index].select = True
        #except KeyError:
        except IndexError:
            pass
    # Back in Edit mode
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    # Restore old selection mode
    bpy.context.tool_settings.mesh_select_mode = sel_mode 

from collections import defaultdict

# Select vertices with errors
def selectErrors(context, errorsMem, errorName):
    names = errorsMem.Names()
    if errorName != 'ALL':
        names = [errorName]
    for i, name in enumerate(names):
        errors = errorsMem.Get(name)
        if not errors:
            continue
        indices = defaultdict(set)
        for mi, vi in errors:
            objectName = errorsMem.Second(mi)
            if objectName:
                indices[objectName].add(vi)
        for objectName, indicesSet in indices.items():
            count = len(indicesSet)
            if count:
                log.info( "Selecting {:d} vertices on {:s} with '{:s}' errors".format(count, objectName, name) )
                selectVertices(context, objectName, indicesSet, i == 0)


#-------------------------------------------------------------------------
# Export main
#-------------------------------------------------------------------------
    
def ExecuteUrhoExport(context):
    global logList

    # Clear log list
    logList[:] = []
    
    # Get exporter UI settings
    settings = context.scene.urho_exportsettings

    # File utils options
    fOptions = FOptions()

    # List where to store tData (decomposed objects)
    tDataList = []
    # Decompose options
    tOptions = TOptions()

    # Scene export data
    uScene = UrhoScene(context.scene)
    # Scene export options
    sOptions = SOptions()
    
    # Addons preferences
    addonPrefs = context.user_preferences.addons[__name__].preferences
    
    # Copy from exporter UI settings to Decompose options
    tOptions.mergeObjects = settings.merge
    tOptions.mergeNotMaterials = settings.mergeNotMaterials
    tOptions.doForceElements = settings.forceElements
    tOptions.useLods = settings.lods
    tOptions.onlySelected = (settings.source == 'ONLY_SELECTED')
    tOptions.scale = settings.scale
    tOptions.globalOrigin = (settings.origin == 'GLOBAL')
    tOptions.applyModifiers = settings.modifiers
    tOptions.applySettings = settings.modifiersRes
    tOptions.doBones = settings.skeletons
    tOptions.doOnlyKeyedBones = settings.onlyKeyedBones
    tOptions.doOnlyDeformBones = settings.onlyDeformBones
    tOptions.doOnlyVisibleBones = settings.onlyVisibleBones
    tOptions.skinBoneParent = settings.parentBoneSkinning
    tOptions.derigifyArmature = settings.derigify
    tOptions.doAnimations = settings.animations
    tOptions.doAllActions = (settings.animationSource == 'ALL_ACTIONS')
    tOptions.doUsedActions = (settings.animationSource == 'USED_ACTIONS')
    tOptions.doSelectedActions = (settings.animationSource == 'SELECTED_ACTIONS')
    tOptions.doSelectedStrips = (settings.animationSource == 'SELECTED_STRIPS')
    tOptions.doSelectedTracks = (settings.animationSource == 'SELECTED_TRACKS')
    tOptions.doStrips = (settings.animationSource == 'ALL_STRIPS')
    tOptions.doTracks = (settings.animationSource == 'ALL_TRACKS')
    tOptions.doTimeline = (settings.animationSource == 'TIMELINE')
    tOptions.doTriggers = settings.animationTriggers
    tOptions.doAnimationZero = settings.animationZero
    tOptions.doAnimationPos = settings.animationPos
    tOptions.doAnimationRot = settings.animationRot
    tOptions.doAnimationSca = settings.animationSca
    tOptions.doGeometries = settings.geometries
    tOptions.doGeometryPos = settings.geometryPos
    tOptions.doGeometryNor = settings.geometryNor
    tOptions.doGeometryCol = settings.geometryCol
    tOptions.doGeometryColAlpha = settings.geometryColAlpha
    tOptions.doGeometryUV  = settings.geometryUV
    tOptions.doGeometryUV2  = settings.geometryUV2
    tOptions.doGeometryTan = settings.geometryTan
    tOptions.doGeometryWei = settings.geometryWei
    tOptions.doMorphs = settings.morphs
    tOptions.doMorphNor = settings.morphNor
    tOptions.doMorphTan = settings.morphTan
    tOptions.doMorphUV = settings.morphTan
    tOptions.doOptimizeIndices = settings.optimizeIndices
    tOptions.doMaterials = settings.materials or settings.textures
    tOptions.bonesGlobalOrigin = settings.bonesGlobalOrigin
    tOptions.actionsGlobalOrigin = settings.actionsGlobalOrigin

    tOptions.orientation = None # ='Y_PLUS'
    if settings.orientation == 'X_PLUS':
        tOptions.orientation = Quaternion((0.0,0.0,1.0), radians(90.0))
    elif settings.orientation == 'X_MINUS':
        tOptions.orientation = Quaternion((0.0,0.0,1.0), radians(-90.0))
    elif settings.orientation == 'Y_MINUS':
        tOptions.orientation = Quaternion((0.0,0.0,1.0), radians(180.0))
    elif settings.orientation == 'Z_PLUS':
        tOptions.orientation = Quaternion((1.0,0.0,0.0), radians(-90.0)) * Quaternion((0.0,0.0,1.0), radians(180.0))
    elif settings.orientation == 'Z_MINUS':
        tOptions.orientation = Quaternion((1.0,0.0,0.0), radians(90.0)) * Quaternion((0.0,0.0,1.0), radians(180.0))

    sOptions.shapeItems = settings.shapeItems
    for shapeItems in settings.shapeItems:
        if shapeItems[0] == settings.shape:
            tOptions.shape = shapeItems[1]
            sOptions.shape = shapeItems[1]
            break

    sOptions.mergeObjects = settings.merge
    sOptions.doIndividualPrefab = settings.individualPrefab
    sOptions.doCollectivePrefab = settings.collectivePrefab
    sOptions.doScenePrefab = settings.scenePrefab
    sOptions.noPhysics = (settings.physics == 'DISABLE')
    sOptions.individualPhysics = (settings.physics == 'INDIVIDUAL')
    sOptions.globalPhysics = (settings.physics == 'GLOBAL')

    fOptions.useSubDirs = settings.useSubDirs
    fOptions.fileOverwrite = settings.fileOverwrite
    fOptions.paths[PathType.ROOT] = settings.outputPath
    fOptions.paths[PathType.MODELS] = settings.modelsPath
    fOptions.paths[PathType.ANIMATIONS] = settings.animationsPath
    fOptions.paths[PathType.TRIGGERS] = settings.animationsPath
    fOptions.paths[PathType.MATERIALS] = settings.materialsPath
    fOptions.paths[PathType.TECHNIQUES] = settings.techniquesPath
    fOptions.paths[PathType.TEXTURES] = settings.texturesPath
    fOptions.paths[PathType.MATLIST] = settings.modelsPath
    fOptions.paths[PathType.OBJECTS] = settings.objectsPath
    fOptions.paths[PathType.SCENES] = settings.scenesPath

    settings.errorsMem.Clear()

    if settings.onlyErrors:
        log.setLevel(logging.WARNING)
    else:
        log.setLevel(logging.DEBUG)

    if not settings.outputPath:
        log.error( "Output path is not set" )
        return False

    if tOptions.mergeObjects and not tOptions.globalOrigin:
        log.warning("Probably you should use Origin = Global")

    # Decompose
    if DEBUG: ttt = time.time() #!TIME
    Scan(context, tDataList, settings.errorsMem, tOptions)
    if DEBUG: print("[TIME] Decompose in {:.4f} sec".format(time.time() - ttt) ) #!TIME

    # Export each decomposed object
    for tData in tDataList:
    
        #PrintAll(tData)
        
        log.info("---- Exporting {:s} ----".format(tData.objectName))

        uExportData = UrhoExportData()
        
        uExportOptions = UrhoExportOptions()
        uExportOptions.splitSubMeshes = settings.geometrySplit
        uExportOptions.useStrictLods = settings.strictLods
        uExportOptions.useRatioTriggers = settings.animationRatioTriggers
        uExportOptions.bonesPerGeometry = addonPrefs.bonesPerGeometry
        uExportOptions.bonesPerVertex = addonPrefs.bonesPerVertex

        if DEBUG: ttt = time.time() #!TIME
        UrhoExport(tData, uExportOptions, uExportData, settings.errorsMem)
        if DEBUG: print("[TIME] Export in {:.4f} sec".format(time.time() - ttt) ) #!TIME
        if DEBUG: ttt = time.time() #!TIME

        uScene.Load(uExportData, tData.blenderObjectName)

        for uModel in uExportData.models:
            if uModel.geometries:
                filepath = GetFilepath(PathType.MODELS, uModel.name, fOptions)
                uScene.AddFile(PathType.MODELS, uModel.name, filepath[1])
                if CheckFilepath(filepath[0], fOptions):
                    log.info( "Creating model {:s}".format(filepath[1]) )
                    UrhoWriteModel(uModel, filepath[0]) 
            
        for uAnimation in uExportData.animations:
            filepath = GetFilepath(PathType.ANIMATIONS, uAnimation.name, fOptions)
            uScene.AddFile(PathType.ANIMATIONS, uAnimation.name, filepath[1])
            if CheckFilepath(filepath[0], fOptions):
                log.info( "Creating animation {:s}".format(filepath[1]) )
                UrhoWriteAnimation(uAnimation, filepath[0])

            if uAnimation.triggers:
                filepath = GetFilepath(PathType.TRIGGERS, uAnimation.name, fOptions)
                uScene.AddFile(PathType.TRIGGERS, uAnimation.name, filepath[1])
                if CheckFilepath(filepath[0], fOptions):
                    log.info( "Creating triggers {:s}".format(filepath[1]) )
                    UrhoWriteTriggers(uAnimation.triggers, filepath[0], fOptions)
                
        for uMaterial in uExportData.materials:
            for textureName in uMaterial.getTextures():
                # Check the texture name (it can be a filename)
                if textureName is None:
                    continue
                # Check if the Blender image data exists
                image = bpy.data.images[textureName]
                if image is None:
                    continue
                # Get the texture file full path
                srcFilename = bpy.path.abspath(image.filepath)
                # Get image filename
                filename = os.path.basename(image.filepath)
                if not filename:
                    filename = textureName
                # Get the destination file full path (preserve the extension)
                fOptions.preserveExtTemp = True
                filepath = GetFilepath(PathType.TEXTURES, filename, fOptions)
                # Check if already exported
                if not uScene.AddFile(PathType.TEXTURES, textureName, filepath[1]):
                    continue
                # Copy or unpack the texture
                if settings.textures and CheckFilepath(filepath[0], fOptions):
                    if image.packed_file:
                        format = str(context.scene.render.image_settings.file_format)
                        mode = str(context.scene.render.image_settings.color_mode)
                        log.info( "Unpacking {:s} {:s} texture to {:s}".format(format, mode, filepath[1]) )
                        render_ext = context.scene.render.file_extension
                        file_ext = os.path.splitext(filepath[0])[1].lower()
                        if file_ext and render_ext != file_ext:
                            log.warning( "Saving texture as {:s} but the file has extension {:s}".format(render_ext, file_ext) )
                        image.save_render(filepath[0])
                    elif not os.path.exists(srcFilename):
                        log.error( "Missing source texture {:s}".format(srcFilename) )
                    else:
                        try:
                            log.info( "Copying texture {:s}".format(filepath[1]) )
                            shutil.copyfile(src = srcFilename, dst = filepath[0])
                        except:
                            log.error( "Cannot copy texture to {:s}".format(filepath[0]) )

        if settings.materials:
            for uMaterial in uExportData.materials:
                filepath = GetFilepath(PathType.MATERIALS, uMaterial.name, fOptions)
                uScene.AddFile(PathType.MATERIALS, uMaterial.name, filepath[1])
                if CheckFilepath(filepath[0], fOptions):
                    log.info( "Creating material {:s}".format(filepath[1]) )
                    UrhoWriteMaterial(uScene, uMaterial, filepath[0], fOptions)
                    
            if settings.materialsList:
                for uModel in uExportData.models:
                    filepath = GetFilepath(PathType.MATLIST, uModel.name, fOptions)
                    uScene.AddFile(PathType.MATLIST, uModel.name, filepath[1])
                    if CheckFilepath(filepath[0], fOptions):
                        log.info( "Creating materials list {:s}".format(filepath[1]) )
                        UrhoWriteMaterialsList(uScene, uModel, filepath[0])

        if DEBUG: print("[TIME] Write in {:.4f} sec".format(time.time() - ttt) ) #!TIME

    settings.errorsMem.Cleanup()
    if settings.selectErrors:
        selectErrors(context, settings.errorsMem, 'ALL')
    
    # Export scene and nodes
    if settings.prefabs:
        UrhoExportScene(context, uScene, sOptions, fOptions)

    return True


def ExecuteAddon(context):

    startTime = time.time()
    print("----------------------Urho export start----------------------")    
    ExecuteUrhoExport(context)
    log.setLevel(logging.DEBUG)
    log.info("Export ended in {:.4f} sec".format(time.time() - startTime) )
    
    bpy.ops.urho.report('INVOKE_DEFAULT')

    
if __name__ == "__main__":
	register()
