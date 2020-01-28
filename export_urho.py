
#
# This script is licensed as public domain.
# Based on the Ogre Importer from the Urho3D project
#

from .utils import FloatToString, WriteXmlFile, BinaryFileWriter

from mathutils import Vector, Matrix, Quaternion
from math import cos, pi
from xml.etree import ElementTree as ET
from collections import defaultdict
import operator
import os
import random

import logging
log = logging.getLogger("ExportLogger")

#--------------------
# Urho enums
#--------------------

ELEMENT_POSITION    = 0x0001
ELEMENT_NORMAL      = 0x0002
ELEMENT_COLOR       = 0x0004
ELEMENT_UV1         = 0x0008
ELEMENT_UV2         = 0x0010
ELEMENT_CUBE_UV1    = 0x0020
ELEMENT_CUBE_UV2    = 0x0040
ELEMENT_TANGENT     = 0x0080
ELEMENT_BWEIGHTS    = 0x0100
ELEMENT_BINDICES    = 0x0200

ELEMENT_BLEND       = 0x0300

MORPH_ELEMENTS      = ELEMENT_POSITION | ELEMENT_NORMAL | ELEMENT_TANGENT

BONE_BOUNDING_SPHERE = 0x0001
BONE_BOUNDING_BOX    = 0x0002

TRACK_POSITION      = 0x0001
TRACK_ROTATION      = 0x0002
TRACK_SCALE         = 0x0004

TRIANGLE_LIST       = 0
LINE_LIST           = 1
            
# Max number of bones supported by HW skinning
MAX_SKIN_MATRICES   = 64
BONES_PER_VERTEX    = 4

#--------------------
# Float comparison
#--------------------

# Max difference between floats to be equal
EPSILON = 1e-6

# Max difference between floats to be equal
INFINITY = float("+inf")

# Returns True is v1 and v2 are both None or their corresponding elements are almost equal
def FloatListAlmostEqual(v1, v2):
    if v1 is None:
        return v2 is None
    if v2 is None:
        return False
    for e1, e2 in zip(v1, v2):
        if abs(e1 - e2) > EPSILON:
            return False
    return True

def RelativeAbs(e1, e2):
    if e1 == 0 and e2 == 0:
        return 0
    diff = abs(e1-e2)
    if diff < EPSILON:
        return 0
    return d / max(abs(e1),abs(e2))
    
def FloatListEqualError(v1, v2):
    if v1 is None:
        if v2 is None:
            return 0
        else:
            return INFINITY
    if v2 is None:
        return INFINITY
    #return sum(RelativeAbs(e1, e2) for e1, e2 in zip(v1, v2))
    return sum(abs(e1 - e2) for e1, e2 in zip(v1, v2))

def VectorDotProduct(v1, v2):
    if v1 is None:
        if v2 is None:
            return 1
        else:
            return -1
    if v2 is None:
        return -1
    return v1.dot(v2)

#--------------------
# Classes
#--------------------

# Bounding box axes aligned
class BoundingBox:
    def __init__(self):
        self.min = None # Vector((0.0, 0.0, 0.0))
        self.max = None # Vector((0.0, 0.0, 0.0))

    def merge(self, point):
        if self.min is None:
            self.min = point.copy()
            self.max = point.copy()
            return
        if point.x < self.min.x:
            self.min.x = point.x
        if point.y < self.min.y:
            self.min.y = point.y
        if point.z < self.min.z:
            self.min.z = point.z
        if point.x > self.max.x:
            self.max.x = point.x
        if point.y > self.max.y:
            self.max.y = point.y
        if point.z > self.max.z:
            self.max.z = point.z


# Exception for a vertex with more or less elements than its vertex buffer
class VertexMaskError(Exception):

    def __init__(self, oldMask, disruptMask):
        self.oldMask = oldMask
        self.disruptMask = disruptMask
        self.elements = { ELEMENT_POSITION: "Position",
                          ELEMENT_NORMAL: "Normal",
                          ELEMENT_COLOR: "Color",
                          ELEMENT_UV1: "UV1",
                          ELEMENT_UV2: "UV2",
                          ELEMENT_CUBE_UV1: "Cube UV1",
                          ELEMENT_CUBE_UV2: "Cube UV2",
                          ELEMENT_TANGENT: "Tangent",
                          ELEMENT_BWEIGHTS: "Blend weights",
                          ELEMENT_BINDICES: "Blend indices" }

    def __str__(self):
        diff = self.oldMask ^ self.disruptMask
        names = [name for mask,name in self.elements.items() if (mask & diff)]
        txt = "{:04X} vs {:04X}".format(self.oldMask, self.disruptMask)
        txt += "\n differences: {:s}".format(", ".join(names))
        return txt

# Exception for a keyframe with more or less elements than its track
class FrameMaskError(Exception):

    def __init__(self, oldMask, disruptMask, newMask):
        self.oldMask = oldMask
        self.disruptMask = disruptMask
        self.newMask = newMask
        self.elements = { TRACK_POSITION: "Position",
                          TRACK_ROTATION: "Rotation",
                          TRACK_SCALE: "Scale" }

    def __str__(self):
        diff = self.oldMask ^ self.disruptMask
        names = [name for mask,name in self.elements.items() if (mask & diff)]
        txt = "{:04X} AND {:04X} = {:04X}".format(self.oldMask, self.disruptMask, self.newMask)
        txt += "\n differences: {:s}".format(", ".join(names))
        return txt

# --- Model classes ---

class UrhoVertex:
    def __init__(self, tVertex):
        # Bit mask of elements present
        self.mask = 0
        # Only used by morphs, original vertex index in the not morphed vertex buffer
        self.index = None 
        # Vertex position: Vector((0.0, 0.0, 0.0)) of floats
        self.pos = tVertex.pos
        if tVertex.pos:
            self.mask |= ELEMENT_POSITION
        # Vertex normal: Vector((0.0, 0.0, 0.0)) of floats
        self.normal = tVertex.normal
        if tVertex.normal:
            self.mask |= ELEMENT_NORMAL
        # Vertex color: (0, 0, 0, 0) of unsigned bytes
        self.color = tVertex.color       
        if tVertex.color:
            self.mask |= ELEMENT_COLOR
        # Vertex UV texture coordinate: (0.0, 0.0) of floats
        self.uv = tVertex.uv
        if tVertex.uv:
            self.mask |= ELEMENT_UV1
        # Vertex UV2 texture coordinate: (0.0, 0.0) of floats
        self.uv2 = tVertex.uv2
        if tVertex.uv2:
            self.mask |= ELEMENT_UV2
        # Vertex tangent: Vector((0.0, 0.0, 0.0, 0.0)) of floats
        self.tangent = tVertex.tangent
        if tVertex.tangent:
            self.mask |= ELEMENT_TANGENT
        # List of tuples: blend weight (float), bone index (unsigned byte), mapped bone index (None if the bone is not mapped)
        self.weights = [(0.0, 0, None)] * BONES_PER_VERTEX
        if tVertex.weights is not None:
            # Sort tuples (index, weight) by decreasing weight
            sortedList = sorted(tVertex.weights, key = operator.itemgetter(1), reverse = True)
            sortedList = sortedList[:BONES_PER_VERTEX]
            # Normalize weights and add to the list
            totalWeight = sum([t[1] for t in sortedList])
            for i, t in enumerate(sortedList):
                self.weights[i] = (t[1] / totalWeight, t[0], None)
            self.mask |= ELEMENT_BLEND

    # used by the function index() of lists
    def __eq__(self, other):
        return (self.pos == other.pos and self.normal == other.normal and 
                self.color == other.color and self.uv == other.uv)

    # compare position, normal, color, UV, UV2 with another vertex, returns True is the error is insignificant
    def AlmostEqual(self, other):
        if not FloatListAlmostEqual(self.pos, other.pos):
            return False
        if not FloatListAlmostEqual(self.normal, other.normal):
            return False
        if self.color != other.color:
            return False
        if not FloatListAlmostEqual(self.uv, other.uv):
            return False
        if not FloatListAlmostEqual(self.uv2, other.uv2):
            return False
        return True

    # compare position, normal, UV with another vertex, returns the error
    def LodError(self, other):
        # If the position is not equal, return max error
        if not FloatListAlmostEqual(self.pos, other.pos):
            return INFINITY
        # If the angle between normals is above 30°, return max error (TODO: document this)
        ncos = VectorDotProduct(self.normal, other.normal)
        if ncos < cos(30 / 180 * pi):
            return INFINITY
        # UV are 0..1 x2, normals -1..1 x1, so this absolute error should be good 
        return (FloatListEqualError(self.uv, other.uv)  + 1-ncos)

    # not unique id of this vertex based on its position
    def __hash__(self):
        hashValue = 0
        if self.pos:
            hashValue ^= hash(self.pos.x) ^ hash(self.pos.y) ^ hash(self.pos.z)
        return hashValue
            
    # used by morph vertex calculations (see AnimatedModel::ApplyMorph)
    def subtract(self, other, mask):
        if mask & ELEMENT_POSITION:
            self.pos -= other.pos
        if mask & ELEMENT_NORMAL:
            self.normal -= other.normal
        if mask & ELEMENT_TANGENT:
            self.tangent -= other.tangent
            # tangent.w it is not modified by morphs (remember, there we
            # have saved bitangent direction)
            self.tangent.w = 0
            
class UrhoVertexBuffer:
    def __init__(self):
        # Flags of the elements contained in every vertex of this buffer
        self.elementMask = None
        # Morph min index and max index in the list vertices TODO: check
        self.morphMinIndex = None
        self.morphMaxIndex = None
        # List of UrhoVertex
        self.vertices = []

    # Check if a vertex is compatible with this buffer or has different elements
    def updateMask(self, vertexMask):
        # Update buffer mask
        if self.elementMask is None:
            self.elementMask = vertexMask
        elif self.elementMask != vertexMask:
            oldMask = self.elementMask
            # Update vertex buffer mask only if mask has the same and more bits
            if (self.elementMask & vertexMask) == self.elementMask:
                self.elementMask = vertexMask
            raise VertexMaskError(oldMask, vertexMask)

class UrhoIndexBuffer:
    def __init__(self):
        # Size of each index: 2 for 16 bits, 4 for 32 bits
        self.indexSize = 0
        # List of triples of indices (in the vertex buffer) to draw triangles
        self.indexes = []
    
class UrhoLodLevel:
    def __init__(self):
        # Distance above which we draw this LOD
        self.distance = 0.0
        # How to draw triangles: TRIANGLE_LIST, LINE_LIST 
        self.primitiveType = 0
        # Index of the vertex buffer used by this LOD in the model list
        self.vertexBuffer = 0
        # Index of the index buffer used by this LOD in the model list
        self.indexBuffer = 0
        # Pointer in the index buffer where starts this LOD
        self.startIndex = 0
        # Length in the index buffer to complete draw this LOD
        self.countIndex = 0

class UrhoGeometry:
    def __init__(self):
        # If the bones in the skeleton are too many for the hardware skinning, we
        # search for only the bones used by this geometry, then create a map from
        # the new bone index to the old bone index (in the skeleton)
        self.boneMap = []
        # List of UrhoLodLevel
        self.lodLevels = []
        # Geometry center based on the position of each triangle of the first LOD
        self.center = Vector((0.0, 0.0, 0.0))
        # Name of the material used (only for materials list)
        self.uMaterialName = None
        
class UrhoVertexMorph:
    def __init__(self):
         # Morph name
        self.name = None
        # Maps from 'vertex buffer index' to 'list of vertex', these are only the 
        # vertices modified by the morph, not all the vertices in the buffer (each 
        # morphed vertex has an index to the original vertex)
        self.vertexBufferMap = {}

class UrhoBone:
    def __init__(self):
        # Bone name
        self.name = None
        # Index of the parent bone in the model bones list
        self.parentIndex = None
        # Bone position in parent space
        self.position = None
        # Bone rotation in parent space
        self.rotation = None
        # Bone scale
        self.scale = Vector((1.0, 1.0, 1.0))
        # Bone transformation in skeleton space
        self.matrix = None
        # Inverse of the above
        self.inverseMatrix = None
        # Position in skeleton space
        self.derivedPosition = None
        # Collision sphere and/or box
        self.collisionMask = 0
        self.radius = None
        self.boundingBox = BoundingBox()
        self.length = 0

class UrhoModel:
    def __init__(self):
        # Model name
        self.name = None
        # List of UrhoVertexBuffer
        self.vertexBuffers = []
        # List of UrhoIndexBuffer
        self.indexBuffers = []
        # List of UrhoGeometry
        self.geometries = []
        # List of UrhoVertexMorph
        self.morphs = []
        # List of UrhoBone
        self.bones = []
        # Bounding box, contains each LOD of each geometry
        self.boundingBox = BoundingBox()
        
# --- Animation classes ---

class UrhoKeyframe:
    def __init__(self, tKeyframe):
        # Bit mask of elements present
        self.mask = 0
        # Time position in seconds: float
        self.time = tKeyframe.time
        # Position: Vector((0.0, 0.0, 0.0))
        self.position = tKeyframe.position
        if tKeyframe.position:
            self.mask |= TRACK_POSITION
        # Rotation: Quaternion()
        self.rotation = tKeyframe.rotation
        if tKeyframe.rotation:
            self.mask |= TRACK_ROTATION
        # Scale: Vector((0.0, 0.0, 0.0))
        self.scale = tKeyframe.scale
        if tKeyframe.scale:
            self.mask |= TRACK_SCALE
            
class UrhoTrack:
    def __init__(self):
        # Track name (practically same as the bone name that should be driven)
        self.name = ""
        # Mask of included animation data
        self.elementMask = None
        # Keyframes
        self.keyframes = []

    # Check if a vertex is compatible with this buffer or has different elements
    def updateMask(self, keyframeMask):
        # Update track mask
        if self.elementMask is None:
            self.elementMask = keyframeMask
        elif self.elementMask != keyframeMask:
            oldMask = self.elementMask
            # Update the track mask, keep the common denominator
            self.elementMask &= keyframeMask
            raise FrameMaskError(oldMask, keyframeMask, self.elementMask)

class UrhoTrigger:
    def __init__(self):
        # Trigger name 
        self.name = ""
        # Time in seconds: float
        self.time = None
        # Time as ratio: float
        self.ratio = None
        # Event data (variant, see typeNames[] in Variant.cpp)
        self.data = None

class UrhoAnimation:
    def __init__(self):
        # Animation name
        self.name = ""
        # Length in seconds: float
        self.length = 0.0
        # Tracks
        self.tracks = []
        # Animation triggers
        self.triggers = []

class UrhoMaterial:
    def __init__(self):
        # Material name
        self.name = None
        # Technique name
        self.techniqueName = None
        # Material diffuse color (0.0, 0.0, 0.0, 0.0) (r,g,b,a)
        self.diffuseColor = None
        # Material specular color (0.0, 0.0, 0.0, 0.0) (r,g,b,power)
        self.specularColor = None
        # Material emissive color (0.0, 0.0, 0.0) (r,g,b)
        self.emissiveColor = None
        # Material is two sided
        self.twoSided = False
        # Material is shadeless
        self.shadeless = False
        # Shader PS defines
        self.psdefines = ""
        # Shader VS defines
        self.vsdefines = ""
        # Textures names (no path, unique names)
        # keys: diffuse, specular, normal, emissive(\ao\lightmap)
        self.texturesNames = {}

    def getTextures(self):
        return list(self.texturesNames.values())

# --- Export options classes ---

class UrhoExportData:
    def __init__(self):
        # List of UrhoModel
        self.models = []
        # List of UrhoAnimation
        self.animations = []
        # List of UrhoMaterial
        self.materials = []
        
class UrhoExportOptions:
    def __init__(self):
        self.splitSubMeshes = False
        self.useStrictLods = True


#--------------------
# Writers
#--------------------
    
def UrhoWriteModel(model, filename):

    if not model.vertexBuffers or not model.indexBuffers or not model.geometries:
        log.error("No model data to export in {:s}".format(filename))
        return

    fw = BinaryFileWriter()
    try:
        fw.open(filename)
    except Exception as e:
        log.error("Cannot open file {:s} {:s}".format(filename, e))
        return

    # File Identifier
    fw.writeAsciiStr("UMDL")
    
    # Number of vertex buffers
    fw.writeUInt(len(model.vertexBuffers))
    # For each vertex buffer
    for buffer in model.vertexBuffers:
        # Vertex count
        fw.writeUInt(len(buffer.vertices))
        # Vertex element mask (determines vertex size)
        mask = buffer.elementMask
        fw.writeUInt(mask)
        # Morphable vertex range start index
        fw.writeUInt(buffer.morphMinIndex)
        # Morphable vertex count
        if buffer.morphMaxIndex != 0:
            fw.writeUInt(buffer.morphMaxIndex - buffer.morphMinIndex + 1)
        else:
            fw.writeUInt(0)
        # Vertex data (vertex count * vertex size)
        for vertex in buffer.vertices:
            if mask & ELEMENT_POSITION:
                fw.writeVector3(vertex.pos)
            if mask & ELEMENT_NORMAL:
                fw.writeVector3(vertex.normal)
            if mask & ELEMENT_COLOR:
                for i in range(4):
                    fw.writeUByte(vertex.color[i])
            if mask & ELEMENT_UV1:
                for i in range(2):
                    fw.writeFloat(vertex.uv[i])
            if mask & ELEMENT_UV2:
                for i in range(2):
                    fw.writeFloat(vertex.uv2[i])
            if mask & ELEMENT_TANGENT:
                fw.writeVector3(vertex.tangent)
                fw.writeFloat(vertex.tangent.w)
            if mask & ELEMENT_BWEIGHTS:
                for i in range(BONES_PER_VERTEX):
                    fw.writeFloat(vertex.weights[i][0])
            if mask & ELEMENT_BINDICES:
                for i in range(BONES_PER_VERTEX):
                    boneIndex = vertex.weights[i][1]
                    remappedBoneIndex = vertex.weights[i][2]
                    if remappedBoneIndex is not None:
                        boneIndex = remappedBoneIndex
                    fw.writeUByte(boneIndex)

    # Number of index buffers
    fw.writeUInt(len(model.indexBuffers))
    # For each index buffer
    for buffer in model.indexBuffers:
        # Index count
        fw.writeUInt(len(buffer.indexes))
        # Index size (2 for 16-bit indices, 4 for 32-bit indices)
        fw.writeUInt(buffer.indexSize)
        # Index data (index count * index size)
        for i in buffer.indexes:
            if buffer.indexSize == 2:
                fw.writeUShort(i)
            else:
                fw.writeUInt(i)

    # Number of geometries
    fw.writeUInt(len(model.geometries))
    # For each geometry
    for geometry in model.geometries:
        # Number of bone mapping entries
        fw.writeUInt(len(geometry.boneMap))
        # For each bone
        for bone in geometry.boneMap:
            fw.writeUInt(bone)
        # Number of LOD levels
        fw.writeUInt(len(geometry.lodLevels))
        # For each LOD level
        for lod in geometry.lodLevels:
            # LOD distance
            fw.writeFloat(lod.distance)
            # Primitive type (0 = triangle list, 1 = line list)
            fw.writeUInt(lod.primitiveType)
            # Vertex buffer index, starting from 0
            fw.writeUInt(lod.vertexBuffer)
            # Index buffer index, starting from 0
            fw.writeUInt(lod.indexBuffer)
            # Draw range: index start
            fw.writeUInt(lod.startIndex)
            # Draw range: index count
            fw.writeUInt(lod.countIndex)

    # Number of morphs
    fw.writeUInt(len(model.morphs))
    # For each morph
    for morph in model.morphs:
        # Name of morph
        fw.writeAsciiStr(morph.name)
        fw.writeUByte(0)
        # Number of affected vertex buffers
        fw.writeUInt(len(morph.vertexBufferMap))
        # For each affected vertex buffers
        for morphBufferIndex, morphBuffer in sorted(morph.vertexBufferMap.items()):
            # Vertex buffer index, starting from 0
            fw.writeUInt(morphBufferIndex)
            # Vertex element mask for morph data
            mask = (morphBuffer.elementMask & MORPH_ELEMENTS)
            fw.writeUInt(mask)
            # Vertex count
            fw.writeUInt(len(morphBuffer.vertices))
            # For each vertex:
            for vertex in morphBuffer.vertices:
                # Moprh vertex index
                fw.writeUInt(vertex.index)
                # Moprh vertex Position
                if mask & ELEMENT_POSITION:
                    fw.writeVector3(vertex.pos)
                # Moprh vertex Normal
                if mask & ELEMENT_NORMAL:
                    fw.writeVector3(vertex.normal)
                # Moprh vertex Tangent
                if mask & ELEMENT_TANGENT:
                    fw.writeVector3(vertex.tangent)
                    
    # Number of bones (may be 0)
    fw.writeUInt(len(model.bones))
    # For each bone
    for bone in model.bones:
        # Bone name
        fw.writeAsciiStr(bone.name)
        fw.writeUByte(0)
        # Parent bone index starting from 0
        fw.writeUInt(bone.parentIndex)
        # Initial position
        fw.writeVector3(bone.position)
        # Initial rotation
        fw.writeQuaternion(bone.rotation)
        # Initial scale
        fw.writeVector3(bone.scale)
        # 4x3 offset matrix for skinning
        for row in bone.inverseMatrix[:3]:
            for v in row:
                fw.writeFloat(v)
        # Bone collision info bitmask
        fw.writeUByte(bone.collisionMask)
        # Bone radius
        if bone.collisionMask & BONE_BOUNDING_SPHERE:
            fw.writeFloat(bone.radius)
        # Bone bounding box minimum and maximum
        if bone.collisionMask & BONE_BOUNDING_BOX:
            fw.writeVector3(bone.boundingBox.min)    
            fw.writeVector3(bone.boundingBox.max)    
         
    # Model bounding box minimum  
    fw.writeVector3(model.boundingBox.min)
    # Model bounding box maximum
    fw.writeVector3(model.boundingBox.max)

    # For each geometry
    for geometry in model.geometries:
        # Geometry center
        fw.writeVector3(geometry.center)
    
    fw.close()

    
def UrhoWriteAnimation(animation, filename):

    if not animation.tracks:
        log.error("No animation data to export in {:s}".format(filename))
        return

    fw = BinaryFileWriter()
    try:
        fw.open(filename)
    except Exception as e:
        log.error("Cannot open file {:s} {:s}".format(filename, e))
        return

    # File Identifier
    fw.writeAsciiStr("UANI")
    # Animation name
    fw.writeAsciiStr(animation.name)
    fw.writeUByte(0)
    # Length in seconds
    fw.writeFloat(animation.length)
    
    # Number of tracks
    fw.writeUInt(len(animation.tracks))
    # For each track
    for track in animation.tracks:
        # Track name (practically same as the bone name that should be driven)
        fw.writeAsciiStr(track.name)
        fw.writeUByte(0)
        # Mask of included animation data
        mask = track.elementMask
        fw.writeUByte(track.elementMask)
        
        # Number of tracks
        fw.writeUInt(len(track.keyframes))
        # For each keyframe
        for keyframe in track.keyframes:
            # Time position in seconds: float
            fw.writeFloat(keyframe.time)
            # Keyframe position
            if mask & TRACK_POSITION:
                fw.writeVector3(keyframe.position)
            # Keyframe rotation
            if mask & TRACK_ROTATION:
                fw.writeQuaternion(keyframe.rotation)
            # Keyframe scale
            if mask & TRACK_SCALE:
                fw.writeVector3(keyframe.scale)

    fw.close()

    
# As described in Animation::Load, Animation::Save
def UrhoWriteTriggers(triggersList, filename, fOptions):
    
    triggersElem = ET.Element('animation')

    for trigger in triggersList:
        triggerElem = ET.SubElement(triggersElem, "trigger")
        if trigger.time is not None:
            triggerElem.set("time", FloatToString(trigger.time))
        if trigger.ratio is not None:
            triggerElem.set("normalizedtime", FloatToString(trigger.ratio))
        # We use a string variant, for other types See typeNames[] in Variant.cpp 
        # and XMLElement::GetVariant()
        triggerElem.set("type", "String")
        triggerElem.set("value", str(trigger.data))

    WriteXmlFile(triggersElem, filename, fOptions)


#--------------------
# Utils
#--------------------

# Search for the most complete element mask
def GetMaxElementMask(indices, vertices):
    maxElementMask = 0
    maxElementMaskCount = 0
    for vertexIndex in indices:
        tVertex = vertices[vertexIndex]
        uVertex = UrhoVertex(tVertex)
        count = bin(uVertex.mask).count("1")
        if maxElementMaskCount < count:
            maxElementMaskCount = count
            maxElementMask = uVertex.mask
    if maxElementMask:
        return maxElementMask
    return None


#---------------------------------------

# NOTE: only different geometries use different buffers

# NOTE: LODs must use the same vertex buffer, and so the same vertices. This means
# normals and tangents are a bit off, but they are good infact they are approximations 
# of the first LOD which uses those vertices.
# Creating a LOD we search for the similar vertex.

# NOTE: vertex buffers can have different mask (ex. skeleton weights)

# NOTE: morph must have what geometry they refer to, or the vertex buffer or better
# the index buffer as vertex buffer is in common.
    
# NOTE: a morph can affect more than one vertex buffer

# NOTE: if a vertex buffer has blendweights then all its vertices must have it

# NOTE: if we use index() we must have __EQ__ in the class.
# NOTE: don't use index(), it's slow.

#--------------------
# Urho exporter
#--------------------

def UrhoExport(tData, uExportOptions, uExportData, errorsMem):

    global MAX_SKIN_MATRICES
    global BONES_PER_VERTEX
    if uExportOptions.bonesPerGeometry:
        MAX_SKIN_MATRICES = uExportOptions.bonesPerGeometry
    if uExportOptions.bonesPerVertex:
        BONES_PER_VERTEX = uExportOptions.bonesPerVertex

    uModel = UrhoModel()
    uModel.name = tData.objectName
    uExportData.models.append(uModel)    
    
    # For each bone
    for boneName, bone in tData.bonesMap.items():
        uBoneIndex = len(uModel.bones)
        # Sanity check for the OrderedDict
        assert bone.index == uBoneIndex
        
        uBone = UrhoBone()
        uModel.bones.append(uBone)
        
        uBone.name = boneName
        if bone.parentName:
            # Child bone
            uBone.parentIndex = tData.bonesMap[bone.parentName].index
        else:
            # Root bone
            uBone.parentIndex = uBoneIndex
        uBone.position = bone.bindPosition
        uBone.rotation = bone.bindRotation
        uBone.scale = bone.bindScale
        uBone.matrix = bone.worldTransform
        uBone.inverseMatrix = uBone.matrix.inverted()
        uBone.derivedPosition = uBone.matrix.to_translation()
        uBone.length = bone.length
    
    totalVertices = len(tData.verticesList) 
    
    # Search in geometries for the maximum number of vertices 
    maxLodVertices = 0
    for tGeometry in tData.geometriesList:
        for tLodLevel in tGeometry.lodLevels:
            vertexCount = len(tLodLevel.indexSet)
            if vertexCount > maxLodVertices:
                maxLodVertices = vertexCount
    
    # If one big buffer needs a 32 bits index but each geometry needs only a 16 bits
    # index then try to use a different buffer for each geometry
    useOneBuffer = True
    if uExportOptions.splitSubMeshes or (totalVertices > 65535 and maxLodVertices <= 65535):
        useOneBuffer = False

    # Urho lod vertex buffer
    vertexBuffer = None
    # Urho lod index buffer
    indexBuffer = None
    # Maps old vertex index to Urho vertex buffer index and Urho vertex index
    modelIndexMap = {}
    
    # For each geometry
    for tGeometry in tData.geometriesList:
        
        uGeometry = UrhoGeometry()
        uModel.geometries.append(uGeometry)
        geomIndex = len(uModel.geometries) - 1

        # Material name (can be None)
        uGeometry.uMaterialName = tGeometry.materialName
        
        # Start value for geometry center (one for each geometry)
        center = Vector((0.0, 0.0, 0.0))
        
        # Set of remapped vertices
        remappedVertices = set()
        
        # For each LOD level
        for lodIndex, tLodLevel in enumerate(tGeometry.lodLevels):
            uLodLevel = UrhoLodLevel()
            uGeometry.lodLevels.append(uLodLevel)
            
            if lodIndex == 0 and tLodLevel.distance != 0.0:
                # Note: if we miss a LOD, its range will be covered by the following LOD (which is this one),
                # this can cause overlapping between LODs of different geometries
                log.error("First LOD of object {:s} Geometry{:d} must have 0.0 distance (found {:.3f})"
                          .format(uModel.name, geomIndex, tLodLevel.distance))

            uLodLevel.distance = tLodLevel.distance
            uLodLevel.primitiveType = TRIANGLE_LIST

            # If needed add a new vertex buffer (only for first LOD of a geometry)
            # For remapping to work a geometry and its LODs must use only one buffer
            if vertexBuffer is None or (lodIndex == 0 and not useOneBuffer):
                vertexBuffer = UrhoVertexBuffer()
                uModel.vertexBuffers.append(vertexBuffer)
                uVerticesMap = {}

            # If needed add a new index buffer (only for first LOD of a geometry)
            if indexBuffer is None or (lodIndex == 0 and not useOneBuffer):
                indexBuffer = UrhoIndexBuffer()
                uModel.indexBuffers.append(indexBuffer)
                uLodLevel.startIndex = 0
            else:
                uLodLevel.startIndex = len(indexBuffer.indexes)    

            # Set how many indices the LOD level will use
            uLodLevel.countIndex = len(tLodLevel.triangleList) * 3
            # Set lod vertex and index buffers
            uLodLevel.vertexBuffer = len(uModel.vertexBuffers) - 1
            uLodLevel.indexBuffer = len(uModel.indexBuffers) - 1
            ##print("Geometry{:d} LOD{:d} using: vertex buffer {:d} ({:d}), index buffer {:d} ({:d})"
            ##      .format(geomIndex, lodIndex, uLodLevel.vertexBuffer, len(tLodLevel.indexSet),
            ##      uLodLevel.indexBuffer, uLodLevel.countIndex))
            
            # Maps old vertex index to new vertex index in the new Urho buffer
            indexMap = {}
            
            # Errors helpers
            warningNewVertices = False
            
            # Try to guess the most complete element mask
            randomIndices = random.sample(tLodLevel.indexSet, min(30, len(tLodLevel.indexSet)) )
            guessedElementMask = GetMaxElementMask(randomIndices, tData.verticesList)
            if vertexBuffer.elementMask is None and guessedElementMask:
                vertexBuffer.elementMask = guessedElementMask
                
            # Add vertices to the vertex buffer
            for tVertexIndex in tLodLevel.indexSet:
            
                tVertex = tData.verticesList[tVertexIndex]

                # Create a Urho vertex
                uVertex = UrhoVertex(tVertex)
                try:
                    vertexBuffer.updateMask(uVertex.mask)
                except VertexMaskError as e:
                    if not tVertex.blenderIndex is None:
                        errorsIndices = errorsMem.Get("element mask " + str(e), set() )
                        errorsIndices.add(tVertex.blenderIndex)
                    log.warning("Incompatible vertex elements in object {:s}, {!s}".format(uModel.name, e))

                # All that this code do is "uVertexIndex = vertexBuffer.vertices.index(uVertex)", but we use
                # a map to speed things up.
            
                # Get an hash of the vertex (more vertices can have the same hash)
                uVertexHash = hash(uVertex)
            
                try:
                    # Get the list of vertices indices with the same hash
                    uVerticesMapList = uVerticesMap[uVertexHash]
                except KeyError:
                    # If the hash is not mapped, create a new list (we could use a set but a list is faster)
                    uVerticesMapList = []
                    uVerticesMap[uVertexHash] = uVerticesMapList
                
                uVertexIndex = None
                if lodIndex == 0 or uExportOptions.useStrictLods:
                    # For each index in the list, get the corresponding vertex and test if it is equal to tVertex.
                    # If Position, Normal and UV are the same, it must be the same vertex, get its index.
                    for ivl in uVerticesMapList:
                        if vertexBuffer.vertices[ivl].AlmostEqual(uVertex):
                            uVertexIndex = ivl
                            break
                else:
                    # For successive LODs, we are more permissive, the vertex position must be the same, but for
                    # the normal and UV we will search the best match in the vertices available.
                    bestLodError = INFINITY
                    for ivl in uVerticesMapList:
                        lodError = vertexBuffer.vertices[ivl].LodError(uVertex)
                        if lodError < bestLodError:
                            bestLodError = lodError
                            uVertexIndex = ivl

                # If we cannot find it, the vertex is new, add it to the list, and its index to the map list
                if uVertexIndex is None:
                    uVertexIndex = len(vertexBuffer.vertices)
                    vertexBuffer.vertices.append(uVertex)
                    uVerticesMapList.append(uVertexIndex)
                    if lodIndex != 0:
                        warningNewVertices = True

                # Populate the 'old tVertex index' to 'new uVertex index' map
                if not tVertexIndex in indexMap:
                    indexMap[tVertexIndex] = uVertexIndex
                elif indexMap[tVertexIndex] != uVertexIndex:
                    log.error("Conflict in vertex index map of object {:s}".format(uModel.name))

                # Update the model bounding box (common to all geometries)
                if vertexBuffer.elementMask & ELEMENT_POSITION:
                    uModel.boundingBox.merge(uVertex.pos)

            if warningNewVertices:
                log.warning("LOD {:d} of object {:s} Geometry{:d} has new vertices."
                            .format(lodIndex, uModel.name, geomIndex))
                            
            # Add the local vertex map to the global map
            for oldIndex, newIndex in indexMap.items():
                # We create a map: Map[old index] = Set( Tuple(new buffer index, new vertex index) )
                # Search if this vertex index was already mapped, get its Set or add a new one.
                # We need a Set because a vertex can be copied in more than one vertex buffer.
                try:
                    vbviSet = modelIndexMap[oldIndex]
                except KeyError:
                    vbviSet = set()
                    modelIndexMap[oldIndex] = vbviSet
                # Add a tuple to the Set: new buffer index, new vertex index
                vbvi = (uLodLevel.vertexBuffer, newIndex)
                vbviSet.add(vbvi)
                
            # Add indices to the index buffer
            centerCount = 0
            for triangle in tLodLevel.triangleList:
                for tVertexIndex in triangle:
                    uVertexIndex = indexMap[tVertexIndex]
                    indexBuffer.indexes.append(uVertexIndex)
                    # Update geometry center (only for the first LOD)
                    if (lodIndex == 0) and (vertexBuffer.elementMask & ELEMENT_POSITION):
                        centerCount += 1
                        center += vertexBuffer.vertices[uVertexIndex].pos;

            # Update geometry center (only for the first LOD)
            if lodIndex == 0 and centerCount:
                uGeometry.center = center / centerCount;
                        
            # If this geometry has bone weights but the number of total bones is over the limit 
            # then let's hope our geometry uses only a subset of the total bones within the limit.
            # If this is true then we can remap the original bone index, which can be over the 
            # limit, to a local, in this geometry, bone index within the limit.
            if len(uModel.bones) > MAX_SKIN_MATRICES and (vertexBuffer.elementMask & ELEMENT_BLEND) == ELEMENT_BLEND:
                discardedBones = defaultdict(float)
                # For each vertex in the buffer
                for uVertexIndex in indexMap.values():
                    # Be sure to not pass a vertex again once its bones are remapped
                    if uVertexIndex in remappedVertices:
                        continue
                    remappedVertices.add(uVertexIndex)
                    vertex = vertexBuffer.vertices[uVertexIndex]
                    for j, (weight, boneIndex, unusedBoneIndex) in enumerate(vertex.weights):
                        if weight < EPSILON:
                            continue
                        # Search if the bone is already present in the map
                        try:
                            remappedBoneIndex = uGeometry.boneMap.index(boneIndex)
                        except ValueError:
                            # New bone, add it in the map
                            remappedBoneIndex = len(uGeometry.boneMap)
                            if remappedBoneIndex < MAX_SKIN_MATRICES:
                                uGeometry.boneMap.append(boneIndex)
                            else:
                                boneName = uModel.bones[boneIndex].name
                                discardedBones[boneName] += weight
                                remappedBoneIndex = 0
                                weight = 0.0
                        # Save the remapped local bone index (tuple[2]), do not replace the global bone index (tuple[1]) as
                        # we'll soon use it to compute the bone bounding box
                        vertex.weights[j] = (weight, boneIndex, remappedBoneIndex)
                bonesIn = len(uModel.bones)
                bonesOut = len(uGeometry.boneMap)
                bonesFree = MAX_SKIN_MATRICES - bonesOut
                print("Geometry{:d} remapping from {:d} bones to {:d} ({:d} free)"
                      .format(geomIndex, bonesIn, bonesOut, bonesFree))
                if discardedBones:
                    text = "Too many bones (+{:d}) in object {:s}: ".format(len(discardedBones), uModel.name)
                    for boneName in sorted(discardedBones, key=discardedBones.get, reverse=False):
                        text += " {:s}={:.2f}".format(boneName, discardedBones[boneName])
                    log.error(text)
            #LOD loop
        #Geometry loop
    #

    if tData.geometriesList and uModel.boundingBox.min is None:
        uModel.boundingBox.min = Vector((0.0, 0.0, 0.0))
        uModel.boundingBox.max = Vector((0.0, 0.0, 0.0))
        log.warning("Vertices of object {:s} have no position.".format(uModel.name))

    # Set index size for indexes buffers
    for uIndexBuffer in uModel.indexBuffers:
        if len(uIndexBuffer.indexes) > 65535:
            # 32 bits indexes
            uIndexBuffer.indexSize = 4
        else:
            # 16 bits indexes
            uIndexBuffer.indexSize = 2

    # Update bones bounding sphere and box
    # For each vertex buffer
    for uVertexBuffer in uModel.vertexBuffers:
        # Skip if the buffer doesn't have bone weights
        if (uVertexBuffer.elementMask & ELEMENT_BLEND) != ELEMENT_BLEND:
            continue
        # For each vertex in the buffer
        for uVertex in uVertexBuffer.vertices:
            vertexPos = uVertex.pos
            for weight, boneIndex, remappedBoneIndex in uVertex.weights:
                # The 0.33 threshold check is to avoid including vertices in the bone hitbox 
                # to which the bone contributes only a little. It is rather arbitrary. (Lasse)
                if weight > 0.33:
                    uBone = uModel.bones[boneIndex]
                    # Bone head position (in model space)
                    bonePos = uBone.derivedPosition
                    # Distance between vertex and bone head
                    distance = (bonePos - vertexPos).length
                    # Search for the maximum distance
                    if uBone.radius is None or distance > uBone.radius:
                        uBone.collisionMask |= BONE_BOUNDING_SPHERE
                        uBone.radius = distance
                    # Calculate the vertex position in bone space
                    boneVertexPos = uBone.inverseMatrix @ vertexPos
                    # Update the bone boundingBox
                    uBone.collisionMask |= BONE_BOUNDING_BOX
                    uBone.boundingBox.merge(boneVertexPos)

    # Do not allow bones bounding box to grow beyond head and tail
    if uExportOptions.clampBoundingBox:
        for bone in uModel.bones:
            if bone.collisionMask & BONE_BOUNDING_BOX:
                bone.boundingBox.min.y = 0.0
                bone.boundingBox.max.y = bone.length
            bone.radius = bone.length

    for tMorph in tData.morphsList:
        uMorph = UrhoVertexMorph()
        uMorph.name = tMorph.name
        uModel.morphs.append(uMorph)

        # Get 90 random vertices hoping to get some in all the vertex buffers
        guessingIndices = defaultdict(list)
        indicesAll = tMorph.vertexMap.keys()
        randomIndicesAll = random.sample(indicesAll, min(90, len(indicesAll)) )
        randomVertices = []
        for tVertexIndex in randomIndicesAll:
            # Get the correspondent Urho vertex buffer and vertex index (there can be more than one)
            vbviSet = modelIndexMap[tVertexIndex]
            # For each corresponding vertex buffer
            for uVertexBufferIndex, uVertexIndex in vbviSet:
                guessingIndices[uVertexBufferIndex].append(len(randomVertices))
                randomVertices.append(tMorph.vertexMap[tVertexIndex])
                
        # Try to guess the most complete element mask for each vertex buffer
        guessedElementMasks = {}
        for bufferIndex, randomIndices in guessingIndices.items():
            elementMask = GetMaxElementMask(randomIndices, randomVertices)
            if elementMask:
                guessedElementMasks[bufferIndex] = elementMask
                        
        # For each vertex affected by the morph
        for tVertexIndex, tMorphVertex in tMorph.vertexMap.items():
            # Get the correspondent Urho vertex buffer and vertex index (there can be more than one)
            vbviSet = modelIndexMap[tVertexIndex]
            # For each corresponding vertex buffer
            for uVertexBufferIndex, uVertexIndex in vbviSet:
                # Search for the vertex buffer in the morph, if not present add it
                try:
                    uMorphVertexBuffer = uMorph.vertexBufferMap[uVertexBufferIndex]
                except KeyError:
                    uMorphVertexBuffer = UrhoVertexBuffer()
                    uMorph.vertexBufferMap[uVertexBufferIndex] = uMorphVertexBuffer
                
                if uMorphVertexBuffer.elementMask is None and uVertexBufferIndex in guessedElementMasks:
                    uMorphVertexBuffer.elementMask = guessedElementMasks[uVertexBufferIndex]
                    
                # Create the morphed vertex
                uMorphVertex = UrhoVertex(tMorphVertex)
                try:
                    uMorphVertexBuffer.updateMask(uMorphVertex.mask)
                except VertexMaskError as e:
                    if not tVertex.blenderIndex is None:
                        errorsMorphIndices = errorsMem.Get("morph element mask " + str(e), set() )
                        errorsMorphIndices.add(tMorphVertex.blenderIndex)
                    log.warning("Incompatible vertex elements in morph {:s} of object {:s}, {!s}"
                                .format(uMorph.name, uModel.name, e))

                # Get the original vertex
                uVertexBuffer = uModel.vertexBuffers[uVertexBufferIndex]
                uVertex = uVertexBuffer.vertices[uVertexIndex]
                
                # Calculate morph values (pos, normal, tangent) relative to the original vertex
                uMorphVertex.subtract(uVertex, uMorphVertexBuffer.elementMask)
                    
                # Add the vertex to the morph buffer
                uMorphVertex.index = uVertexIndex
                uMorphVertexBuffer.vertices.append(uMorphVertex)

                # Update min and max morphed vertex index in the vertex buffer
                if uVertexBuffer.morphMinIndex is None:
                    uVertexBuffer.morphMinIndex = uVertexIndex
                    uVertexBuffer.morphMaxIndex = uVertexIndex
                elif uVertexIndex < uVertexBuffer.morphMinIndex:
                    uVertexBuffer.morphMinIndex = uVertexIndex
                elif uVertexIndex > uVertexBuffer.morphMaxIndex:
                    uVertexBuffer.morphMaxIndex = uVertexIndex

    # Set to zero min and max morphed vertex index of buffers with no morphs
    for i, uVertexBuffer in enumerate(uModel.vertexBuffers):
        if uVertexBuffer.morphMinIndex is None:
            uVertexBuffer.morphMinIndex = 0
            uVertexBuffer.morphMaxIndex = 0

            
    uAnimations = uExportData.animations
    for tAnimation in tData.animationsList:
        uAnimation = UrhoAnimation()
        uAnimation.name = tAnimation.name
        uAnimation.length = None
        
        for tTrack in tAnimation.tracks:
            uTrack = UrhoTrack()
            uTrack.name = tTrack.name
            
            for tFrame in tTrack.frames:
                uKeyframe = UrhoKeyframe(tFrame)
                try:
                    uTrack.updateMask(uKeyframe.mask)
                except FrameMaskError as e:
                    log.warning("Incompatible elements in track {:s} of animation {:s}, {!s}"
                                .format(uTrack.name, uAnimation.name, e))
                uTrack.keyframes.append(uKeyframe)

            # Make sure keyframes are sorted from beginning to end
            uTrack.keyframes.sort(key = operator.attrgetter('time'))

            # Add only tracks with keyframes
            if uTrack.keyframes and uTrack.elementMask:
                uAnimation.tracks.append(uTrack)
                # Update animation length
                length = uTrack.keyframes[-1].time
                if uAnimation.length is None or uAnimation.length < length:
                    uAnimation.length = length

        # Add the triggers for the animation
        for tTrigger in tAnimation.triggers:
            uTrigger = UrhoTrigger()
            uTrigger.name = tTrigger.name
            if uExportOptions.useRatioTriggers:
                uTrigger.ratio = tTrigger.ratio
            else:
                uTrigger.time = tTrigger.time
            uTrigger.data = tTrigger.data
            uAnimation.triggers.append(uTrigger)
                    
        # Add only animations with tracks
        if uAnimation.tracks:
            uAnimations.append(uAnimation)
    
    uMaterials = uExportData.materials
    for tMaterial in tData.materialsList:
        uMaterial = UrhoMaterial()
        # For material list to work the name must be the same 
        uMaterial.name = tMaterial.name
        uMaterial.texturesNames = tMaterial.texturesNames.copy()
        
        """
        alpha = 1.0
        if tMaterial.opacity:
            alpha = tMaterial.opacity

        isEmissive = False
        emissiveKey = None
        
        technique = "NoTexture"
        if "diffuse" in tMaterial.texturesNames:
            technique = "Diff"
            if "normal" in tMaterial.texturesNames:
                technique += "Normal"
            if "specular" in tMaterial.texturesNames:
                technique += "Spec"
            # Emission map, light map and AO (Ambient light map) use the same
            # emission texture slot, we have to pick one
            if "emissive" in tMaterial.texturesNames:
                emissiveKey = "emissive"
                technique += "Emissive"
                isEmissive = True
            elif "ao" in tMaterial.texturesNames:
                emissiveKey = "ao"
                technique += "AO"
            elif "lightmap" in tMaterial.texturesNames:
                emissiveKey = "lightmap"
                technique += "LightMap"
        if tMaterial.shadeless:
            technique += "Unlit";
        if tMaterial.opacity:
            technique += "Alpha";
            if tMaterial.alphaMask:
                uMaterial.psdefines += " ALPHAMASK"

        uMaterial.techniqueName = technique

        if tMaterial.diffuseColor:
            diffuse = tMaterial.diffuseColor * tMaterial.diffuseIntensity
            uMaterial.diffuseColor = (diffuse.r, diffuse.g, diffuse.b, alpha)
            
        if tMaterial.specularColor and tMaterial.specularHardness:
            specular = tMaterial.specularColor * tMaterial.specularIntensity
            power = tMaterial.specularHardness
            uMaterial.specularColor = (specular.r, specular.g, specular.b, power)

        if isEmissive and tMaterial.emitColor and tMaterial.emitIntensity:
            emissive = tMaterial.emitColor * tMaterial.emitIntensity
            uMaterial.emissiveColor = (emissive.r, emissive.g, emissive.b)

        uMaterial.twoSided = tMaterial.twoSided
        uMaterial.shadeless = tMaterial.shadeless

        uMaterial.texturesNames = tMaterial.texturesNames.copy()

        # Remove emissive/ao/lightmap and add one of them as emissive
        uMaterial.texturesNames.pop("emissive", None)
        uMaterial.texturesNames.pop("ao", None)
        uMaterial.texturesNames.pop("lightmap", None)
        if emissiveKey:
            uMaterial.texturesNames["emissive"] = tMaterial.texturesNames[emissiveKey]

        """
        uMaterials.append(uMaterial)
       

 
    
    
    
    