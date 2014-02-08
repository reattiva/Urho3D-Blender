
#
# This script is licensed as public domain.
# Based on "Export Inter-Quake Model (.iqm/.iqe)" by Lee Salzman
#

#  http://www.blender.org/documentation/blender_python_api_2_63_2/info_best_practice.html
#  http://www.blender.org/documentation/blender_python_api_2_63_2/info_gotcha.html
# Blender types:
#  http://www.blender.org/documentation/blender_python_api_2_63_7/bpy.types.Mesh.html
#  http://www.blender.org/documentation/blender_python_api_2_63_7/bpy.types.MeshTessFace.html
#  http://www.blender.org/documentation/blender_python_api_2_63_7/bpy.types.Material.html
# UV:
#  http://www.blender.org/documentation/blender_python_api_2_63_2/bpy.types.MeshTextureFaceLayer.html
#  http://www.blender.org/documentation/blender_python_api_2_63_2/bpy.types.MeshTextureFace.html
# Skeleton:
#  http://www.blender.org/documentation/blender_python_api_2_66_4/bpy.types.Armature.html
#  http://www.blender.org/documentation/blender_python_api_2_66_4/bpy.types.Bone.html
#  http://www.blender.org/documentation/blender_python_api_2_66_4/bpy.types.Pose.html
#  http://www.blender.org/documentation/blender_python_api_2_66_4/bpy.types.PoseBone.html
# Animations:
#  http://www.blender.org/documentation/blender_python_api_2_66_4/bpy.types.Action.html
#  http://www.blender.org/documentation/blender_python_api_2_66_4/bpy.types.AnimData.html
# Vertex color:
#  http://www.blender.org/documentation/blender_python_api_2_66_4/bpy.types.MeshColor.html
# Moprhs (Shape keys):
#  http://www.blender.org/documentation/blender_python_api_2_66_4/bpy.types.Key.html
#  http://www.blender.org/documentation/blender_python_api_2_66_4/bpy.types.ShapeKey.html

# Inverse transpose for normals
#  http://www.arcsynthesis.org/gltut/Illumination/Tut09%20Normal%20Transformation.html

# Pthon binary writing:
#  http://docs.python.org/2/library/struct.html

DEBUG = 0

import bpy
import bmesh
import math
import time
from mathutils import Vector, Matrix, Quaternion
from collections import OrderedDict
import os
import operator
import heapq
import logging

log = logging.getLogger("ExportLogger")

#------------------
# Geometry classes
#------------------

# Vertex class
class TVertex:
    def __init__(self):
        # Index of the vertex in the Blender buffer
        self.blenderIndex = None
        # Position of the vertex: Vector((0.0, 0.0, 0.0))
        self.pos = None
        # Normal of the vertex: Vector((0.0, 0.0, 0.0))
        self.normal = None
        # Color of the vertex: (0, 0, 0, 0)...(255, 255, 255, 255)
        self.color = None
        # UV coordinates of the vertex: Vector((0.0, 0.0))..Vector((1.0, 1.0))
        self.uv = None
        # UV2 coordinates of the vertex: Vector((0.0, 0.0))..Vector((1.0, 1.0))
        self.uv2 = None
        # Tangent of the vertex: Vector((0.0, 0.0, 0.0, 0.0))
        self.tangent = None
        # Bitangent of the vertex: Vector((0.0, 0.0, 0.0))
        self.bitangent = None
        # Bones weights: list of tuple(boneIndex, weight)
        self.weights = None

    # used by the function index() of lists
    def __eq__(self, other):
        # TODO: can we do without color and weights?
        #return (self.__dict__ == other.__dict__)
        return (self.pos == other.pos and 
                self.normal == other.normal and 
                self.uv == other.uv)
    
    def __hash__(self):
        hashValue = 0
        if self.pos:
            hashValue ^= hash(self.pos.x) ^ hash(self.pos.y) ^ hash(self.pos.z)
        if self.normal:
            hashValue ^= hash(self.normal.x) ^ hash(self.normal.y) ^ hash(self.normal.z)
        if self.uv:
            hashValue ^= hash(self.uv.x) ^ hash(self.uv.y)
        return hashValue
    
    def __str__(self):
        s  = "  coords: {: .3f} {: .3f} {: .3f}".format(self.pos.x, self.pos.y, self.pos.z)
        s += "\n normals: {: .3f} {: .3f} {: .3f}".format(self.normal.x, self.normal.y, self.normal.z)
        if self.color:
            s += "\n   color: {:3d} {:3d} {:3d} {:3d}".format(self.color[0], self.color[1], self.color[2], self.color[3])
        if self.uv:
            s += "\n      uv: {: .3f} {: .3f}".format(self.uv[0], self.uv[1])
        if self.uv2:
            s += "\n     uv2: {: .3f} {: .3f}".format(self.uv2[0], self.uv2[1])
        if self.tangent:
            s += "\n tangent: {: .3f} {: .3f} {: .3f}".format(self.tangent.x, self.tangent.y, self.tangent.z)
        if self.weights:
            s += "\n weights: "
            for w in self.weights:
                s += "{:d} {:.3f}  ".format(w[0],w[1])
        return s

# Geometry LOD level class
class TLodLevel:
    def __init__(self):
        self.distance = 0.0
        # Set of all vertex indices use by this LOD
        self.indexSet = set()
        # List of triangles of the LOD (triples of vertex indices)
        self.triangleList = []

    def __str__(self):  
        s = "  distance: {:.3f}\n".format(self.distance)
        s += "  triangles: "
        for i, t in enumerate(self.triangleList):
            if i and (i % 5) == 0:
                s += "\n             "
            s += "{:3d} {:3d} {:3d} |".format(t[0],t[1],t[2])
        return s
    
# Geometry class
class TGeometry:
    def __init__(self):
        # List of TLodLevel
        self.lodLevels = []

    def __str__(self):
        s = ""
        for i, l in enumerate(self.lodLevels):
            s += " {:d}\n".format(i) + str(l)
        return s

#------------------
# Morph classes
#------------------

class TMorph:
    def __init__(self, name):
        # Morph name
        self.name = name
        # Set of all vertex indices use by this morph
        self.indexSet = set()
        # List of triangles of the morph (triples of vertex indices)
        self.triangleList = []
        # Maps vertex index to morphed TVertex
        self.vertexMap = {}

    def __str__(self):  
        s = " name: {:s}\n".format(self.name)
        s += " Vertices: "
        for k, v in sorted(self.vertices.items()):
            s += "\n  index: {:d}".format(k)
            s += "\n" + str(v)
        return s

#-------------------
# Materials classes
#-------------------

# NOTE: in Blender images names are unique

class TMaterial:
    def __init__(self, name):
        # Material name
        self.name = name
        # Diffuse color (0.0, 0.0, 0.0)
        self.diffuseColor = None
        # Diffuse intesity (0.0)
        self.diffuseIntensity = None
        # Specular color (0.0, 0.0, 0.0)
        self.specularColor = None
        # Specular intesity (0.0)
        self.specularIntensity = None
        # Specular hardness (1.0)
        self.specularHardness = None
        # Opacity (1.0) 
        self.opacity = None
        # Material is two sided
        self.twoSided = False
        # Diffuse color texture filename (no path)
        self.diffuseTexName = None
        # Normal texture filename (no path)
        self.normalTexName = None
        # Specular texture filename (no path)
        self.specularTexName = None
        # Emissive texture filename (no path)
        self.lightmapTexName = None

    def __eq__(self, other):
        if hasattr(other, 'name'):
            return (self.name == other.name)
        return (self.name == other)

    def __str__(self):  
        return (" name: {:s}\n"
                " image: \"{:s}\""
                .format(self.name, self.diffuseTexName) )

#--------------------
# Animations classes
#--------------------

class TBone:    
    def __init__(self, index, parentName, position, rotation, scale, transform):
        # Position of the bone in the OrderedDict
        self.index = index
        # Name of the parent bone
        self.parentName = parentName
        # Bone position in the parent bone tail space (you first apply this)
        self.bindPosition = position
        # Bone rotation in the parent bone tail space (and then this)
        self.bindRotation = rotation
        # Bone scale
        self.bindScale = scale
        # Bone transformation in object space
        self.worldTransform = transform

    def __str__(self):
        s = " bind pos " + str(self.bindPosition)
        s += "\n bind rot " + str(self.bindRotation) #+ "\n" + str(self.bindRotation.to_axis_angle())
        #s += "\n" + str(self.worldTransform.inverted())
        s += "\n" + str(self.worldTransform)
        return s

class TFrame:
    def __init__(self, time, position, rotation, scale):
        self.time = time
        self.position = position
        self.rotation = rotation
        self.scale = scale
        
    def hasMoved(self, other):
        return (self.position != other.position or self.rotation != other.rotation or self.scale != other.scale)

class TTrack:
    def __init__(self, name):
        self.name = name
        self.frames = []

class TAnimation:
    def __init__(self, name):
        self.name = name
        self.tracks = []

#---------------------
# Export data classes
#---------------------

class TData:
    def __init__(self):
        self.objectName = None
        # List of all the TVertex of all the geometries
        self.verticesList = []
        # List of TGeometry, they contains triangles, triangles are made of vertex indices
        self.geometriesList = []
        # List of TMorph: a subset of the vertices list with modified position
        self.morphsList = []
        # List of TMaterial
        self.materialsList = []
        # Material name to geometry index map
        self.materialGeometryMap = {}
        # Ordered dictionary of TBone: bone name to TBone
        self.bonesMap = OrderedDict()
        # List of TAnimation
        self.animationsList = []
        # Dictionary container for errors
        self.errorsDict = {}

class TOptions:
    def __init__(self):
        self.newLod = True
        self.lodDistance = 0.0
        self.doForceElements = False
        self.mergeObjects = False
        self.useLods = False
        self.onlySelected = False
        self.scale = 1.0
        self.globalOrigin = True
        self.bonesGlobalOrigin = False  #useless
        self.actionsGlobalOrigin = False
        self.applyModifiers = False
        self.applySettings = 'PREVIEW'
        self.doBones = True
        self.doOnlyKeyedBones = False   #TODO: check
        self.derigifyArmature = False
        self.doAnimations = True
        self.doAllActions = True
        self.doUsedActions = False
        self.doSelectedStrips = False
        self.doSelectedTracks = False
        self.doStrips = False
        self.doTracks = False
        self.doTimeline = False
        self.doAnimationPos = True
        self.doAnimationRot = True
        self.doAnimationSca = True
        self.doGeometries = True
        self.doGeometryPos = True
        self.doGeometryNor = True
        self.doGeometryCol = True
        self.doGeometryColAlpha = False
        self.doGeometryUV  = True
        self.doGeometryUV2 = False
        self.doGeometryTan = True
        self.doGeometryWei = True
        self.doMorphs = True
        self.doMorphNor = True
        self.doMorphTan = True
        self.doMorphUV = True
        self.doOptimizeIndices = True
        self.doMaterials = True
        

#--------------------
# “Computing Tangent Space Basis Vectors for an Arbitrary Mesh” by Lengyel, Eric. 
# Terathon Software 3D Graphics Library, 2001.
# http://www.terathon.com/code/tangent.html
#--------------------
        
def GenerateTangents(tLodLevel, tVertexList, invalidUvIndices):

    if not tLodLevel.indexSet or not tLodLevel.triangleList or not tVertexList:
        log.warning("Missing data, tangent generation cancelled.")
        return

    tangentOverwritten = False
    minVertexIndex = None
    maxVertexIndex = None
    for vertexIndex in tLodLevel.indexSet:
        if minVertexIndex is None:
            minVertexIndex = vertexIndex
            maxVertexIndex = vertexIndex
        elif minVertexIndex > vertexIndex:
            minVertexIndex = vertexIndex
        elif maxVertexIndex < vertexIndex:
            maxVertexIndex = vertexIndex

        vertex = tVertexList[vertexIndex]
        
        # Check if we have already calculated tangents for this vertex and we're overwriting them
        if vertex.tangent:
            print("x")
            tangentOverwritten = True
            
        # Check if we have all the needed data to do the calculations
        if vertex.pos is None:
            invalidUvIndices.add(vertex.blenderIndex)
            log.warning("Missing position on vertex {:d}, tangent generation cancelled.".format(vertex.blenderIndex))
            return
        if vertex.normal is None:
            invalidUvIndices.add(vertex.blenderIndex)
            log.warning("Missing normal on vertex {:d}, tangent generation cancelled.".format(vertex.blenderIndex))
            return
        if vertex.uv is None:
            invalidUvIndices.add(vertex.blenderIndex)
            log.warning("Missing UV on vertex {:d}, tangent generation cancelled.".format(vertex.blenderIndex))
            return
        
        # Init tangent and bitangent vectors
        vertex.tangent = Vector((0.0, 0.0, 0.0))
        vertex.bitangent = Vector((0.0, 0.0, 0.0))

    if tangentOverwritten:
        log.warning("Overwriting tangent and bitangent")

    invalidUV = False
    for i, triangle in enumerate(tLodLevel.triangleList):
        # For each triangle, we have 3 vertices vertex1, vertex2, vertex3, each of the have their UV coordinates, we want to 
        # find two unit orthogonal vextors (tangent and bitangent) such as we can express each vertices position as a function
        # of the vertices UV: 
        #  VertexPosition = Tangent * f'(VertexUV) + BiTangent * f"(VertexUV)
        # Actually we are going to express them relatively to a vertex choosen as origin (vertex1):
        #  vertex - vertex1 = Tangent * (vertex.u - vertex1.u) + BiTangent * (vertex.v - vertex1.v)
        # We have two equations, one for vertex2-vertex1 and one for vertex3-vertex1, if we put them in a system and solve it
        # we can obtain Tangent and BiTangent:
        #  [T; B] = [u1, v1; u2, v2]^-1 * [V2-V1; V3-V1]
        
        vertex1 = tVertexList[triangle[0]]
        vertex2 = tVertexList[triangle[1]]
        vertex3 = tVertexList[triangle[2]]

        # First equation: [x1, y1, z1] = Tangent * u1 + BiTangent * v1
        x1 = vertex2.pos.x - vertex1.pos.x
        y1 = vertex2.pos.y - vertex1.pos.y
        z1 = vertex2.pos.z - vertex1.pos.z

        u1 = vertex2.uv.x - vertex1.uv.x
        v1 = vertex2.uv.y - vertex1.uv.y

        # Second equation: [x2, y2, z2] = Tangent * u2 + BiTangent * v2
        x2 = vertex3.pos.x - vertex1.pos.x
        y2 = vertex3.pos.y - vertex1.pos.y
        z2 = vertex3.pos.z - vertex1.pos.z

        u2 = vertex3.uv.x - vertex1.uv.x
        v2 = vertex3.uv.y - vertex1.uv.y

        # Determinant of the matrix [u1 v1; u2 v2]
        d = u1 * v2 - u2 * v1
        
        # If the determinant is zero then the points (0,0), (u1,v1), (u2,v2) are in line, this means
        # the area on the UV map of this triangle is null. This is an error, we must skip this triangle.
        if d == 0:
            invalidUvIndices.add(vertex1.blenderIndex)
            invalidUvIndices.add(vertex2.blenderIndex)
            invalidUvIndices.add(vertex3.blenderIndex)
            invalidUV = True
            continue

        t = Vector( ((v2 * x1 - v1 * x2) / d, (v2 * y1 - v1 * y2) / d, (v2 * z1 - v1 * z2) / d) )
        b = Vector( ((u1 * x2 - u2 * x1) / d, (u1 * y2 - u2 * y1) / d, (u1 * z2 - u2 * z1) / d) )
        
        vertex1.tangent += t;
        vertex2.tangent += t;
        vertex3.tangent += t;
        
        vertex1.bitangent += b;
        vertex2.bitangent += b;
        vertex3.bitangent += b;

    if invalidUV:
        log.error("Invalid UV, the area in the UV map is too small.")

    for vertexIndex in tLodLevel.indexSet:
        vertex = tVertexList[vertexIndex]

        # Gram-Schmidt orthogonalize
        
        # Unit vector perpendicular to normal and in the same plane of normal and tangent
        tOrtho = ( vertex.tangent - vertex.normal * vertex.normal.dot(vertex.tangent) ).normalized()
        # Unit vector perpendicular to the plane of normal and tangent
        bOrtho = vertex.normal.cross(vertex.tangent).normalized()

        # Calculate handedness: if bOrtho and bitangent have the different directions, save the verse
        # in tangent.w, so we can reconstruct bitangent by: tangent.w * normal.cross(tangent)
        w = 1.0 if bOrtho.dot(vertex.bitangent) >= 0.0 else -1.0
        
        vertex.bitangent = bOrtho
        vertex.tangent = Vector((tOrtho.x, tOrtho.y, tOrtho.z, w))


        
#--------------------
# Linear-Speed Vertex Cache Optimisation algorithm by Tom Forsyth
#  https://home.comcast.net/~tom_forsyth/papers/fast_vert_cache_opt.html
#--------------------

# This is an optimized version, but it is still slow.
# (on an average pc, 5 minutes for 30K smooth vertices)

#  We try to sort triangles in the index buffer so that we gain an optimal use
#  of the hardware vertices cache.
#  We assign a score to each triangle, we find the best and save it in a new 
#  ordered list.
#  The score of each triangle is the sum of the score of its vertices, and the
#  score of a vertex is higher if it is:
#  - used recently (it is still in the cache) but we also try to avoid the last
#    triangle added (n this way we get better result),
#  - lonely isolated vertices (otherwise the will be keep for last and drawing
#    them will require an higher cost)
#  The order of vertices in the triangle does not matter.
#  We'll apply this optimization to each lod of each geometry.

# These are the constants used in the algorithm:
VERTEX_CACHE_SIZE = 32
CACHE_DECAY_POWER = 1.5
LAST_TRI_SCORE = 0.75
VALENCE_BOOST_SCALE = 2.0
VALENCE_BOOST_POWER = 0.5

def CalculateScore(rank):

    if rank.useCount == 0:
        rank.score = -1.0
        return

    score = 0.0
    cachePosition = rank.cachePosition
    
    if cachePosition < 0:
        # Vertex is not in FIFO cache - no score
        pass
    elif cachePosition < 3:
        # This vertex was used in the last triangle,
        # so it has a fixed score, whichever of the three
        # it's in. Otherwise, you can get very different
        # answers depending on whether you add
        # the triangle 1,2,3 or 3,1,2 - which is silly.
        score = LAST_TRI_SCORE
    else:
        # Points for being high in the cache
        score = 1.0 - float(rank.cachePosition - 3) / (VERTEX_CACHE_SIZE - 3)
        score = pow(score, CACHE_DECAY_POWER)

    # Bonus points for having a low number of tris still to
    # use the vert, so we get rid of lone verts quickly
    valenceBoost = VALENCE_BOOST_SCALE * pow(rank.useCount, -VALENCE_BOOST_POWER);
    rank.score = score + valenceBoost;

# Triangles score list sizes
TRIANGLERANK_SIZE = 500
TRIANGLERANK_MAX_SIZE = 505

def OptimizeIndices(lodLevel):
    
    # Ranks are used to store data for each vertex
    class Rank:
        def __init__(self):
            self.score = 0.0
            self.useCount = 1
            self.cachePosition = -1
    
    # Create a map: vertex index to its corresponding Rank
    ranking = {}
    
    # This list contains the original triangles (not in optimal order), we'll move them 
    # one by one in a new list following the optimal order
    oldTriangles = lodLevel.triangleList

    # For each vertex index of each triangle increment the use counter
    # (we can find the same vertex index more than once)
    for triangle in oldTriangles:
        for index in triangle:
            try:
                ranking[index].useCount += 1
            except KeyError:
                ranking[index] = Rank()

    # Calculate the first round of scores
    # (Rank is mutable, so CalculateScore will be able to modify it)
    for rank in ranking.values():        
        CalculateScore(rank)

    # Ths list will contain the triangles sorted in optimal order
    newTriangles = []

    # Cache of vertex indices
    vertexCache = []
    
    # The original algorithm was:
    # - scan all the old triangles and find the one with the best score;
    # - move it to the new triangles;
    # - move its vertices in the cache;
    # - recalculate the score on all the vertices on the cache.
    # The slowest part is the first step, scanning all the old triangles,
    # but in the last step we update only a little subset of these triangles,
    # and it is a waste to recalculate the triangle score of each old triamgle.
    # So we do this:
    # - create a map 'trianglesMap': vertex index to triangles;
    # - keep a list 'trianglesRanking' of the best triangles;
    # - at first this list is empty, we start adding triangles; we add tuples like
    #   (score, triangle) and we keep track of the min score, we don't add triangles
    #   with score lower than the min; for now we add triangles without bothering
    #   about order; if the triangle is already present in the list we only update
    #   its score (even if it is lower);
    # - when the list is a little too big (TRIANGLERANK_MAX_SIZE), we sort the list 
    #   by score and we only keep the best TRIANGLERANK_SIZE triangles, we update 
    #   the min score;
    # - after scanning all the old triangles, we take out from the list the best
    #   triangle;
    # - move it to the new triangles and remove it from the map;
    # - move its vertices in the cache;
    # - recalculate the score on all the vertices in the cache, if the score of one
    #   vertex is changed, we use the map to find what triangles are affected and
    #   we add them to the list (unordered and only if their score is > min);
    # - now when we repeat we have the list already populated, so we don't need to
    #   recalculate all old triangles scores, we only need to sort the list and take
    #   out the best triangle.

        
    # Vertex index to triangle indices list map
    trianglesMap = {}
    # Populate the map
    for triangle in oldTriangles:
        for vertexIndex in triangle:
            try:
                triangleList = trianglesMap[vertexIndex]
            except KeyError:
                triangleList = []
                trianglesMap[vertexIndex] = triangleList
            triangleList.append(triangle)

    class TrianglesRanking:
        def __init__(self):
            self.ranklist = []
            self.min = None
            self.isSorted = True
    
        def update(self, triangle):            
            # Sum the score of all its vertex. 
            # >> This is the slowest part of the algorithm <<
            triangleScore = ranking[triangle[0]].score + ranking[triangle[1]].score + ranking[triangle[2]].score
            # If needed, add it to the list
            if not self.ranklist:
                self.ranklist.append( (triangleScore, triangle) )
                self.min = triangleScore
            else:
                # We add only triangles with score > min
                if triangleScore > self.min:
                    found = False
                    # Search of the triangle is already present in the list
                    for i, rank in enumerate(self.ranklist):
                        if triangle == rank[1]:
                            if triangleScore != rank[0]:
                                self.ranklist[i] = (triangleScore, triangle)
                                self.isSorted = False
                            found = True
                            break
                    # It is a new triangle
                    if not found:
                        self.ranklist.append( (triangleScore, triangle) )
                        self.isSorted = False

        def sort(self):
            if self.isSorted:
                return
            #self.ranklist = sorted(self.ranklist, key=operator.itemgetter(0), reverse=True)[:TRIANGLERANK_SIZE]
            self.ranklist = heapq.nlargest(TRIANGLERANK_SIZE, self.ranklist, key = operator.itemgetter(0))
            self.min = self.ranklist[-1][0]
            self.isSorted = True
        
        def popBest(self):
            bestTriangle = self.ranklist[0][1]
            del self.ranklist[0]
            return bestTriangle

    trianglesRanking = TrianglesRanking()

    # Progress counter
    progressCur = 0
    progressTot = 0.01 * len(oldTriangles)

    if DEBUG: ttt = time.time() #!TIME

    # While there still are unsorted triangles
    while oldTriangles:
        # Print progress
        if (progressCur & 0x7F) == 0:
            print("{:.3f}%\r".format(progressCur / progressTot), end='' )
        progressCur += 1
        
        # When the list is empty, we need to scan all the old triangles
        if not trianglesRanking.ranklist:
            for triangle in oldTriangles:
                # We add the triangle but we don't search for the best one
                trianglesRanking.update(triangle)
                # If the list is too big, sort and truncate it
                if len(trianglesRanking.ranklist) > TRIANGLERANK_MAX_SIZE:
                    trianglesRanking.sort()

        if trianglesRanking:
            # Only if needed, we sort and truncate
            trianglesRanking.sort()
            # We take the best triangles out of the list
            bestTriangle = trianglesRanking.popBest()
        else:
            log.error("Could not find next triangle")
            return        
        
        # Move the best triangle to the output list
        oldTriangles.remove(bestTriangle)
        newTriangles.append(bestTriangle)
            
        # Model the LRU cache behaviour
        # Recreate the cache removing the vertices of the best triangle
        vertexCache = [i for i in vertexCache if i not in bestTriangle]
        
        for vertexIndex in bestTriangle:
            # Then push them to the front
            vertexCache.insert(0, vertexIndex)
            # Decrement the use counter of its vertices
            ranking[vertexIndex].useCount -= 1
            # Remove best triangle from the map
            triangleList = trianglesMap[vertexIndex]
            triangleList.remove(bestTriangle)

        # Update positions & scores of all vertices in the cache
        # Give position -1 if vertex is going to be erased
        for i, vertexIndex in enumerate(vertexCache):
            rank = ranking[vertexIndex]
            if (i > VERTEX_CACHE_SIZE):
                rank.cachePosition = -1
            else:
                rank.cachePosition = i
            # Calculate the new score
            oldScore = rank.score
            CalculateScore(rank)
            # If the score is changed
            if oldScore != rank.score:
                # Add to the list all the triangles affected
                triangleList = trianglesMap[vertexIndex]
                for triangle in triangleList:   
                    trianglesRanking.update(triangle)
                
        # Finally erase the extra vertices
        vertexCache[:] = vertexCache[:VERTEX_CACHE_SIZE]

    if DEBUG: print("[TIME2] {:.4f}".format(time.time() - ttt) ) #!TIME

    # Rewrite the index data now
    lodLevel.triangleList = newTriangles


#--------------------
# Decompose armatures
#--------------------

def DerigifyArmature(armature):

    # Map {ORG bone name: Blender ORG bone} 
    orgbones = {}
    # Map {DEF bone name: Blender DEF bone} 
    defbones = {}
    # Map {ORG bone name: list of DEF bones names}
    org2defs = {}
    # Map {DEF bone name: ORG bone name}
    def2org = {}
    # Map {DEF bone name: list of children DEF bones}
    defchildren = {}
    # Map {DEF bone name: its parent DEF bone name}    
    defparent = {}
        
    # Scan the armature and collect ORG bones and DEF bones in the maps by their names,
    # we remove ORG- or DEF- from names
    for bone in armature.bones.values():
        if bone.name.startswith('ORG-'):
            orgbones[bone.name[4:]] = bone
            org2defs[bone.name[4:]] = []
        elif bone.name.startswith('DEF-'):
            defbones[bone.name[4:]] = bone
            defchildren[bone.name[4:]] = []

    # For each DEF bone in the map get its name and Blender bone
    for name, bone in defbones.items():
        orgname = name
        # Search if exist an ORG bone with the same name of this DEF bone (None if not found)
        orgbone = orgbones.get(orgname)
        # If this ORG bone does not exist, then the DEF bone name could be DEF-<name>.<number>,
        # so we remove .<number> and search for the ORG bone again
        if not orgbone:
            splitname = name.rfind('.')
            if splitname >= 0 and name[splitname+1:].isdigit():
                orgname = name[:splitname]
                orgbone = orgbones.get(orgname)
        # Map the ORG name (can be None) to the DEF name (one to many)
        org2defs[orgname].append(name)
        # Map the DEF name to the ORG name (can be None) (one to one)
        def2org[name] = orgname

    # Sort DEF bones names in the ORG to DEF map, so we get: <name>.0, <name>.1, <name>.2 ...
    for defs in org2defs.values():
        defs.sort()

    # For each DEF bone in the map get its name and Blender bone
    for name, bone in defbones.items():
        # Get the relative ORG bone name (can be None)
        orgname = def2org[name]
        # Get the ORG bone
        orgbone = orgbones.get(orgname)
        # Get the list (sorted by number) of DEF bones associated to the ORG bone
        defs = org2defs[orgname]
        if orgbone:
            # Get the index of the DEF bone in the list
            i = defs.index(name)
            # If it is the first (it has the lowest number, e.g. <name>.0)
            if i == 0:
                orgparent = orgbone.parent
                # If the ORG parent bone exists and it is an ORG bone
                if orgparent and orgparent.name.startswith('ORG-'):
                    orgpname = orgparent.name[4:]
                    # Map this DEF bone to the last DEF bone of the ORG parent bone
                    defparent[name] = org2defs[orgpname][-1]
            else:
                # Map this DEF bone to the previous DEF bone in the list (it has a lower number)
                defparent[name] = defs[i-1]
        # If this DEF bone has a parent, append it as a children of the parent
        if name in defparent:
            defchildren[defparent[name]].append(name)

    boneList = []
    
    # Recursively add children
    def Traverse(boneName):
        # Get the Blender bone
        bone = defbones[boneName]
        parent = None
        # If it has a parent, get its Blender bone
        if boneName in defparent:
            parentName = defparent[boneName]
            parent = defbones[parentName]
        bonesList.append( (bone, parent) )
        # Proceed with children bones
        for childName in defchildren[boneName]:
            Traverse(childName)            
    
    # Start from bones with no parent (root bones)
    for boneName in defbones:
        if boneName not in defparent: 
            Traverse(boneName)    
                
    return boneList

# How to read a skeleton: 
# start from the root bone, move it of bindPosition in the armature space
# then rotate the armature space with bindRotation, this will be the parent
# space used by its childs. For each child bone move it of bindPosition in 
# the parent space then rotate the parent space with bindRotation, and so on.

# We need each bone position and rotation in parent bone space:
# upAxis = Matrix.Rotation(pi/2, 4, 'X')
# poseMatrix = bone.matrix
# if parent:
#   poseMatrix = parentBone.matrix.inverted() * poseMatrix
# else:
#   poseMatrix = upAxis.matrix.inverted() * origin.matrix * poseMatrix  

def DecomposeArmature(scene, armatureObj, tData, tOptions):
    
    bonesMap = tData.bonesMap

    ## armature.pose_position = 'REST'
    ## bpy.data.armatures[0].pose_position = 'REST'

    # 'armature.pose.bones' contains bones data for the current frame
    # 'armature.data.bones' contains bones data for the rest position
    armature = armatureObj.data
    
    if not armature.bones:
        log.warning('Armature {:s} has no bones'.format(armatureObj.name))
        return

    log.info("Decomposing armature: {:s} ({:d} bones)".format(armatureObj.name, len(armature.bones)) )

    originMatrix = Matrix.Identity(4)    
    if tOptions.bonesGlobalOrigin:
        originMatrix = armatureObj.matrix_world

    # Get a list of bones
    if tOptions.derigifyArmature:
        # from a Rigify armature
        bonesList = DerigifyArmature(armature)
        print("Derifigy")
    else:
        # from a standard armature
        bonesList = []
        # Recursively add children
        def Traverse(bone, parent):
            bonesList.append( (bone, parent) )
            for child in bone.children:
                Traverse(child, bone)
        # Start from bones with no parent (root bones)
        for bone in armature.bones.values():
            if bone.parent is None: 
                Traverse(bone, None)    

    if not bonesList:
        log.warning('Armature {:s} has no bone to export'.format(armatureObj.name))
        return

    for bone, parent in bonesList:
    
        # 'bone.matrix_local' is referred to the armature, we need
        # the trasformation between the current bone and its parent.
        boneMatrix = bone.matrix_local.copy()
        
        # Here 'bone.matrix_local' is in object(armature) space, so we have to
        # calculate the bone trasformation in parent bone space
        if parent:
            boneMatrix = parent.matrix_local.inverted() * boneMatrix
        else:
            # Normally we don't have to worry that Blender is Z up and we want
            # Y up because we use relative trasformations between bones. However
            # the parent bone is relative to the armature so we need to convert
            # Z up to Y up by rotating its matrix by -90° on X
            boneMatrix = Matrix.Rotation(math.radians(-90.0), 4, 'X' ) * originMatrix * boneMatrix

        if tOptions.scale != 1.0:
            boneMatrix.translation *= tOptions.scale

        # Extract position and rotation relative to parent in parent space        
        t = boneMatrix.to_translation()
        q = boneMatrix.to_quaternion()
        s = boneMatrix.to_scale()
                
        # Convert position and rotation to left hand:
        tl = Vector((t.x, t.y, -t.z))
        ql = Quaternion((q.w, -q.x, -q.y, q.z))
        sl = Vector((s.x, s.y, s.z))
        
        # Now we need the bone matrix relative to the armature. 'matrix_local' is
        # what we are looking for, but it needs to be converted:
        # 1) rotate of -90° on X axis:
        # - swap column 1 with column 2
        # - negate column 1
        # 2) convert bone trasformation in object space to left hand:        
        # - swap row 1 with row 2
        # - swap column 1 with column 2
        # So putting them together:
        # - swap row 1 with row 2
        # - negate column 2
        ml = bone.matrix_local.copy()
        if tOptions.scale != 1.0:
            ml.translation *= tOptions.scale
        (ml[1][:], ml[2][:]) = (ml[2][:], ml[1][:])
        ml[0][2] = -ml[0][2]
        ml[1][2] = -ml[1][2]
        ml[2][2] = -ml[2][2]

        # Create a new bone
        parentName = parent and parent.name
        tBone = TBone(len(bonesMap), parentName, tl, ql, sl, ml)

        # If new, add the bone to the map with its name
        if bone.name not in bonesMap:
            bonesMap[bone.name] = tBone
        else:
            log.critical("Bone {:s} already present in the map.".format(bone.name))
        

#--------------------
# Decompose animations
#--------------------

def DecomposeActions(scene, armatureObj, tData, tOptions):

    bonesMap = tData.bonesMap
    animationsList = tData.animationsList
    
    if not armatureObj.animation_data:
        log.warning('Armature {:s} has no animation data'.format(armatureObj.name))
        return

    # armatureObj.animation_data.action ???
            
    originMatrix = Matrix.Identity(4)
    if tOptions.actionsGlobalOrigin:
        originMatrix = armatureObj.matrix_world
        if tOptions.globalOrigin and originMatrix != Matrix.Identity(4):
            # Blender moves/rotates the armature together with the mesh, so if you set a global origin
            # for Mesh and Actions you'll have twice the transformations. Set only one global origin.
            log.warning("Use local origin for the object otherwise trasformations are applied twice")
    
    # Save current action and frame, we'll restore them later
    savedAction = armatureObj.animation_data.action
    savedFrame = scene.frame_current
    savedUseNla = armatureObj.animation_data.use_nla
    
    # Here we collect every animation objects we want to export
    animationObjects = []

    # Scan all the Tracks not muted of the armature
    for track in armatureObj.animation_data.nla_tracks:
        track.is_solo = False
        if track.mute:
            continue
        # Add Track
        if tOptions.doTracks or (self.doSelectedTracks and track.select):
            animationObjects.append(track)
        # Scan all the Strips of the Track
        for strip in track.strips:
            # Add Strip (every Strip is unique, no need to check for duplicates)
            if tOptions.doStrips or (self.doSelectedStrips and strip.select):
                animationObjects.append(strip)
            # Add an used Action 
            action = strip.action
            if tOptions.doUsedActions and action and not action in animationObjects:
                animationObjects.append(action)

    # armatureObj.animation_data.action
    # armatureObj.animation_data.nla_tracks.active
                
    # Add all the Actions (even if unused or deleted)
    if tOptions.doAllActions:
        animationObjects.extend(bpy.data.actions)

    # Add Timeline (as the armature object)
    if tOptions.doTimeline:
        animationObjects.append(armatureObj)

    if not animationObjects:
        log.warning('Armature {:s} has no animation to export'.format(armatureObj.name))
        return
    
    for object in animationObjects:
        tAnimation = TAnimation(object.name)
    
        if type(object) is bpy.types.Action:
            # Actions have their frame range
            (startframe, endframe) = object.frame_range
            startframe = int(startframe)
            endframe = int(endframe+1)
        elif type(object) is bpy.types.NlaStrip:
            # Strips also have their frame range
            startframe = int(object.frame_start)
            endframe = int(object.frame_end+1)
        else:
            # For Tracks and Timeline we use the scene playback range
            startframe = int(scene.frame_start)
            endframe = int(scene.frame_end+1)

        # Here we collect every action used by this animation, so we can filter the only used bones
        actionSet = set()

        # Clear current action on the armature
        try:
            armatureObj.animation_data.action = None
        except AttributeError:
            log.error("You need to exit action edit mode")
            return
        
        # If it is an Action, set the current Action; also disable NLA to disable influences from others NLA tracks
        if type(object) is bpy.types.Action:
            log.info("Decomposing action: {:s} (frames {:.1f} {:.1f})".format(object.name, startframe, endframe))
            # Set Action on the armature
            armatureObj.animation_data.use_nla = False
            armatureObj.animation_data.action = object
            # Get the Actions
            actionSet.add(object)
            
        # If it is a Track (not muted) or a Strip in a Track, set it as solo
        if type(object) is bpy.types.NlaTrack or type(object) is bpy.types.NlaStrip:
            log.info("Decomposing track: {:s} (frames {:.1f} {:.1f})".format(object.name, startframe, endframe))
            # Set the NLA Track as solo
            object.is_solo = True
            armatureObj.animation_data.use_nla = True
            # Get the Actions
            for strip in object.strips:
                if strip.action:
                    actionSet.add(strip.action)
            
        # If it is the Timeline, merge all the Tracks (not muted)
        if type(object) is bpy.types.Object:        
            log.info("Decomposing animation: {:s} (frames {:.1f} {:.1f})".format(object.name, startframe, endframe))
            armatureObj.animation_data.use_nla = True
            # Get the Actions
            for track in object.animation_data.nla_tracks:
                for strip in track.strips:
                    if strip.action:
                        actionSet.add(strip.action)

        if not animationObjects:
            log.warning("No actions for animation {:s}".format(object.name))

        # Get the bones names
        bones = []
        if tOptions.doOnlyKeyedBones:
            # Get all the names of the bones used by the actions
            boneSet = set()
            for action in actionSet:
                for group in action.groups:
                    bonesSet.update(group.name)
            # Add the bones name respecting the order of bonesMap
            for bone in bonesMap.keys():
                if bone in boneSet:
                    bones.append(bone)
                    boneSet.remove(bone)
            # Check if any bones used by actions is missing in the map
            for bone in boneSet:
                log.warning("Action group(bone) {:s} is not in the skeleton".format(bone))
        else:
            # Get all the names of the bones in the map
            bones = bonesMap.keys()
	
        if not bones:
            log.warning("No bones for animation {:s}".format(object.name))
            continue
        
        # Reset position/rotation/scale of each bone
        for poseBone in armatureObj.pose.bones:
            poseBone.matrix_basis = Matrix.Identity(4)

        # Progress counter
        progressCur = 0
        progressTot = 0.01 * len(bones) * (endframe-startframe)/scene.frame_step
    
        for boneName in bones:
            if not boneName in bonesMap:
                log.warning("Skeleton does not contain bone {:s}".format(boneName))
                continue

            if not boneName in armatureObj.pose.bones:
                log.warning("Pose does not contain bone {:s}".format(boneName))
                continue
            
            tTrack = TTrack(boneName)
            
            # Get the Blender pose bone (bpy.types.PoseBone)
            poseBone = armatureObj.pose.bones[boneName]
            parent = poseBone.parent
        
            # For each frame
            for time in range( startframe, endframe, scene.frame_step):
                
                if (progressCur % 10) == 0:
                    print("{:.3f}%\r".format(progressCur / progressTot), end='' )
                progressCur += 1
                
                # Set frame
                scene.frame_set(time)
            
                # This matrix is referred to the armature (object space)
                poseMatrix = poseBone.matrix.copy()

                if parent:
                    # Bone matrix relative to its parent bone
                    poseMatrix = parent.matrix.inverted() * poseMatrix
                else:
                    # Root bone matrix relative to the armature
                    poseMatrix = Matrix.Rotation(math.radians(-90.0), 4, 'X' ) * originMatrix * poseMatrix

                if tOptions.scale != 1.0:
                    poseMatrix.translation *= tOptions.scale

                # Extract position and rotation relative to parent in parent space        
                t = poseMatrix.to_translation()
                q = poseMatrix.to_quaternion()
                s = poseMatrix.to_scale()
                
                # Convert position and rotation to left hand:
                tl = Vector((t.x, t.y, -t.z))
                ql = Quaternion((q.w, -q.x, -q.y, q.z))
                sl = Vector((s.x, s.y, s.z))
                
                if not tOptions.doAnimationPos:
                    tl = None
                if not tOptions.doAnimationRot:
                    ql = None
                if not tOptions.doAnimationSca:
                    sl = None
        
                tFrame = TFrame(time / scene.render.fps, tl, ql, sl)
                
                if not tTrack.frames or tTrack.frames[-1].hasMoved(tFrame):
                    tTrack.frames.append(tFrame)
                
            if tTrack.frames:
                tAnimation.tracks.append(tTrack)

        if tAnimation.tracks:
            animationsList.append(tAnimation)
        
        if type(object) is bpy.types.NlaTrack:
            object.is_solo = False

    # Restore initial action and frame
    armatureObj.animation_data.action = savedAction
    armatureObj.animation_data.use_nla = savedUseNla
    scene.frame_set(savedFrame)


#--------------------
# Decompose geometries and morphs
#--------------------

def DecomposeMesh(scene, meshObj, tData, tOptions, errorsDict):

    try:
        invalidUvIndices = errorsDict["invalid UV"]
    except KeyError:
        invalidUvIndices = set()
        errorsDict["invalid UV"] = invalidUvIndices

    verticesList = tData.verticesList
    geometriesList = tData.geometriesList
    materialsList = tData.materialsList
    materialGeometryMap = tData.materialGeometryMap
    morphsList = tData.morphsList
    bonesMap = tData.bonesMap
    
    verticesMap = {}
    
    # Create a Mesh datablock with modifiers applied
    # (note: do not apply if not needed, it loses precision)
    mesh = meshObj.to_mesh(scene, tOptions.applyModifiers, tOptions.applySettings)
    
    log.info("Decomposing mesh: {:s} ({:d} vertices)".format(meshObj.name, len(mesh.vertices)) )
    
    # If we use the object local origin (orange dot) we don't need trasformations
    posMatrix = Matrix.Identity(4)
    normalMatrix = Matrix.Identity(4)
    
    if tOptions.globalOrigin:
        posMatrix = meshObj.matrix_world
        # Use the inverse transpose to rotate normals without scaling (math trick)
        normalMatrix = meshObj.matrix_world.inverted().transposed()
    
    # Apply custom scaling last
    if tOptions.scale != 1.0:
        posMatrix = Matrix.Scale(tOptions.scale, 4) * posMatrix 

    # Vertices map: vertex Blender index to TVertex index
    faceVertexMap = {}

    # Here we store geometriesList indices of geometries with new vertices in its last LOD
    # We use this to create a new LOD only once per geometry and to filter where we have
    # to optimize and recalculate tangents
    updatedGeometryIndices = set()

    # Mesh vertex groups
    meshVertexGroups = meshObj.vertex_groups
    notBonesGroups = set()
    missingGroups = set()

    # Python trick: C = A and B, if A is False (None, empty list) then C=A, if A is
    # True (object, populated list) then C=B
    
    # Check if the mesh has UV data
    uvs = None
    uvs2 = None
    # In every texture of every material search if the name ends in "_UV1" or "_UV2"
    for material in mesh.materials:
        if not material:
            continue
        for texture in material.texture_slots:
            if texture and texture.texture_coords == "UV" and texture.uv_layer:
                if texture.name.endswith("_UV") or texture.name.endswith("_UV1"):
                    uvs = mesh.tessface_uv_textures[texture.uv_layer].data
                elif texture.name.endswith("_UV2"):
                    uvs2 = mesh.tessface_uv_textures[texture.uv_layer].data
    # If still we don't have UV1, try the current UV map selected
    if not uvs and mesh.tessface_uv_textures.active:
        uvs = mesh.tessface_uv_textures.active.data
    # If still we don't have UV1, try the first UV map in Blender
    if not uvs and mesh.tessface_uv_textures:
        uvs = mesh.tessface_uv_textures[0].data
    if tOptions.doGeometryUV and not uvs:
        log.warning("Object {:s} has no UV data".format(meshObj.name))
    if tOptions.doGeometryUV2 and not uvs2:
        log.warning("Object {:s} has no texture with UV2 data. Append _UV2 to the texture slot name".format(meshObj.name))
    
    # Check if the mesh has vertex color data
    colorsRgb = None
    colorsAlpha = None
    # In vertex colors layer search if the name ends in "_RGB" or "_ALPHA"
    for vertexColors in mesh.tessface_vertex_colors:
        if not colorsRgb and vertexColors.name.endswith("_RGB"):
            colorsRgb = vertexColors.data
        if not colorsAlpha and vertexColors.name.endswith("_ALPHA"):
            colorsAlpha = vertexColors.data
    # If still we don't have RGB, try the current vertex color layer selected
    if not colorsRgb and mesh.tessface_vertex_colors.active:
        colorsRgb = mesh.tessface_vertex_colors.active.data
    # If still we don't have RGB, try the first vertex color layer in Blender
    if not colorsRgb and mesh.tessface_vertex_colors:
        colorsRgb = mesh.tessface_vertex_colors[0].data
    if tOptions.doGeometryCol and not colorsRgb:
        log.warning("Object {:s} has no rgb color data".format(meshObj.name))
    if tOptions.doGeometryColAlpha and not colorsAlpha:
        log.warning("Object {:s} has no alpha color data. Append _ALPHA to the color layer name".format(meshObj.name))

    if tOptions.doMaterials:
        if scene.render.engine == 'CYCLES':
            log.warning("Cycles render engine not supported")
        if not mesh.materials:
            log.warning("Object {:s} has no materials data".format(meshObj.name))

    # Progress counter
    progressCur = 0
    progressTot = 0.01 * len(mesh.tessfaces)

    for face in mesh.tessfaces:

        if (progressCur % 10) == 0:
            print("{:.3f}%\r".format(progressCur / progressTot), end='' )
        progressCur += 1

        # Skip if this face has less than 3 unique vertices
        # (a frozenset is an immutable set of unique elements)
        if len(frozenset(face.vertices)) < 3: 
            face.hide = True
            continue

        if face.hide:
            continue

        # Get face vertices UV, type: MeshTextureFace(bpy_struct)
        faceUv = uvs and uvs[face.index]
        faceUv2 = uvs2 and uvs2[face.index]

        # Get face 4 vertices colors
        fcol = colorsRgb and colorsRgb[face.index]
        faceRgbColor = fcol and (fcol.color1, fcol.color2, fcol.color3, fcol.color4)
        fcol = colorsAlpha and colorsAlpha[face.index]
        faceAlphaColor = fcol and (fcol.color1, fcol.color2, fcol.color3, fcol.color4)

        # Get the face material
        # If no material is associated then face.material_index is 0 but mesh.materials
        # is not None
        material = None
        if mesh.materials and len(mesh.materials):
            material = mesh.materials[face.material_index]
        
        # Add the material if it is new
        if tOptions.doMaterials and material and (not material.name in materialsList):
            tMaterial = TMaterial(material.name)
            materialsList.append(tMaterial)

            tMaterial.diffuseColor = material.diffuse_color
            tMaterial.diffuseIntensity = material.diffuse_intensity
            tMaterial.specularColor = material.specular_color
            tMaterial.specularIntensity = material.specular_intensity
            tMaterial.specularHardness = material.specular_hardness
            tMaterial.twoSided = mesh.show_double_sided 
            if material.use_transparency:
                tMaterial.opacity = material.alpha
                
            # In reverse order so the first slots have precedence
            for texture in reversed(material.texture_slots):
                if texture is None or texture.texture_coords != 'UV':
                    continue
                textureData = bpy.data.textures[texture.name]
                if textureData.type != 'IMAGE':
                    continue
                if textureData.image is None:
                    continue
                imageName = textureData.image.name
                if texture.use_map_color_diffuse:
                    tMaterial.diffuseTexName = imageName
                if texture.use_map_normal:
                    tMaterial.normalTexName = imageName
                if texture.use_map_color_spec:
                    tMaterial.specularTexName = imageName
                if "_LIGHTMAP" in texture.name:
                    tMaterial.lightmapTexName = imageName
                ##tMaterial.imagePath = bpy.path.abspath(faceUv.image.filepath)

        # From the material name search for the geometry index, or add it to the map if missing
        materialName = material and material.name
        try:
            geometryIndex = materialGeometryMap[materialName]
        except KeyError:
            geometryIndex = len(geometriesList)
            geometriesList.append(TGeometry())
            materialGeometryMap[materialName] = geometryIndex
            log.info("New Geometry{:d} created for material {:s}".format(geometryIndex, materialName))

        # Get the geometry associated to the material
        geometry = geometriesList[geometryIndex]
        
        # Get the last LOD level, or add a new one if requested in the options
        if not geometry.lodLevels or (tOptions.newLod and geometryIndex not in updatedGeometryIndices):
            lodLevelIndex = len(geometry.lodLevels)
            tLodLevel = TLodLevel()
            tLodLevel.distance = tOptions.lodDistance
            geometry.lodLevels.append(tLodLevel)
            log.info("New LOD{:d} created for material {:s}".format(lodLevelIndex, materialName))
        else:
            tLodLevel = geometry.lodLevels[-1]

        # Add the index of the geometry we are going to update
        updatedGeometryIndices.add(geometryIndex)

        indexSet = tLodLevel.indexSet
        triangleList = tLodLevel.triangleList
            
        # Here we store all the indices of the face, then we decompose it into triangles
        tempList = []

        for i, vertexIndex in enumerate(face.vertices):
            # i: vertex index in the face (0..2 tris, 0..3 quad)
            # vertexIndex: vertex index in Blender buffer

            # Blender vertex
            vertex = mesh.vertices[vertexIndex]

            position = posMatrix * vertex.co
                
            # if face is smooth use vertex normal else use face normal
            if face.use_smooth:
                normal = vertex.normal
            else:
                normal = face.normal
            normal = normalMatrix * normal
            
            # Create a new vertex
            tVertex = TVertex()
            
            # Set Blender index
            tVertex.blenderIndex = vertexIndex

            # Set Vertex position
            if tOptions.doGeometryPos:
                tVertex.pos = Vector((position.x, position.z, position.y))

            # Set Vertex normal
            if tOptions.doGeometryNor:
                tVertex.normal = Vector((normal.x, normal.z, normal.y))
                
            # Set Vertex UV coordinates
            if tOptions.doGeometryUV:
                if faceUv:
                    uv = faceUv.uv[i]
                    tVertex.uv = Vector((uv[0], 1.0 - uv[1]))
                elif tOptions.doForceElements:
                    tVertex.uv = Vector((0.0, 0.0))
            if tOptions.doGeometryUV2:
                if faceUv2:
                    uv2 = faceUv.uv[i]
                    tVertex.uv2 = Vector((uv2[0], 1.0 - uv2[1]))
                elif tOptions.doForceElements:
                    tVertex.uv2 = Vector((0.0, 0.0))

            # Set Vertex color
            if tOptions.doGeometryCol or tOptions.doGeometryColAlpha:
                color = [0, 0, 0, 255]
                if faceRgbColor or faceAlphaColor:
                    if faceRgbColor:
                        # This is an array of 3 floats from 0.0 to 1.0
                        rgb = faceRgbColor[i]
                        # Approx 255*float to the closest int
                        color[:3] = ( int(round(rgb.r * 255.0)), 
                                      int(round(rgb.g * 255.0)), 
                                      int(round(rgb.b * 255.0)) )
                    if faceAlphaColor:
                        # For Alpha use Value of HSV
                        alpha = faceAlphaColor[i]
                        color[3] = int(round(alpha.v * 255.0))
                    tVertex.color = tuple(color)
                elif tOptions.doForceElements:
                    tVertex.color = tuple(color)
                    
            # Set Vertex bones weights
            if tOptions.doGeometryWei:
                weights = []
                # Scan all the vertex group associated to the vertex, type: VertexGroupElement(bpy_struct)
                for g in vertex.groups:
                    # The group name should be the bone name, but it can also be an user made vertex group
                    try:
                        boneName = meshVertexGroups[g.group].name
                        try:
                            boneIndex = bonesMap[boneName].index
                            if g.weight > 0.0 or not weights:
                                weights.append( (boneIndex, g.weight) )
                        except KeyError:
                            notBonesGroups.add(boneName)
                    except IndexError:
                        missingGroups.add(str(g.group))
                # If we found no bone weight (not even one with weight zero) leave the list equal to None
                if weights:
                    tVertex.weights = weights
                elif tOptions.doForceElements:
                    tVertex.weights = [(0, 0.0)]
                
            # All this code do is "tVertexIndex = verticesMapList.index(tVertex)", but we use
            # a map to speed up.

            # Get an hash of the vertex (different vertices with the same hash are ok)
            vertexHash = hash(tVertex)
            
            try:
                # Get a list of vertex indices with the same hash
                verticesMapList = verticesMap[vertexHash]
            except KeyError:
                # If the hash is not mapped, create a new list (we should use sets but lists are faster)
                verticesMapList = []
                verticesMap[vertexHash] = verticesMapList
            
            # For each index in the list, test if it is the same as the current tVertex.
            # If Position, Normal and UV must be the same get its index.
            ## tVertexIndex = next((j for j in verticesMapList if verticesList[j] == tVertex), None)
            tVertexIndex = None
            for j in verticesMapList:
                if verticesList[j] == tVertex:
                    tVertexIndex = j
                    break

            # If we cannot find it, the vertex is new, add it to the list, and its index to the map list
            if tVertexIndex is None:
                tVertexIndex = len(verticesList)
                verticesList.append(tVertex)
                verticesMapList.append(tVertexIndex)


            # Add the vertex index to the temp list to create triangles later
            tempList.append(tVertexIndex)
                        
            # Map Blender face index and Blender vertex index to our TVertex index (this is used later by Morphs)
            faceVertexMap[(face.index, vertexIndex)] = tVertexIndex
            
            # Save every unique vertex this LOD is using
            indexSet.add(tVertexIndex)

            # Create triangles
            if i == 2:
                triangle = (tempList[0], tempList[2], tempList[1])
                triangleList.append(triangle)

            if i == 3:
                triangle = (tempList[0], tempList[3], tempList[2])
                triangleList.append(triangle)
        # end loop vertices
    # end loop faces

    tOptions.newLod = False

    if notBonesGroups:
        log.warning("Maybe these groups have no bone: {:s}".format( ", ".join(notBonesGroups) ))
    if missingGroups:
        log.warning("These group indices are missing: {:s}".format( ", ".join(missingGroups) ))

    #for geometry in geometriesList:
    #    # Generate tangents (only for first LOD level)
    #    if tOptions.doGeometryTan and geometry.lodLevels:
    #        lodLevel = geometry.lodLevels[-1] # era: geometry.lodLevels[0]
    #        log.info("Generating tangents for {:s} on {:d} indices".format(meshObj.name, len(lodLevel.indexSet)) )
    #        GenerateTangents(lodLevel, verticesList, invalidUvIndices)
    #    # Optimize vertex index buffer for each LOD level
    #    if tOptions.doOptimizeIndices:
    #        for lodLevel in geometry.lodLevels:
    #            log.info("Optimizing indices {:s}".format(meshObj.name) )
    #            OptimizeIndices(lodLevel)
    
    #if geometry.lodLevels:
    #    lodLevel = geometry.lodLevels[-1]
    #    # Generate tangents for this LOD level
    #    if tOptions.doGeometryTan:
    #        log.info("Generating tangents on {:d} indices for {:s}".format(len(lodLevel.indexSet), meshObj.name) )
    #        GenerateTangents(lodLevel, verticesList, invalidUvIndices)
    #    # Optimize vertex index buffer for this LOD level
    #    if tOptions.doOptimizeIndices:
    #        log.info("Optimizing {:d} indices for {:s}".format(len(lodLevel.indexSet), meshObj.name) )
    #        OptimizeIndices(lodLevel)

    # For each geometries with new vertices
    for geometryIndex in updatedGeometryIndices:
        geometry = geometriesList[geometryIndex]
        geometriesList[geometryIndex]
        # Only the last LOD was modified (even if it wasn't a new LOD)
        lodLevel = geometry.lodLevels[-1]
        # Generate tangents for this LOD level
        if tOptions.doGeometryTan:
            log.info("Generating tangents on {:d} indices for {:s} Geometry{:d}"
                    .format(len(lodLevel.indexSet), meshObj.name, geometryIndex) )
            GenerateTangents(lodLevel, verticesList, invalidUvIndices)
        # Optimize vertex index buffer for this LOD level
        if tOptions.doOptimizeIndices:
            log.info("Optimizing {:d} indices for {:s} Geometry{:d}"
                    .format(len(lodLevel.indexSet), meshObj.name, geometryIndex) )
            OptimizeIndices(lodLevel)
    
    # Check if we need and can work on shape keys (morphs)
    shapeKeys = meshObj.data.shape_keys
    keyBlocks = []
    if tOptions.doMorphs:
        if not shapeKeys or len(shapeKeys.key_blocks) < 1:
            log.warning("Object {:s} has no shape keys".format(meshObj.name))
        else:
            keyBlocks = shapeKeys.key_blocks

    # Decompose shape keys (morphs)
    for j, block in enumerate(keyBlocks):
        # Skip 'Basis' shape key
        if j == 0:
            continue
        
        tMorph = TMorph(block.name)
        
        log.info("Decomposing shape: {:s} ({:d} vertices)".format(block.name, len(block.data)) )

        # Make a temporary copy of the mesh
        shapeMesh = mesh.copy()
        
        if len(shapeMesh.vertices) != len(block.data):
            log.error("Vertex count mismatch on shape {:s}.".format(block.name))
            continue
        
        # Appy the shape
        for i, data in enumerate(block.data):
            shapeMesh.vertices[i].co = data.co

        # Recalculate normals
        shapeMesh.update(calc_edges = True, calc_tessface = True)
        ##shapeMesh.calc_tessface()
        ##shapeMesh.calc_normals()
        
        # TODO: if set use 'vertex group' of the shape to filter affected vertices
        # TODO: can we use mesh tessfaces and not shapeMesh tessfaces ?
        
        for face in shapeMesh.tessfaces:
            if face.hide:
                continue

            # TODO: add only affected triangles not faces, use morphed as a mask
            morphed = False

            # In this list we store vertex index and morphed vertex of each face, we'll add them
            # to the morph only if at least one vertex on the face is affected by the moprh
            tempList = []
            
            # For each Blender vertex index in the face
            for vertexIndex in face.vertices:

                # Get the Blender morphed vertex
                vertex = shapeMesh.vertices[vertexIndex]
                
                position = posMatrix * vertex.co
                
                # If face is smooth use vertex normal else use face normal
                if face.use_smooth:
                    normal = vertex.normal
                else:
                    normal = face.normal
                normal = normalMatrix * normal

                # Try to find the TVertex index corresponding to this Blender vertex index
                try:
                    tVertexIndex = faceVertexMap[(face.index, vertexIndex)]
                except KeyError:
                    log.error("Cannot find vertex {:d} of face {:d} of shape {:s}."
                          .format(vertexIndex, face.index, block.name) )
                    continue

                # Get the original not morphed TVertex
                tVertex = verticesList[tVertexIndex]
                   
                # Create a new morphed vertex
                # (note: this vertex stores absolute values, not relative to original values)
                tMorphVertex = TVertex()

                # Set Blender index
                tMorphVertex.blenderIndex = vertexIndex

                # Set Vertex position
                tMorphVertex.pos = Vector((position.x, position.z, position.y))

                # Set Vertex normal
                if tOptions.doMorphNor:
                    tMorphVertex.normal = Vector((normal.x, normal.z, normal.y))
                
                # If we have UV, copy them to the TVertex, we only need them to calculate tangents
                if tOptions.doMorphUV:
                    if tVertex.uv:
                        tMorphVertex.uv = tVertex.uv
                    elif tOptions.doForceElements:
                        tVertex.uv = Vector(0.0, 0.0)
                
                # Save vertex index and morphed vertex, to be added later if at least one
                # vertex in the face was morphed
                tempList.append((tVertexIndex, tMorphVertex))
                
                # Check if the morph has effect
                if tMorphVertex != tVertex or tVertex.pos is None:
                    morphed = True
            
            # If at least one vertex in the face was morphed
            if morphed:
                # Add vertices to the morph
                for i, (tVertexIndex, tMorphVertex) in enumerate(tempList):
                    try:
                        # Check if already present
                        oldTMorphVertex = tMorph.vertexMap[tVertexIndex]
                        if tMorphVertex != oldTMorphVertex:
                            log.error('Different vertex {:d} of face {:d} of shape {:s}.'
                                .format(vertexIndex, face.index, block.name) )
                            continue
                    except KeyError:
                        # Add a new morph vertex
                        tMorph.vertexMap[tVertexIndex] = tMorphVertex
                        
                    # Save how many unique vertex this LOD is using (for tangents calculation)
                    tMorph.indexSet.add(tVertexIndex)

                    # Create triangles (for tangents calculation)
                    if i == 2:
                        triangle = (tempList[0][0], tempList[2][0], tempList[1][0])
                        tMorph.triangleList.append(triangle)

                    if i == 3:
                        triangle = (tempList[0][0], tempList[3][0], tempList[2][0])
                        tMorph.triangleList.append(triangle)
                    
        if tOptions.doMorphTan:
            log.info("Generating morph tangents {:s}".format(block.name) )
            GenerateTangents(tMorph, tMorph.vertexMap, None)

        # If valid add the morph to the model list
        if tMorph.vertexMap:
            morphsList.append(tMorph)
        else:
            log.warning('Empty shape {:s}.'.format(block.name))

        # Delete the temporary copy 
        bpy.data.meshes.remove(shapeMesh)

    bpy.data.meshes.remove(mesh)    

    return

#--------------------
# Scan objects
#--------------------

# Scan and decompose objects
def Scan(context, tDataList, tOptions):
    
    scene = context.scene
    
    # Get all objects in the scene or only the selected
    if tOptions.onlySelected: 
        objs = context.selected_objects 
    else:
        objs = scene.objects
    
    # Sort by name
    objs = sorted(objs, key = operator.attrgetter('name'))

    tData = None
    noWork = True
    lodName = None
    
    for obj in objs:
        if obj.type == 'MESH':

            noWork = False
        
            name = obj.name
            # Are we creating a new container (TData) for a new mesh?
            createNew = True
            
            log.info("---- Decomposing {:s} ----".format(name))

            # Search in the object's name if it is a '_LOD': <name>_LOD<distance>
            # (LODs must have dot aligned distance, ex. nameLOD09.0, nameLOD12.0)
            if tOptions.useLods:
                splitted = name.rsplit(sep="_LOD", maxsplit=1)
                try:
                    distance = float(splitted[1])
                    name = splitted[0].rstrip()
                    if lodName is None or lodName != name:
                        # This is the first LOD of a new object
                        lodName = name
                        if distance != 0.0:
                            log.warning("First LOD should have 0.0 distance")
                    else:
                        # This is a lower LOD of the same object
                        createNew = False
                        tOptions.newLod = True
                        if distance <= tOptions.lodDistance:
                            log.warning("Wrong LOD sequence: {:d} then {:d}".format(tOptions.lodDistance, distance) )
                        tOptions.lodDistance = distance
                    log.info("Added as LOD with distance {:.3f}".format(distance))
                except (IndexError, ValueError):
                    log.warning("Object {:s} has no LODs".format(name) )
        
            if tOptions.mergeObjects and createNew:
                if tData:
                    # To create a new geometry in the same tData, clear the 'Material to Geometry' dict
                    tData.materialGeometryMap.clear()
                if tOptions.useLods:
                    log.warning("Merge and LODs should be not used together")
                createNew = False
            
            # Create a new container where to save decomposed data
            if not tData or createNew:
                tData = TData()
                # If we are merging objects, if it exists use the current object name
                if tOptions.mergeObjects and context.selected_objects and context.selected_objects[0].name:
                    tData.objectName = context.selected_objects[0].name
                else:
                    tData.objectName = name
                tDataList.append(tData)
                tOptions.newLod = True
                tOptions.lodDistance = 0.0
            
            # First we need to populate the skeleton, then animations and then geometries
            if tOptions.doBones:
                armatureObj = None
                # Check if obj has an armature parent, and if it is not attached to a bone (like hair to head bone)
                if obj.parent and obj.parent.type == 'ARMATURE' and obj.parent_type != 'BONE':
                    armatureObj = obj.parent
                else:
                    # Check if there is an Armature modifier
                    for modifier in obj.modifiers:
                        if modifier.type == 'ARMATURE' and modifier.object and modifier.object.type == 'ARMATURE':
                            armatureObj = modifier.object
                            break
                # Decompose armature and animations
                if armatureObj:
                    DecomposeArmature(scene, armatureObj, tData, tOptions)
                    if tOptions.doAnimations:
                        DecomposeActions(scene, armatureObj, tData, tOptions)
                else:
                    log.warning("Object {:s} has no armature".format(name) )

            # Decompose geometries
            if tOptions.doGeometries:
                DecomposeMesh(scene, obj, tData, tOptions, tData.errorsDict)

    if noWork:
        log.warning("No objects to work on")

#-----------------------------------------------------------------------------

if __name__ == "__main__":

    print("------------------------------------------------------")
    startTime = time.time()

    tDataList = []
    tOptions = TOptions()
    
    Scan(bpy.context, tDataList, tOptions)
    if tDataList:
        PrintAll(tDataList[0])
                
    print("Executed in {:.4f} sec".format(time.time() - startTime) )
    print("------------------------------------------------------")
    
    