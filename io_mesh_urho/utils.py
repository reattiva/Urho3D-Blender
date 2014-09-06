
#
# This script is licensed as public domain.
#

import os
import logging

log = logging.getLogger("ExportLogger")


def enum(**enums):
    return type('Enum', (), enums)
PathType = enum(
    ROOT        = "ROOT",
    MODELS      = "MODE",
    ANIMATIONS  = "ANIM",
    TRIGGERS    = "TRIG",
    MATERIALS   = "MATE",
    TECHNIQUES  = "TECH",
    TEXTURES    = "TEXT",
    MATLIST     = "MATL",
    OBJECTS     = "OBJE",
    SCENES      = "SCEN")

# Options for file utils
class FOptions:
    def __init__(self):
        self.useStandardDirs = True
        self.fileOverwrite = False
        self.paths = {}
        self.exts = {
                        PathType.MODELS : "mdl",
                        PathType.ANIMATIONS : "ani",
                        PathType.TRIGGERS : "xml",
                        PathType.MATERIALS : "xml",
                        PathType.TECHNIQUES : "xml",
                        PathType.TEXTURES : "png",
                        PathType.MATLIST : "txt",
                        PathType.OBJECTS : "xml",
                        PathType.SCENES : "xml"
                    }
        self.preserveExtTemp = False

'''
# Get a file path for the object 'name' in a folder of type 'pathType'
def GetFilepath(pathType, name, fOptions):

    filename = name

    if type(filename) is list or type(filename) is tuple:
        filename = os.path.sep.join(filename)

    # Replace all characters besides A-Z, a-z, 0-9 with '_'
    #filename = bpy.path.clean_name(filename)

    # Add extension to the filename, if 'name' has an exntesion we have the option
    # to preserve it
    ext = fOptions.exts[pathType]
    if ext and (not fOptions.preserveExtTemp or os.path.extsep not in filename):
        filename += os.path.extsep + ext
        #filename = bpy.path.ensure_ext(filename, ".mdl")
    fOptions.preserveExtTemp = False

    # Append the relative path
    if fOptions.useStandardDirs:
        filename = os.path.join(fOptions.paths[pathType], filename)

    # Remove the separator if present
    if len(filename) > 0 and filename[0] == os.path.sep:
        filename = filename[1:]
        
    # Compose the full file path
    fullpath = os.path.join(fOptions.paths[PathType.ROOT], filename)

    # Create the path is missing
    if not os.path.isdir(fullpath):
        log.info( "Creating path {:s}".format(fullpath) )
        os.makedirs(fullpath)

    #os.path.relpath(fullpath, basepath) -> rel1\\rel2
    
    # Return full file path and relative file path
    return (fullpath, filename)
'''

# Get a file path for the object 'name' in a folder of type 'pathType'
def GetFilepath(pathType, name, fOptions):

    # Get the root path
    rootPath = fOptions.paths[PathType.ROOT]

    # Append the relative path to get the full path
    fullPath = rootPath
    if fOptions.useStandardDirs:
        fullPath = os.path.join(fullPath, fOptions.paths[pathType])

    # Create the full path if missing
    if not os.path.isdir(fullPath):
        log.info( "Creating path {:s}".format(fullPath) )
        os.makedirs(fullPath)

    # Compose filename
    filename = name
    if type(filename) is list or type(filename) is tuple:
        filename = os.path.sep.join(filename)

    # Add extension to the filename, if present we can preserve the extension
    ext = fOptions.exts[pathType]
    if ext and (not fOptions.preserveExtTemp or os.path.extsep not in filename):
        filename += os.path.extsep + ext
        #filename = bpy.path.ensure_ext(filename, ".mdl")
    fOptions.preserveExtTemp = False

    # Replace all characters besides A-Z, a-z, 0-9 with '_'
    #filename = bpy.path.clean_name(filename)

    # Compose the full file path
    fileFullPath = os.path.join(fullPath, filename)

    # Get the Urho path (relative to root)
    fileUrhoPath = os.path.relpath(fileFullPath, rootPath)
    fileUrhoPath = fileUrhoPath.replace(os.path.sep, '/')

    # Return full file path and relative file path
    return (fileFullPath, fileUrhoPath)


# Check if 'filepath' is valid
def CheckFilepath(filepath, fOptions):

    fp = filepath
    if type(filepath) is tuple:
        fp = filepath[0]

    if os.path.exists(fp) and not fOptions.fileOverwrite:
        log.error( "File already exists {:s}".format(fp) )
        return False
        
    return True
