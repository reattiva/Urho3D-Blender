
#
# This script is licensed as public domain.
#

import os
import logging

log = logging.getLogger("ExportLogger")

# Create output path
def ComposePath(path, standardDir, useStandardDirs):
    if useStandardDirs:
        path = os.path.join(path, standardDir)

    if not os.path.isdir(path):
        log.info( "Creating path {:s}".format(path) )
        os.makedirs(path)
        
    return path
