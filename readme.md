Improvements
============
## PBR Support
Urho3D/RBFX engines takes metallic and roughness maps from different channels of a single RGB image instead of two grey scale images. This plugin has the ability of combining two grey scale metallic and roughness maps into single RGB image. To enable this feature "Copy textures" and "Compress specular map" options should be selected. Material file will be created automatically with respect to corresponding technique.
## Others
#### Alpha technique appends to material technique if alpha map are used
#### Roughness, metallic and alpha maps can be exported with "Copy textures" option now
#### Added functionality to cast shadows in exported scene, 'Cast shadows' checkbox can be found under 'Export Urho Prefabs' menu
## Bugfixes
#### A bug related with Python 3.11 fixed 
#### A bug caused to block material export on Blender 2.8 and above has been fixed

Urho3D-Blender
==============

[Blender](http://www.blender.org) to [Urho3D](https://urho3d.github.io) mesh exporter.

Guide [here](https://github.com/reattiva/Urho3D-Blender/blob/master/guide.txt).

Installation:
- download the repository zip file        
![download](https://cloud.githubusercontent.com/assets/5704756/26752822/f5ebaecc-4858-11e7-8e7c-35082ee751d3.png)
- menu "File"
- select "User Preferences..."
- select page "Add-ons"
- click "Install from File..."        
![install](https://cloud.githubusercontent.com/assets/5704756/26752823/fd119d7e-4858-11e7-9795-5d3b9d1a895c.png)
- select the downloaded zip file
- enable the addon

The addon is located in the "Properties" panel, at the end of the "Render" page (camera icon):
![location](https://cloud.githubusercontent.com/assets/5704756/26752826/0145c014-4859-11e7-9eb3-15f1724f3d6e.png)
