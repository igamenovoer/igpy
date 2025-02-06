
# data in scene-related graphs
    
class CoordinateType:
    World = 'world',    # relative to world coordinate
    Parent = 'parent',  # relative to parent coordinate. If not parent exists, it is the world coordinate
    Local = 'local'     # relative to itself
    
# keys for parsing unreal scene structure file
class UE_SceneDescKeys:
    RootActors = "actors"
    Components = 'components'
    UEClassName = 'class_name'
    UERootCompName = 'root_component'
    ObjectName = 'object_name'
    GLTFNode = 'gltf_node'
    AssetPath = 'asset_path'
    ParentActor = 'parent_actor'
    
class UE_ClassName:
    StaticMeshComponent = 'StaticMeshComponent'
    
class UE_MetaClassType:
    Component = 'Component'
    Actor = 'Actor'
    
class UE_SnapshotUserDataKeys:
    SemanticLabel = 'semantic_label'
    InstanceLabel = 'instance_label'
    BoundingBox = 'bbox'