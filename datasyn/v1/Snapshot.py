import numpy as np
import networkx as nx

class Snapshot:
    ''' represent an instant moment of the scene, including geometry and rendering
    '''
    NodeKey : str = 'snapshot_node'
        
    @property
    def graph(self) -> nx.DiGraph:
        ''' graph of snapshot nodes
        '''
        raise NotImplementedError()
    
    def get_node_by_name(self, node_name : str) -> "SnapshotNode":
        ''' get a snapshot node by its name, None if the name does not exists
        '''
        g = self.graph
        node_data = g.nodes.get(node_name)
        if not node_data:
            return None
        x = node_data.get(__class__.NodeKey)
        return x
    
    def get_nodes(self, root_only : bool = False) -> list["SnapshotNode"]:
        ''' get all snapshot nodes, indexed by their name. If the graph does not exists, return None.
        
        parameters
        ------------
        root_only
            If True, only include nodes that have no parent
            
        return
        ----------
        name2node
            snapshot nodes indexed by name
        '''
        if self.graph is None:
            return None
        
        output : list[SnapshotNode] = []
        g = self.graph
        for name, data in g.nodes.items():
            if root_only and g.in_degree[name] > 0:
                continue
            x = data.get(__class__.NodeKey)
            output.append(x)
        return output
    
class SnapshotNode:
    ''' object in snapshot
    '''
        
    @property
    def name(self) -> str:
        ''' unique name of this node
        '''
        raise NotImplementedError()
    
    @property
    def owner(self) -> Snapshot:
        ''' the snapshot that contains this node
        '''
        raise NotImplementedError()
    
    @property
    def depth_in_graph(self)->int:
        ''' number of hops from the root in the snapshot graph
        '''
        if not self.owner:
            return None
        
        g = self.owner.graph
        dist_from_root = len(list(nx.edge_dfs(g, self.name, orientation='reverse')))
        return dist_from_root