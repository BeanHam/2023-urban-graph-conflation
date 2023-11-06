import os
import json
import momepy
import geojson
import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely import LineString
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


def segment_check(lines):
    
    """
    Check if geo linestring is made up with several small linestrings.
    
    If yes and if the start and end node is the same (a loop), split the linestring into small segments.
    """
    
    new_lines = []
    for line in lines.geometry:
        coords = list(line.coords)
        if len(coords) == 2: new_lines.append(line)
        elif (coords[0] != coords[-1]): new_lines.append(line)
        else: new_lines += [LineString([coords[i], coords[i+1]]) for i in range(len(coords)-1)]
    new_lines = gpd.GeoDataFrame(new_lines, columns=['geometry'])       
    return new_lines

def node_alignment(graphs, names):
    
    # ====================
    # detect largest graph
    # ====================    
    large_index = np.argmax([len(G) for G in graphs])
    other_index = set(range(len(graphs)))-{large_index}
    G_large = graphs[large_index]
    G_large_name = names[large_index]
    G_other = [graphs[i] for i in other_index]
    G_other_name = [names[i] for i in other_index]
    
    # ====================    
    # large graph mapping
    # ====================    
    node_index = len(G_large.nodes())
    G_mappings = {}
    G_mappings[G_large_name] = {old_label:new_label for new_label, old_label in enumerate(G_large.nodes())}
    
    # ========================    
    # other two graphs mapping
    # ========================    
    for g in range(len(G_other)):
        G = G_other[g]
        name = G_other_name[g]
        
        # calculate pairwise distances
        dist = pairwise_distances(np.stack(G.nodes()), np.stack(G_large.nodes()))
        index=np.argmin(dist, axis=1).tolist()
        
        # find repetitive values
        rep_values = [item for item, count in Counter(index).items() if count > 1]
        
        # if no repetitive values, save mapping
        if len(rep_values) == 0:
            G_mappings[name] = {old_label:index[i] for i, old_label in enumerate(G.nodes())}    
            
        # if have repetitive values, check them
        else:
            for v in rep_values:
                rep_index = [i for i, x in enumerate(index) if x==v]
                d = [dist[r, v] for r in rep_index]
                min_index = np.argmin(d)
                other_index = set(range(len(d)))-{min_index}
                for i in other_index:
                    index[rep_index[i]] = node_index
                    
                    # add new nodes into the largest grapg mapping                    
                    G_mappings[G_large_name][list(G.nodes())[rep_index[i]]] = node_index        
                    node_index+=1
            G_mappings[name] = {old_label:index[i] for i, old_label in enumerate(G.nodes())}

    # ==========    
    # new graphs
    # ==========
    new_graphs = []
    for g in range(len(graphs)):
        G = graphs[g]
        name = names[g]
        G = nx.relabel_nodes(G, G_mappings[name])
        for i in set(range(node_index)) - set(G.nodes()):
            G.add_node(i)    
        
        H = nx.Graph()
        H.add_nodes_from(sorted(G.nodes(data=True)))
        H.add_edges_from(G.edges(data=True))
        new_graphs.append(H)
      
    # ==========    
    # attributes
    # ==========    
    attributes = {}
    for mapping in G_mappings:
        attributes[mapping] = {v: k for k, v in G_mappings[mapping].items()}
    for g in range(len(G_other)):
        name = G_other_name[g]
        for k in attributes[G_large_name]:
            if k not in attributes[name]:
                attributes[name][k] = attributes[G_large_name][k]
    # sort attributes
    for k in attributes:
        attributes[k] = dict(sorted(attributes[k].items()))    
    
    # sanity check
    assert len(attributes['G_osw']) == len(new_graphs[0])
    assert len(attributes['G_osm']) == len(new_graphs[1])
    assert len(attributes['G_sdot']) == len(new_graphs[2])    
        
    return new_graphs, attributes

def graph_extraction(data_path, data, osw, sdot):
    
    """
    Given street annotations in linestrings, and city block polygons,
    
    extract street annotations in each city block.
    
    Convert linestrings into graph objects.
    
    """
    
    tags = {"highway": ["residential", "services"]}
    names = ['G_osw', 'G_osm', 'G_sdot']
    for i in tqdm(range(len(data))):
        
        # loop through seattle blocks
        geom = data.iloc[[i]]
        
        # joint lines with seattle block
        osw_lines=gpd.sjoin(osw, geom)[['geometry']]
        sdot_lines=gpd.sjoin(sdot, geom)[['geometry']]
            
        # sometimes, no response from OSM
        try: osm_lines=ox.features_from_polygon(geom.geometry.values[0],tags)
        except: continue
        
        # construct graphs
        try:
            G_osw = momepy.gdf_to_nx(osw_lines, approach='primal',multigraph=True)
            G_sdot = momepy.gdf_to_nx(sdot_lines, approach='primal',multigraph=True)
            G_osm = momepy.gdf_to_nx(osm_lines, approach='primal',multigraph=True)
            
            # check graph sizes
            lens = np.array([
                len(G_osw.nodes()),
                len(G_sdot.nodes()),
                len(G_osm.nodes()),
                len(G_osw.edges()),
                len(G_osw.edges()),
                len(G_osw.edges())                
            ])
            
            # need to have at least 1 nodes
            if np.any(lens<=1): continue
            
            # node alignment & attributes            
            new_graphs, attributes = node_alignment([G_osw, G_osm, G_sdot], names)
            G_osw, G_osm, G_sdot = new_graphs
            G_osw_a, G_osm_a, G_sdot_a = attributes['G_osw'],attributes['G_osm'], attributes['G_sdot']
            
            # save graphs & attributes
            nx.write_graph6(G_osw, data_path+f"graphs/osw/graph_{i}")
            nx.write_graph6(G_osm, data_path+f"graphs/osm/graph_{i}")
            nx.write_graph6(G_sdot, data_path+f"graphs/sdot/graph_{i}")
            with open(data_path+f"attributes/osw/graph_{i}.json", 'w') as f:
                json.dump(G_osw_a, f)
            with open(data_path+f"attributes/osm/graph_{i}.json", 'w') as f:
                json.dump(G_osm_a, f)
            with open(data_path+f"attributes/sdot/graph_{i}.json", 'w') as f:
                json.dump(G_sdot_a, f)
        except:
            continue    

def main():
    
    # =============
    # parameters
    # =============
    epsg=4326
    crs=4326
    data_path = '../../../data/2023-graph-conflation/'
    city_block_path = data_path+'city-blocks/2010_Census_Block_Seattle_-_Housing_Statistics.shp'
    osw_data_path = data_path+'osw.geojson'
    sdot_data_path = data_path+'/sdot/Street_Network_Database_SND.shp'
    west_seattle_tract = 100*np.array(
        [96, 97.01, 97.02, 98, 99,105,106,107.01,107.02,108,112,113,114.01,114.02,115,116,120,121,264,265]
    )
    
    # =============
    # city blocks
    # =============
    print('  -- Load City Blocks...')
    seattle = gpd.read_file(city_block_path)
    seattle = seattle.to_crs(epsg=epsg)

    # =============
    # OSW data
    # =============
    print('  -- Load Open SideWalk Data...')
    with open(osw_data_path) as f: gj = geojson.load(f)
    osw = gj['features']
    osw = [f for f in osw if f['properties']['highway'] in ['residential', 'service']]
    osw = gpd.GeoDataFrame(
        [LineString(osw[i]['geometry']['coordinates']) for i in range(len(osw))], 
        columns=['geometry'],
        crs=crs
    )
    
    # =============
    # SDOT data
    # ============= 
    print('  -- Load Seattle Department of Transportation Data...')    
    sdot = gpd.read_file(sdot_data_path)
    sdot = sdot.to_crs(crs)
    
    # ==============
    # Extract Graphs
    # ==============
    print('  -- Extracting Seattle Graphs...')
    graph_extraction(data_path, seattle, osw, sdot)
    
    print('Done...')
    
if __name__ == "__main__":
    main()