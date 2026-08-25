# environment16.py
import gym
import numpy as np
import networkx as nx
import random
from gym import error, spaces, utils
from random import choice
import pandas as pd
import pickle
import json 
import os
import os.path
# PROJECT_ROOT = ls2ic_sdn_routing/  (5 levels up from this file's directory)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
import sys
sys.path.insert(0, os.path.join(PROJECT_ROOT, "utils"))
from setting import MONITOR_PERIOD
import gc
import defo_process_results as defoResults
import matplotlib.pyplot as plt
from itertools import cycle
import json
from types import SimpleNamespace

def estimate_lu_from_delay_loss(
    D_curr, D_prev, dt, loss_rate, tx_util_ratio,
    delay_thr, loss_thr, D_max_link, combine="sum", txutil_gate=0.95,
    clamp_max=3.0,
):
    """Return unclamped per-link LU (arrival/capacity, >=0) from observable signals.

    D_curr, D_prev : queue delay in seconds (env.graph['delay'])
    dt             : step duration (seconds)
    loss_rate      : per-link packet loss ratio in [0, 1]
    tx_util_ratio  : sent_throughput / cap in [0, 1]; below-saturation fallback
    D_max_link     : per-link queue-saturated delay ceiling (seconds)
    txutil_gate    : suppress delay/loss signals when tx_util below this; a link
                     averaged over dt cannot truly have A>C unless tx_util ~ 1.
                     Blocks burst-transient false positives on uncongested links.
    clamp_max      : upper ceiling for the returned LU. The formula
                     `1 + loss/(1-loss)` diverges as loss→1 (estimator numerical
                     noise), and util^p shaping amplifies spikes. Clamp prevents
                     replay-buffer poisoning by early-training random actions.
                     Unlike MLU=1 clamping, this is far from the policy's
                     operating range ([0.3, 1.5]) so no signal loss in practice.
    """
    # Pre-gate: if throughput util is clearly below saturation, treat any
    # delay/loss reading as burst-transient noise, not real congestion.
    if tx_util_ratio < txutil_gate:
        return float(tx_util_ratio)

    dDdt = (D_curr - D_prev) / max(dt, 1e-9)
    excess_delay = max(0.0, dDdt) if D_curr >= delay_thr else 0.0
    excess_loss = (loss_rate / max(1.0 - loss_rate, 1e-3)) if loss_rate >= loss_thr else 0.0
    saturated = (D_max_link > 0.0) and (D_curr >= 0.9 * D_max_link)

    if saturated:
        lu = max(1.0 + excess_delay, 1.0 + excess_loss)
        if excess_loss == 0.0 and D_max_link > 0.0:
            lu = max(lu, 1.0 + D_max_link / max(dt, 1e-9))
        return min(lu, clamp_max)

    if excess_delay == 0.0 and excess_loss == 0.0:
        return float(tx_util_ratio)

    if combine == "max":
        return min(1.0 + max(excess_delay, excess_loss), clamp_max)
    return min(1.0 + excess_delay + excess_loss, clamp_max)


class Env16(gym.Env):
    """
    Here I only take X% of the demands. There are some flags
    that indicate if to take the X% larger demands, the X% from the 5 most loaded links
    or random.

    Environment used in the middlepoint routing problem. Here we compute the SP to reach a middlepoint.
    We are using bidirectional links in this environment!
    In this environment we make the MP between edges.
    self.edge_state[:][0] = link utilization
    self.edge_state[:][1] = link capacity
    self.edge_state[:][2] = bw allocated (the one that goes from src to dst)
    """
    def __init__(self):
        self.graph = None # Here we store the graph as DiGraph (without repeated edges)
        self.source = None
        self.destination = None
        self.demand = None

        self.edge_state = None
        self.graph_topology_name = None # Here we store the name of the graph topology from the repetita dataset
        self.dataset_folder_name = None # Here we store the name of the repetita dataset being used: 2015Defo, 2016TopologyZoo_unary,2016TopologyZoo_inverseCapacity, etc. 

        self.diameter = None

        # Nx Graph where the nodes have features. Betweenness is allways normalized.
        # The other features are "raw" and are being normalized before prediction
        self.first = None
        self.firstTrueSize = None
        self.second = None
        self.between_feature = None

        self.percentage_demands = None # X% of the most loaded demands we use for optimization
        self.shufle_demands = False # If True we shuffle the list of traffic demands
        self.top_K_critical_demands = False # If we want to take the top X% of the 5 most loaded links
        self.num_critical_links = 5

        self.sp_middlepoints = None # For each src,dst we store the nodeId of the sp middlepoint
        self.shortest_paths = None # For each src,dst we store the shortest path to reach d
        self.sp_middlepoints_step = dict() # We store the midlepoint assignation before step() finishes

        # Mean and standard deviation of link betweenness
        self.mu_bet = None
        self.std_bet = None

        # Episode length in timesteps
        self.episode_length = None
        self.currentVal = None # Value used in hill_climbing way of choosing the next demand
        self.initial_maxLinkUti = None
        self.iter_list_elig_demn = None

        # Error at the end of episode to evaluate the learning process
        self.error_evaluation = None

        # Per-pair EMA state for reward_def="per_pair_ema_v1" (process-level
        # persistence: survives kpath_reset, resets each new process).
        # Mirrors _drl_or_s_state pattern.
        self._per_pair_ema_state = None
        # Ideal target link capacity: self.sumTM/self.numEdges
        self.target_link_capacity = None

        self.TM = None # Traffic matrix where self.TM[src][dst] indicates how many packets are sent from src to dst
        self.sumTM = None
        self.routing = None # Loaded routing matrix
        self.paths_Matrix_from_routing = None # We store a list of paths extracted from the routing matrix for each src-dst pair

        self.K = None
        self.use_K_path = False
        self.sp_pathk = None # For each src,dst we store the nodeId of the path k
        self.nodes = None # List of nodes to pick randomly from them
        self.ordered_edges = None
        self.edgesDict = dict() # Stores the position id of each edge in order
        self.previous_path = None

        self.src_dst_k_middlepoints = None # For each src, dst, we store the k middlepoints
        self.list_eligible_demands = None # Here we store those demands from DEFO that have one middlepoint. These demands are going to be eligible by our DRL agent.
        self.link_capacity_feature = None

        self.numNodes = None
        self.numEdges = None
        self.next_state = None

        # We store the edge that has maximum utilization
        # (src, dst, MaxUtilization)
        self.edgeMaxUti = None
        # (src, dst, StdUtilization)
        self.edgeStdUti = None
        # We store the edge that has minimum utilization
        # (src, dst, MaxUtilization)
        self.edgeMinUti = None 
        # We store the path with more bandwidth from the edge with maximum utilization
        # (src, dst, MaxBandwidth)
        self.patMaxBandwth = None 
        self.maxBandwidth = None

        self.episode_over = True
        self.reward = 0
        self.allPaths = dict() # Stores the paths for each src:dst pair

        self.paths_metrics_minmax_dict = None  # 跟 train_loader 全域變數同角色
        self.metrics = ['bwd_paths','delay_paths','loss_paths']
        self.K = None  # action_dim

    def init_minmax_dic_like_trainloader(self, num_node_1based: int):
        # num_node_1based = config["num_node"]
        size = num_node_1based + 1  # 1..N
        d = {}
        for i in range(1, size):
            d.setdefault(str(i), {})
            for j in range(1, size):
                d[str(i)].setdefault(str(j), {})
                for m in self.metrics:
                    d[str(i)][str(j)].setdefault(m, {})
                    d[str(i)][str(j)][m]['min'] = 100000000
                    d[str(i)][str(j)][m]['max'] = -1
        self.paths_metrics_minmax_dict = d


    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def add_features_to_edges(self):
        incId = 1
        for node in self.graph:
            for adj in self.graph[node]:
                if not 'betweenness' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['betweenness'] = 0
                if not 'edgeId' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['edgeId'] = incId
                if not 'numsp' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['numsp'] = 0
                if not 'utilization' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['utilization'] = 0
                if not 'capacity' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['capacity'] = 0
                if not 'weight' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['weight'] = 0
                if not 'kshortp' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['kshortp'] = 0
                if not 'crossing_paths' in self.graph[node][adj][0]: # We store all the src,dst from the paths crossing each edge
                    self.graph[node][adj][0]['crossing_paths'] = dict()
                incId = incId + 1

    # def num_shortest_path(self, topology):
    #     self.diameter = nx.diameter(self.graph)
    #     # Iterate over all node1,node2 pairs from the graph
    #     for n1 in range (0,self.numNodes):
    #         for n2 in range (0,self.numNodes):
    #             if (n1 != n2):
    #                 # Check if we added the element of the matrix
    #                 if str(n1)+':'+str(n2) not in self.allPaths:
    #                     self.allPaths[str(n1)+':'+str(n2)] = []
    #                 # First we compute the shortest paths taking into account the diameter
    #                 # [self.allPaths[str(n1)+':'+str(n2)].append(p) for p in nx.all_simple_paths(self.graph, source=n1, target=n2,weight='weight', cutoff=self.diameter*2)]
    #                 [self.allPaths[str(n1)+':'+str(n2)].append(p) for p in nx.shortes_simple_paths(self.graph, source=n1, target=n2,weight='weight', cutoff=self.diameter*2)]

    #                 # We take all the paths from n1 to n2 and we order them according to the path length
    #                 # sorted() ordena los paths de menor a mayor numero de
    #                 # saltos y los que tienen los mismos saltos te los ordena por indice
    #                 self.allPaths[str(n1)+':'+str(n2)] = sorted(self.allPaths[str(n1)+':'+str(n2)], key=lambda item: (len(item), item))
    #                 path = 0
    #                 while path < self.K and path < len(self.allPaths[str(n1)+':'+str(n2)]):
    #                     currentPath = self.allPaths[str(n1)+':'+str(n2)][path]
    #                     i = 0
    #                     j = 1

    #                     # Iterate over pairs of nodes and allocate linkDemand
    #                     while (j < len(currentPath)):
    #                         self.graph.get_edge_data(currentPath[i], currentPath[j])[0]['numsp'] = \
    #                             self.graph.get_edge_data(currentPath[i], currentPath[j])[0]['numsp'] + 1
    #                         i = i + 1
    #                         j = j + 1

    #                     path = path + 1

    #                 # Remove paths not needed
    #                 del self.allPaths[str(n1)+':'+str(n2)][path:len(self.allPaths[str(n1)+':'+str(n2)])]
    #                 gc.collect()

    def load_k_paths_from_file(self, k_paths_file, dataset_folder_name, one_based=True, make_directed=True):
        """
        Load precomputed k paths from json into self.allPaths.

        Expected json format:
        { "1": { "2": [ [1,7,2], [1,16,4,2], ... ], ... }, ... }
        or any src/dst keys as strings.

        After loading, self.allPaths will be:
        { "0:1": [ [0,6,1], [0,15,3,1], ... ], ... }  (0-based inside Env16)
        """
        with open(k_paths_file, "r") as f:
            raw = json.load(f)

        allPaths = {}

        for s_str, dst_dict in raw.items():
            for d_str, path_list in dst_dict.items():
                # src/dst id convert
                s = int(s_str) - 1 if one_based else int(s_str)
                d = int(d_str) - 1 if one_based else int(d_str)

                fixed_paths = []
                for p in path_list:
                    # node ids convert
                    pp = [(int(x) - 1) if one_based else int(x) for x in p]

                    # optional sanity: ensure starts/ends match
                    # if pp[0] != s or pp[-1] != d: continue

                    # If Env16 graph is directed, ensure every hop exists in that direction.
                    # If your json was computed on undirected graph, you can "directify" by
                    # checking both directions and choosing the correct directed hop.
                    if make_directed:
                        ok = True
                        for i in range(len(pp) - 1):
                            u, v = pp[i], pp[i + 1]
                            if not (u in self.graph and v in self.graph[u]):
                                ok = False
                                break
                        if not ok:
                            continue

                    fixed_paths.append(pp)

                if len(fixed_paths) == 0:
                    continue

                # pad / truncate to K
                if len(fixed_paths) >= self.K:
                    fixed_paths = fixed_paths[: self.K]
                else:
                    last = fixed_paths[-1]
                    while len(fixed_paths) < self.K:
                        fixed_paths.append(last)

                allPaths[f"{s}:{d}"] = fixed_paths

        self.allPaths = allPaths
        print(f"[load_k_paths_from_file] loaded {len(self.allPaths)} pairs from {k_paths_file}")
        # 拿來人工檢查看有沒有對齊的而已
        out_file = os.path.join(dataset_folder_name, "k_shortest_paths.json")
        try:
            with open(out_file, "w") as f:
                json.dump(self.allPaths, f, indent=2)
            print(f"[k_shortest_path] saved to {out_file}")
        except PermissionError:
            print(f"[k_shortest_path] skip save (permission denied): {out_file}")

    # def k_shortest_path(self, dataset_folder_name):
        # D = nx.DiGraph()
        # for u, v in self.graph.edges():
        #     if not D.has_edge(u, v):
        #         D.add_edge(u, v, weight=self.graph[u][v][0]['weight'])

        # self.allPaths = {}

        # for n1 in range(self.numNodes):
        #     for n2 in range(self.numNodes):
        #         if n1 == n2:
        #             continue

        #         # 產生「依 hop 遞增」的 simple paths
        #         gen = nx.shortest_simple_paths(D, n1, n2,weight='weight')

        #         paths = []
        #         for p in gen:
        #             paths.append(p)
        #             if len(paths) >= self.K:
        #                 break

        #         if len(paths) == 0:
        #             raise RuntimeError(f"No path between {n1} and {n2}")

        #         # hop 數優先，其次 lexicographical order
        #         # paths = sorted(paths, key=lambda p: (len(p), p))

        #         # 裁切 / 補齊
        #         if len(paths) >= self.K:
        #             paths = paths[:self.K]
        #         else:
        #             last = paths[-1]
        #             while len(paths) < self.K:
        #                 paths.append(last)

        #         self.allPaths[f"{n1}:{n2}"] = paths


        # # 拿來人工檢查看有沒有對齊的而已
        # out_file = os.path.join(dataset_folder_name, "k_shortest_paths.json")
        # with open(out_file, "w") as f:
        #     json.dump(self.allPaths, f, indent=2)

        # print(f"[k_shortest_path] saved to {out_file}")
        # gc.collect()

    def decrease_links_utilization_sp(self, src, dst, init_source, final_destination):
        # In this function we desallocate the bandwidth by segments. This funcion is used when we want
        # to desallocate from a src to a middlepoint and then from middlepoint to a dst using the sp

        # We obtain the demand from the original source,destination pair
        # bw_allocated = self.TM[init_source][final_destination]
        # currentPath = self.shortest_paths[src,dst]

        # srcdst = str(init_source)+':'+str(final_destination)
        # if self.use_K_path and srcdst in self.sp_pathk:
        #     currentPath = self.allPaths[srcdst][self.sp_pathk[srcdst]]

        # i = 0
        # j = 1
        # while (j < len(currentPath)):
        #     firstNode = currentPath[i]
        #     secondNode = currentPath[j]

        #     self.graph[firstNode][secondNode][0]['utilization'] -= bw_allocated 
        #     if str(init_source)+':'+str(final_destination) in self.graph[firstNode][secondNode][0]['crossing_paths']:
        #         del self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)]
        #     self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
        #     i = i + 1
        #     j = j + 1
        bw_allocated = self.TM[init_source][final_destination]
        srcdst = str(init_source)+':'+str(final_destination)

        if self.use_K_path:
            # reset / 初始化階段還沒 sp_pathk → 一律用 k=0
            if srcdst in self.sp_pathk:
                kp = self.sp_pathk[srcdst]
            else:
                kp = 0
            currentPath = self.allPaths[srcdst][kp]
        else:
            currentPath = self.shortest_paths[src, dst]

        i = 0
        j = 1
        while j < len(currentPath):
            firstNode = currentPath[i]
            secondNode = currentPath[j]

            self.graph[firstNode][secondNode][0]['utilization'] -= bw_allocated
            self.graph[firstNode][secondNode][0]['crossing_paths'].pop(
                str(init_source)+':'+str(final_destination), None
            )
            self.edge_state[
                self.edgesDict[str(firstNode)+':'+str(secondNode)]
            ][0] = self.graph[firstNode][secondNode][0]['utilization']

            i += 1
            j += 1

    def _get_top_k_critical_flows(self, list_ids):
        self.list_eligible_demands.clear()
        for linkId in list_ids:
            i = linkId[1]
            j = linkId[2]
            for demand, value in self.graph[i][j][0]['crossing_paths'].items():
                src, dst = int(demand.split(':')[0]), int(demand.split(':')[1])
                if (src, dst, self.TM[src,dst]) not in self.list_eligible_demands:  
                    self.list_eligible_demands.append((src, dst, self.TM[src,dst]))

        self.list_eligible_demands = sorted(self.list_eligible_demands, key=lambda tup: tup[2], reverse=True)
        if len(self.list_eligible_demands)>int(np.ceil(self.numNodes*(self.numNodes-1)*self.percentage_demands)):
            self.list_eligible_demands = self.list_eligible_demands[:int(np.ceil(self.numNodes*(self.numNodes-1)*self.percentage_demands))]

    def _generate_tm(self, tm_id):
        # 1. 抓TM 出來
        # 2. 抓出15% eligible_demand

        # Sample a file randomly to initialize the tm
        graph_file = self.dataset_folder_name+"/"+self.graph_topology_name+".graph"
        # This 'results_file' file is ignored!
        results_file = self.dataset_folder_name+"/res_"+self.graph_topology_name+"_"+str(tm_id)
        tm_file = self.dataset_folder_name+"/TM/"+self.graph_topology_name+'.'+str(tm_id)+".demands"
        
        self.defoDatasetAPI = defoResults.Defo_results(graph_file,results_file)
        self.links_bw = self.defoDatasetAPI.links_bw
        self.MP_matrix = self.defoDatasetAPI.MP_matrix
        self.TM = self.defoDatasetAPI._get_traffic_matrix(tm_file)

        self.iter_list_elig_demn = 0
        self.list_eligible_demands.clear()

        # 沒用到
        min_links_bw = 1000000.0
        for src in range (0,self.numNodes):
            for dst in range (0,self.numNodes):
                if src!=dst:
                    self.list_eligible_demands.append((src, dst, self.TM[src,dst]))
                    # If we have a link between src and dst
                    if src in self.graph and dst in self.graph[src]:
                        # Store the link with minimum bw
                        if self.links_bw[src][dst]<min_links_bw:
                            min_links_bw = self.links_bw[src][dst]
                        
                        # Clear the link utilization and crossing paths for each link
                        self.graph[src][dst][0]['utilization'] = 0.0
                        self.graph[src][dst][0]['crossing_paths'].clear()
        
        # If we want to take the X% random demands
        if self.shufle_demands:
            random.shuffle(self.list_eligible_demands)
            self.list_eligible_demands = self.list_eligible_demands[:int(np.ceil(len(self.list_eligible_demands)*self.percentage_demands))]
        elif not self.top_K_critical_demands:
            # If we want to take the x% bigger demands
            self.list_eligible_demands = sorted(self.list_eligible_demands, key=lambda tup: tup[2], reverse=True)
            self.list_eligible_demands = self.list_eligible_demands[:int(np.ceil(len(self.list_eligible_demands)*self.percentage_demands))]

    def compute_link_utilization_reset(self):
        # Allocate for each src,dst the corresponding traffic on the corresponding SP
        for src in range (0,self.numNodes):
            for dst in range (0,self.numNodes):
                if src!=dst:
                    self.allocate_to_destination_sp(src, dst, src, dst)
    
    def _obtain_path_more_bandwidth_rand_link(self):
        # Obtain path with largest bandwidth from the edge with highest utilization
        # We sort the paths by bandwidth and pick random from the top 4
        sorted_dict = list((k, v) for k, v in sorted(self.graph[self.edgeMaxUti[0]][self.edgeMaxUti[1]][0]['crossing_paths'].items(), key=lambda item: item[1], reverse=True))
        path = random.randint(0, 1)
        # In case there is only one bandwidth
        if path>=len(sorted_dict):
            path = 0
        srcPath = int(sorted_dict[path][0].split(':')[0])
        dstPath = int(sorted_dict[path][0].split(':')[1])
        self.patMaxBandwth = (srcPath, dstPath, self.TM[srcPath][dstPath])
    
    def _obtain_path_from_set_rand(self):
        len_demans = len(self.list_eligible_demands)-1
        path = random.randint(0, len_demans)
        srcPath = int(self.list_eligible_demands[path][0])
        dstPath = int(self.list_eligible_demands[path][1])
        self.patMaxBandwth = (srcPath, dstPath, int(self.list_eligible_demands[path][2]))
    
    def _obtain_demand(self):
        src = self.list_eligible_demands[self.iter_list_elig_demn][0]
        dst = self.list_eligible_demands[self.iter_list_elig_demn][1]
        bw = self.list_eligible_demands[self.iter_list_elig_demn][2]
        self.patMaxBandwth = (src, dst, int(bw))
        self.iter_list_elig_demn += 1
    
    def get_value(self, source, destination, action):
        # We get the K-middlepoints between source-destination
        middlePointList = self.src_dst_k_middlepoints[str(source) +':'+ str(destination)]
        middlePoint = middlePointList[action]

        # First we allocate until the middlepoint
        self.allocate_to_destination_sp(source, middlePoint, source, destination)
        # If we allocated to a middlepoint that is not the final destination
        if middlePoint!=destination:
            # Then we allocate from the middlepoint to the destination
            self.allocate_to_destination_sp(middlePoint, destination, source, destination)
            # We store that the pair source,destination has a middlepoint
            self.sp_middlepoints[str(source)+':'+str(destination)] = middlePoint
        
        currentValue = -1000000
        # Get the maximum loaded link and it's value after allocating to the corresponding middlepoint
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                link_capacity = self.links_bw[i][j]
                if self.edge_state[position][0]/link_capacity>currentValue:
                    currentValue = self.edge_state[position][0]/link_capacity
        
        # Dissolve allocation step so that later we can try another action
        # Remove bandwidth allocated until the middlepoint and then from the middlepoint on
        if str(source)+':'+str(destination) in self.sp_middlepoints:
            middlepoint = self.sp_middlepoints[str(source)+':'+str(destination)]
            self.decrease_links_utilization_sp(source, middlepoint, source, destination)
            self.decrease_links_utilization_sp(middlepoint, destination, source, destination)
            del self.sp_middlepoints[str(source)+':'+str(destination)] 
        else: # Remove the bandwidth allocated from the src to the destination
            self.decrease_links_utilization_sp(source, destination, source, destination)
        
        return -currentValue  

    def _obtain_demand_hill_climbing(self):
        dem_iter = 0
        nextVal = -1000000
        self.next_state = None
        # Iterate for each demand possible
        for source in range(self.numNodes):
            for dest in range(self.numNodes):
                if source!=dest:
                    for action in range(len(self.src_dst_k_middlepoints[str(source)+':'+str(dest)])):
                        middlepoint = -1
                        # First we need to desallocate the current demand before we explore all it's possible actions
                        # Check if there is a middlepoint to desallocate from src-middlepoint-dst
                        if str(source)+':'+str(dest) in self.sp_middlepoints:
                            middlepoint = self.sp_middlepoints[str(source)+':'+str(dest)]
                            self.decrease_links_utilization_sp(source, middlepoint, source, dest)
                            self.decrease_links_utilization_sp(middlepoint, dest, source, dest)
                            del self.sp_middlepoints[str(source)+':'+str(dest)] 
                        else: # Remove the bandwidth allocated from the src to the destination
                            self.decrease_links_utilization_sp(source, dest, source, dest)

                        evalState = self.get_value(source, dest, action)
                        if evalState > nextVal:
                            nextVal = evalState
                            self.next_state = (action, source, dest)
                        
                        # Allocate back the demand whose actions we explored
                        # If the current demand had a middlepoint, we allocate src-middlepoint-dst
                        if middlepoint>=0:
                            # First we allocate until the middlepoint
                            self.allocate_to_destination_sp(source, middlepoint, source, dest)
                            # Then we allocate from the middlepoint to the destination
                            self.allocate_to_destination_sp(middlepoint, dest, source, dest)
                            # We store that the pair source,destination has a middlepoint
                            self.sp_middlepoints[str(source)+':'+str(dest)] = middlepoint
                        else:
                            # Then we allocate from the middlepoint to the destination
                            self.allocate_to_destination_sp(source, dest, source, dest)
        self.patMaxBandwth = (self.next_state[1], self.next_state[2], self.TM[self.next_state[1]][self.next_state[2]])

    def compute_middlepoint_set_random(self):
        # We choose the K-middlepoints for each src-dst randomly
        self.src_dst_k_middlepoints = dict()
        # Iterate over all node1,node2 pairs from the graph
        for n1 in range (0,self.numNodes):
            for n2 in range (0,self.numNodes):
                if (n1 != n2):
                    num_middlepoints = 0
                    self.src_dst_k_middlepoints[str(n1)+':'+str(n2)] = list()
                    # We add the destination as a candidate middlepoint (in case we have direct connection)
                    self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(n2)
                    num_middlepoints += 1
                    while num_middlepoints<self.K:
                        middlpt = np.random.randint(0, self.numNodes)
                        while middlpt==n1 or middlpt==n2 or middlpt in self.src_dst_k_middlepoints[str(n1)+':'+str(n2)]:
                            middlpt = np.random.randint(0, self.numNodes)
                        self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(middlpt)
                        num_middlepoints += 1         

    def mark_edges(self, action_flags, src, dst, init_source, final_destination):
        currentPath = self.shortest_paths[src,dst]
        
        i = 0
        j = 1

        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]

            action_flags[self.edgesDict[str(firstNode)+':'+str(secondNode)]] += 1.0
            i = i + 1
            j = j + 1
    
    def mark_action_to_edges(self, first_node, init_source, final_destination): 
        # In this function we mark for each link which is the bw that it will allocate. This we will
        # use to avoid repeated actions
        action_flags = np.zeros(self.numEdges)
        
        # Mark until first_node
        self.mark_edges(action_flags, init_source, first_node, init_source, final_destination)

        # If the first node is a middlepoint
        if first_node!=final_destination:
            self.mark_edges(action_flags, first_node, final_destination, init_source, final_destination)
        
        return action_flags

    # def compute_middlepoint_set_remove_rep_actions_no_loop(self):
    #     # In this function we compute the middlepoint set but we don't take into account the middlepoints whose 
    #     # actions are repeated and neither those middlepoints whose SPs pass over the DST or SRC nodes
        
    #     # Compute SPs for each src,dst pair
    #     self.compute_SPs()

    #     # We compute the middlepoint set for each src,dst pair and we don't consider repeated actions
    #     self.src_dst_k_middlepoints = dict()
    #     # Iterate over all node1,node2 pairs from the graph
    #     for n1 in range (0,self.numNodes):
    #         for n2 in range (0,self.numNodes):
    #             if (n1 != n2):
    #                 self.src_dst_k_middlepoints[str(n1)+':'+str(n2)] = list()
    #                 repeated_actions = list()
    #                 for midd in range (0,self.K):
    #                     # If the middlepoint is not the source node
    #                     if midd!=n1:
    #                         action_flags = self.mark_action_to_edges(midd, n1, n2)
    #                         # If we allocated to a middlepoint that is not the final destination
    #                         if midd!=n2:
    #                             # If the repeated_actions list is empty we make the following verifications
    #                             if len(repeated_actions) == 0:
    #                                 #print(" A...... ")

    #                                 path1 = self.shortest_paths[n1, midd]
    #                                 path2 = self.shortest_paths[midd, n2]

    #                                 # Check that the dst node is not in the SP to avoid loops!
    #                                 currentPath = path1[:len(path1)-1]+path2
    #                                 dst_counter = 0
    #                                 for node in currentPath:
    #                                     if node==n2 or node==n1:
    #                                         dst_counter += 1
    #                                 # If there is only one dst node
    #                                 if dst_counter==2:
    #                                     repeated_actions.append(action_flags)
    #                                     self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(midd)
    #                             else:
    #                                 #print(" B...... ")
    #                                 repeatedAction = False
    #                                 # Compare the current action with the previous ones
    #                                 for previous_actions in repeated_actions:
    #                                     subtraction = np.absolute(np.subtract(action_flags,previous_actions))
    #                                     if np.sum(subtraction)==0.0:
    #                                         repeatedAction = True
    #                                         break
    #                                 # If we didn't find any identical action, we make the following verifications
    #                                 if not repeatedAction:                                        
    #                                     path1 = self.shortest_paths[n1, midd]
    #                                     path2 = self.shortest_paths[midd, n2]
    #                                     # Check that the dst node is not in the SP to avoid loops!
    #                                     currentPath = path1[:len(path1)-1]+path2
    #                                     dst_counter = 0
    #                                     for node in currentPath:
    #                                         if node==n2 or node==n1:
    #                                             dst_counter += 1
    #                                     # If there is only one dst node
    #                                     if dst_counter==2:
    #                                         self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(midd)
    #                                         repeated_actions.append(action_flags)

    #                         else: 
    #                             # If it's the first action we add it to the repeated actions list
    #                             if len(repeated_actions) == 0:
    #                                 #print(" C...... ")
    #                                 self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(midd)
    #                                 repeated_actions.append(action_flags)
    #                             else:
    #                                 #print(" D...... ")
    #                                 repeatedAction = False
    #                                 # Compare the current action with the previous ones
    #                                 for previous_actions in repeated_actions:
    #                                     subtraction = np.absolute(np.subtract(action_flags,previous_actions))
    #                                     if np.sum(subtraction)==0.0:
    #                                         repeatedAction = True
    #                                         break
                                    
    #                                 # If we didn't find any identical action, we add the middlepoint to the set
    #                                 if not repeatedAction:
    #                                     self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(midd)
    #                                     repeated_actions.append(action_flags)

    # def compute_SPs(self):
        # diameter = nx.diameter(self.graph)
        # self.shortest_paths = np.zeros((self.numNodes,self.numNodes),dtype=object)
        
        # allPaths = dict()
        # sp_path = self.dataset_folder_name+"/shortest_paths.json"

        # if not os.path.isfile(sp_path):
        #     for n1 in range (0,self.numNodes):
        #         for n2 in range (0,self.numNodes):
        #             if (n1 != n2):
        #                 allPaths[str(n1)+':'+str(n2)] = []
        #                 # First we compute the shortest paths taking into account the diameter
        #                 [allPaths[str(n1)+':'+str(n2)].append(p) for p in nx.all_simple_paths(self.graph, source=n1, target=n2,weight='weight', cutoff=diameter*2)]                    # We take all the paths from n1 to n2 and we order them according to the path length
        #                 # sorted() ordena los paths de menor a mayor numero de
        #                 # saltos y los que tienen los mismos saltos te los ordena por indice
        #                 aux_sorted_paths = sorted(allPaths[str(n1)+':'+str(n2)], key=lambda item: (len(item), item))                    # self.shortest_paths[n1,n2] = nx.shortest_path(self.graph, n1, n2,weight='weight')
        #                 allPaths[str(n1)+':'+str(n2)] = aux_sorted_paths[0]
        
        #     with open(sp_path, 'w') as fp:
        #         json.dump(allPaths, fp)
        # else:
        #     allPaths = json.load(open(sp_path))

        # for n1 in range (0,self.numNodes):
        #     for n2 in range (0,self.numNodes):
        #         if (n1 != n2):
        #             self.shortest_paths[n1,n2] = allPaths[str(n1)+':'+str(n2)]
    
    def _first_second(self):
        # Link (1, 2) recibe trafico de los links que inyectan en el nodo 1
        # un link que apunta a un nodo envía mensajes a todos los links que salen de ese nodo
        first = list()
        second = list()

        for i in self.graph:
            for j in self.graph[i]:
                neighbour_edges = self.graph.edges(j)
                # Take output links of node 'j'

                for m, n in neighbour_edges:
                    if ((i != m or j != n) and (i != n or j != m)):
                        first.append(self.edgesDict[str(i) +':'+ str(j)])
                        second.append(self.edgesDict[str(m) +':'+ str(n)])

        self.first = first
        self.second = second

    def generate_environment(self, dataset_folder_name, graph_topology_name, EPISODE_LENGTH, K, X):
        self.episode_length = EPISODE_LENGTH
        self.graph_topology_name = graph_topology_name
        self.dataset_folder_name = dataset_folder_name
        self.list_eligible_demands = list()
        self.iter_list_elig_demn = 0
        self.percentage_demands = X

        self.maxCapacity = 0 # We take the maximum capacity to normalize

        # Just select some random file, the only thing we need is the links features and the topology
        graph_file = self.dataset_folder_name+"/"+self.graph_topology_name+".graph"
        # This 'results_file' file is ignored!
        results_file = self.dataset_folder_name+"/res_"+self.graph_topology_name+"_0"
        tm_file = self.dataset_folder_name+"/TM/"+self.graph_topology_name+".0.demands"
        self.defoDatasetAPI = defoResults.Defo_results(graph_file,results_file)
        
        self.graph = self.defoDatasetAPI.Gbase
        self.add_features_to_edges()
        self.numNodes = len(self.graph.nodes())
        self.numEdges = len(self.graph.edges())
        btwns = nx.edge_betweenness_centrality(self.graph)

        self.K = K
        if not self.use_K_path and self.K>self.numNodes:
            self.K = self.numNodes

        self.edge_state = np.zeros((self.numEdges, 3))
        self.betweenness_centrality = np.zeros(self.numEdges) # Used in the fully connected
        self.shortest_paths = np.zeros((self.numNodes,self.numNodes),dtype="object")

        position = 0
        for i in self.graph:
            for j in self.graph[i]:
                self.edgesDict[str(i)+':'+str(j)] = position
                self.graph[i][j][0]['capacity'] = self.defoDatasetAPI.links_bw[i][j]
                self.graph[i][j][0]['weight'] = self.defoDatasetAPI.links_weight[i][j]
                if self.graph[i][j][0]['capacity']>self.maxCapacity:
                    self.maxCapacity = self.graph[i][j][0]['capacity']
                self.edge_state[position][1] = self.graph[i][j][0]['capacity']
                # self.betweenness_centrality[position] = btwns[i,j]
                self.betweenness_centrality[position] = btwns.get((i,j), 0.0)
                self.graph[i][j][0]['utilization'] = 0.0
                self.graph[i][j][0]['crossing_paths'].clear()
                position += 1

        # Fluid queue state (bits) per directed link — persists across steps, resets per TM
        self.queue_bits = {}
        # Previous-step queue delay per directed link (seconds) — used by delay_loss LU estimator
        self.prev_link_delay = {}
        for i in self.graph:
            for j in self.graph[i]:
                self.queue_bits[(i, j)] = 0.0
                self.prev_link_delay[(i, j)] = 0.0

        # Build undirected edge ordering from edgesDict
        self.undirected_edges = []
        seen = set()

        for key, idx in self.edgesDict.items():
            u_str, v_str = key.split(':')
            u, v = int(u_str), int(v_str)
            a, b = (u, v) if u < v else (v, u)

            if (a, b) in seen:
                continue
            # 只收那些確實有雙向邊的（Env16 是 bidirectional）
            if (f"{a}:{b}" in self.edgesDict) and (f"{b}:{a}" in self.edgesDict):
                seen.add((a, b))
                self.undirected_edges.append((a, b))
        # optional: 也可以做 dict 方便查
        self.undirected_edgesDict = {e: k for k, e in enumerate(self.undirected_edges)}


        self._first_second()
        self.firstTrueSize = len(self.first)

        # 在 model 的 input_transform 裡面會被用到
        # env.link_capacity_feature
        self.link_capacity_feature = np.divide(self.edge_state[:,1], self.maxCapacity)

        # We create the list of nodes ids to pick randomly from them
        self.nodes = list(range(0,self.numNodes))
        
        self.build_pair_index()

        if self.use_K_path:
            # 不要再在這裡生成了 直接去抓dataset生成好的轉成0-based的k-shortest-paths
            # sp 也會一併搞定
            # self.compute_SPs()
            # self.k_shortest_path(dataset_folder_name)
            k_paths_file = getattr(self, "k_paths_file_override", None)
            if k_paths_file is None:
                k_paths_file = os.path.join(
                    PROJECT_ROOT,
                    "dataset/geant_traffic/k_paths.json"
                )
            self.load_k_paths_from_file(
                k_paths_file,
                dataset_folder_name,
                one_based=True,        # 你的檔是 1-based
                make_directed=True     # Env16 是 directed 展開（雙向)
            )
        else:
            self.compute_middlepoint_set_remove_rep_actions_no_loop()        

    def step(self, action, demand, source, destination):
        # Action is the middlepoint. Careful because it can also be action==destination if src,dst are connected directly by an edge
        self.episode_over = False
        self.reward = 0

        kp = None
        if self.use_K_path:
            kp = action
            self.sp_pathk[str(source)+':'+str(destination)] = kp
            middlePoint = destination
        else:
            # We get the K-middlepoints between source-destination
            middlePointList = self.src_dst_k_middlepoints[str(source) + ':' + str(destination)]
            middlePoint = middlePointList[action]

        # First we allocate until the middlepoint
        self.allocate_to_destination_sp(source, middlePoint, source, destination, kp)
        # If we allocated to a middlepoint that is not the final destination
        if middlePoint!=destination:
            # Then we allocate from the middlepoint to the destination
            self.allocate_to_destination_sp(middlePoint, destination, source, destination)
            # We store that the pair source,destination has a middlepoint
            self.sp_middlepoints[str(source)+':'+str(destination)] = middlePoint
        
      
        if self.use_K_path:
            self.sp_middlepoints_step = dict(self.sp_pathk)
        else:
            self.sp_middlepoints_step = dict(self.sp_middlepoints)

        
        # Find new maximum and minimum utilization link
        # 這是整個TM 都使用 SP排出來的　edgeMaxUti (normalized)
        old_Utilization = self.edgeMaxUti[2]
        old_Utilization_std = self.edgeStdUti
        self.edgeMaxUti = (0, 0, 0)
        self.edgeStdUti = 0
        uti_list = []
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                link_capacity = self.links_bw[i][j]
                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity
                uti_list.append(norm_edge_state_capacity)
                if norm_edge_state_capacity>self.edgeMaxUti[2]:
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity)
         
        self.currentVal = -self.edgeMaxUti[2]

        self.edgeStdUti = np.std(uti_list)
        self.reward = np.around(10*(0.6*(old_Utilization-self.edgeMaxUti[2]) + 0.4*(old_Utilization_std-self.edgeStdUti)), 3)
        #self.reward = np.around((10*(old_Utilization-self.edgeMaxUti[2])-np.std(uti_list)/10), 3)

        # If we didn't iterate over all demands 
        if self.iter_list_elig_demn<len(self.list_eligible_demands):
            self._obtain_demand()
        else:
            src = 1
            dst = 2
            bw = self.TM[src][dst]
            self.patMaxBandwth = (src, dst, int(bw))
            self.episode_over = True

        # Remove bandwidth allocated until the middlepoint and then from the middlepoint on
        if str(self.patMaxBandwth[0])+':'+str(self.patMaxBandwth[1]) in self.sp_middlepoints:
            middlepoint = self.sp_middlepoints[str(self.patMaxBandwth[0])+':'+str(self.patMaxBandwth[1])]
            self.decrease_links_utilization_sp(self.patMaxBandwth[0], middlepoint, self.patMaxBandwth[0], self.patMaxBandwth[1])
            self.decrease_links_utilization_sp(middlepoint, self.patMaxBandwth[1], self.patMaxBandwth[0], self.patMaxBandwth[1])
            del self.sp_middlepoints[str(self.patMaxBandwth[0])+':'+str(self.patMaxBandwth[1])] 
        else: # Remove the bandwidth allocated from the src to the destination
            self.decrease_links_utilization_sp(self.patMaxBandwth[0], self.patMaxBandwth[1], self.patMaxBandwth[0], self.patMaxBandwth[1])
        
        # We desmark the bw_allocated
        self.edge_state[:,2] = 0

        return self.reward, self.episode_over, 10*(old_Utilization-self.edgeMaxUti[2]), self.TM[self.patMaxBandwth[0]][self.patMaxBandwth[1]], self.patMaxBandwth[0], self.patMaxBandwth[1], self.edgeMaxUti, 10*(old_Utilization_std-self.edgeStdUti), np.std(self.edge_state[:,0])
        '''
        return (
            reward,
            episode_over,
            10*(old_MLU - new_MLU),            # MLU 改善項
            TM[next_src][next_dst],            # 下一筆 demand 的大小
            next_src, next_dst,                # 下一筆 demand 的 src/dst
            edgeMaxUti,                        # 當前最大利用率的那條 edge (i,j,val)
            10*(old_std - new_std),            # std 改善項
            std(edge_state[:,0])               # raw util std（沒除 capacity）
        )
        '''

    def reset(self, tm_id, best_routing=None):
        """
        Reset environment and setup for new episode. 
        Generate new TM but load the same routing. We remove the path with more bandwidth
        from the link with more utilization to later allocate it on a new path in the act().
        """
        self.current_tm_id = int(tm_id)
        self._generate_tm(tm_id)

        self.sp_middlepoints = dict()
        self.sp_pathk = dict()

        # For each link we store the total sum of bandwidths of the paths crossing each link without middlepoints
        self.compute_link_utilization_reset()

        if best_routing is not None:
            for key, kp in best_routing.items():
                source = int(key.split(':')[0])
                dest = int(key.split(':')[1])
                self.decrease_links_utilization_sp(source, dest, source, dest)
                self.allocate_to_destination_sp(source, dest, source, dest, kp)
            self.sp_pathk = best_routing

        # We iterate over all links in an ordered fashion and store the features to edge_state
        self.edgeMaxUti = (0, 0, 0)
        # This list is used to obtain the top K flows from the critical links
        list_link_uti_id = list()
        uti_list = []
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                self.edge_state[position][1] = self.graph[i][j][0]['capacity']
                link_capacity = self.links_bw[i][j]
                # We store the link utilization and the corresponding edge
                list_link_uti_id.append((self.edge_state[position][0], i, j))
                
                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity
                uti_list.append(norm_edge_state_capacity)
                if norm_edge_state_capacity>self.edgeMaxUti[2]:
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity)
        self.edgeStdUti = np.std(uti_list)
        
        if self.top_K_critical_demands:
            list_link_uti_id = sorted(list_link_uti_id, key=lambda tup: tup[0], reverse=True)[:self.num_critical_links]
            self._get_top_k_critical_flows(list_link_uti_id)

        self.currentVal = -self.edgeMaxUti[2]
        self.initial_maxLinkUti = -self.edgeMaxUti[2]
        # From the link with more utilization, we obtain a random path of the 5 with more bandwidth
        #self._obtain_path_more_bandwidth_rand_link()
        #self._obtain_path_from_set_rand()
        #self._obtain_demand_hill_climbing()
        self._obtain_demand()

        # Remove bandwidth allocated for the path with more bandwidth from the link with more utilization
        self.decrease_links_utilization_sp(self.patMaxBandwth[0], self.patMaxBandwth[1], self.patMaxBandwth[0], self.patMaxBandwth[1])

        # We desmark the bw_allocated
        self.edge_state[:,2] = 0

        return self.TM[self.patMaxBandwth[0]][self.patMaxBandwth[1]], self.patMaxBandwth[0], self.patMaxBandwth[1]
    
    def allocate_to_destination_sp(self, src, dst, init_source, final_destination, kp=None):
        # In this function we allocated the bandwidth by segments. This funcion is used when we want
        # to allocate from a src to a middlepoint and then from middlepoint to a dst using the sp
        # bw_allocate = self.TM[init_source][final_destination]
        # currentPath = self.shortest_paths[src,dst]

        # if kp != None:
        #     currentPath = self.allPaths[str(init_source)+':'+str(final_destination)][kp]
        
        bw_allocate = self.TM[init_source][final_destination]
        if self.use_K_path:
            # reset / 初始化階段一定要走 k=0
            if kp is None:
                kp = 0
            currentPath = self.allPaths[str(init_source)+':'+str(final_destination)][kp]
        else:
            currentPath = self.shortest_paths[src, dst]

        i = 0
        j = 1

        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]

            self.graph[firstNode][secondNode][0]['utilization'] += bw_allocate  
            self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)] = bw_allocate
            self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
            i = i + 1
            j = j + 1
    
    def mark_action_sp(self, src, dst, init_source, final_destination): 
        # In this function we mark the action in the corresponding edges of the SP between src,dst
        bw_allocate = self.TM[init_source][final_destination]
        currentPath = self.shortest_paths[src,dst]
        
        i = 0
        j = 1

        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]

            self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][2] = bw_allocate/self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][1]
            i = i + 1
            j = j + 1

    # ==================================================
    # == General utils for TM generation and cleaning ==
    # ==================================================

    def reset_queues(self):
        """Reset all link queues to empty. Called when TM changes (new episode)."""
        if hasattr(self, 'queue_bits'):
            for key in self.queue_bits:
                self.queue_bits[key] = 0.0
            for i in self.graph:
                for j in self.graph[i]:
                    self.graph[i][j][0]['delay'] = 0.0
                    self.graph[i][j][0]['pkloss'] = 0.0
                    self.graph[i][j][0]['queue_pkts'] = 0
        if hasattr(self, 'prev_link_delay'):
            for key in self.prev_link_delay:
                self.prev_link_delay[key] = 0.0

    def _update_queues(self, config=None):
        """
        Fluid queue model: advance each directed link's queue by 1 step.

        For each link, computes:
          arrival (bits)  = utilization (Kbps) × 1000 × step_duration
          service (bits)  = capacity    (Kbps) × 1000 × step_duration
          queue += arrival; served = min(queue, service); queue -= served
          overflow = max(0, queue - max_queue_bits) → tail-drop

        Writes to graph edges: delay (sec), pkloss (ratio 0~1), queue_pkts (int).
        These are read by _link_arrays_from_graph() / compute_path_metrics_fast().

        Config keys (from kpath_cfg or a passed config):
          queue_step_duration:  seconds per step  (default: MONITOR_PERIOD from setting.py)
          queue_max_pkts:       max buffer in pkts (default: 1000, matching netem limit=1000)
          queue_avg_pkt_bytes:  avg packet size    (default: 1488, L3 skb = IP20+UDP8+payload1460.
                                tc htb counts at L3 level; Eth header not included.
                                Verified: 1000*1488*8/1550000 = 7680ms ≈ real 7675ms)
        """
        cfg = config or getattr(self, 'kpath_cfg', {}) or {}
        step_duration    = float(cfg.get("queue_step_duration", MONITOR_PERIOD))
        max_queue_pkts   = float(cfg.get("queue_max_pkts", 1000))
        avg_pkt_bytes    = float(cfg.get("queue_avg_pkt_bytes", 1488))
        # Overhead: iperf3 -b sets UDP payload rate (1460B per datagram), but tc htb
        # rate-limits at L3 (IP+UDP+payload = 1488B).  Additional sources: LLDP/ARP/OF
        # background traffic + tc byte accounting.  Empirically calibrated from real
        # fill rate (73 pkts/step on 1.55Mbps link, TM03 OSPF session 20260413_130143).
        overhead_factor  = float(cfg.get("queue_overhead_factor", 1.032))
        max_queue_bits   = max_queue_pkts * avg_pkt_bytes * 8

        for i in self.graph:
            for j in self.graph[i]:
                edge = self.graph[i][j][0]
                arrival_kbps  = float(edge.get('utilization', 0.0))
                capacity_kbps = float(edge.get('capacity', self.links_bw[i][j]))

                arrival_bits = arrival_kbps * overhead_factor * 1000.0 * step_duration
                service_bits = capacity_kbps * 1000.0 * step_duration

                # Queue update
                q = self.queue_bits.get((i, j), 0.0)
                q += arrival_bits
                served = min(q, service_bits)
                q -= served

                # Tail-drop
                if q > max_queue_bits:
                    overflow_bits = q - max_queue_bits
                    q = max_queue_bits
                else:
                    overflow_bits = 0.0

                self.queue_bits[(i, j)] = q

                # Metrics
                edge['delay']      = q / (capacity_kbps * 1000.0) if capacity_kbps > 0 else 0.0
                edge['pkloss']     = overflow_bits / arrival_bits if arrival_bits > 0 else 0.0
                edge['queue_pkts'] = int(q / (avg_pkt_bytes * 8))

    def clear_utilization(self, clear_crossing=False):
        """清空 topo utilization（不重讀 TM）"""
        for i in self.graph:
            for j in self.graph[i]:
                self.graph[i][j][0]['utilization'] = 0.0
                if clear_crossing and 'crossing_paths' in self.graph[i][j][0]:
                    self.graph[i][j][0]['crossing_paths'].clear()

        if self.edge_state is not None:
            self.edge_state[:, 0] = 0.0
            self.edge_state[:, 2] = 0.0

    def clean_and_generate_tm(self, tm_id):
        self.current_tm_id = int(tm_id)
        # 這個是給 multi agent all to all 用的
        # 跟 generate_tm 差不多, 但不用算 eligible demand 
        # Sample a file randomly to initialize the tm
        graph_file = self.dataset_folder_name+"/"+self.graph_topology_name+".graph"
        # This 'results_file' file is ignored!
        results_file = self.dataset_folder_name+"/res_"+self.graph_topology_name+"_"+str(tm_id)
        tm_file = self.dataset_folder_name+"/TM/"+self.graph_topology_name+'.'+str(int(tm_id))+".demands"

        self.defoDatasetAPI = defoResults.Defo_results(graph_file,results_file)
        self.links_bw = self.defoDatasetAPI.links_bw
        self.MP_matrix = self.defoDatasetAPI.MP_matrix
        self.TM = self.defoDatasetAPI._get_traffic_matrix(tm_file)
        # 清 topo utilization / crossing_paths
        self.clear_utilization(clear_crossing=True)
        # 新 TM = 新 episode → queue 歸零
        self.reset_queues()

    #===============================================
    #== For regular Multi Agent related functions ==
    #===============================================

    def reset_and_get_state_by_NX(self,config, masks, model_result_dir, tm_id):
        # reset and route by ma model return state
        # for multi agent 
        # allocate bw using drl_paths generated by MA models
        # 沒有 step 的概念
        # 目前有 clip mlu, util to 0
        
        self.clean_and_generate_tm(tm_id)
        self.allocate_by_ma(model_result_dir)

        # === build state aligned with get_state ===
        num_undirected = len(self.undirected_edges)
        global_state_2d = np.zeros((num_undirected, 3), dtype=float)

        mlu = 0.0

        for idx, (u, v) in enumerate(self.undirected_edges):
            util_uv = self.graph[u][v][0]['utilization']
            util_vu = self.graph[v][u][0]['utilization']
            cap = (
                self.graph[u][v][0]['capacity']
                + self.graph[v][u][0]['capacity']
            )

            # 先 clamp 每個 directed 方向，再加總
            cap_uv = self.graph[u][v][0]['capacity']
            cap_vu = self.graph[v][u][0]['capacity']
            used_bw = min(util_uv, cap_uv) + min(util_vu, cap_vu)
            cur_bwd = cap - used_bw                     # 剩餘 bandwidth
            # Aggregate directed queue metrics → undirected (max of both directions)
            delay_uv = self.graph[u][v][0].get('delay', 0.0)
            delay_vu = self.graph[v][u][0].get('delay', 0.0)
            loss_uv  = self.graph[u][v][0].get('pkloss', 0.0)
            loss_vu  = self.graph[v][u][0].get('pkloss', 0.0)
            cur_delay  = max(delay_uv, delay_vu) + 1e-6
            cur_pkloss = max(loss_uv, loss_vu)

            mlu = min(1.0, max(mlu, used_bw / cap))

            global_state_2d[idx, 0] = np.clip(cur_bwd / cap, 0.0, 1.0)   # normalize to [0,1]
            global_state_2d[idx, 1] = cur_delay / config.get("delay_norm_div", 200.0)
            global_state_2d[idx, 2] = cur_pkloss
                        
        if config.get("use_bwd_only", False):
            global_state_2d = global_state_2d[:, 0:1]   # (num_links, 1)
        
        global_state_2d_expanded = np.expand_dims(global_state_2d, axis=0)  # (1, num_links, 3)
        local_state = masks[:, :, :global_state_2d.shape[1]] * global_state_2d_expanded # 如果只用 delay mask 也跟著縮惟度

        # local_state = masks * global_state_2d_expanded
        global_state_2d = global_state_2d.flatten()

        return local_state, mlu, global_state_2d

    def allocate_by_ma(self,model_result_dir):
        with open(model_result_dir + "/drl_paths.json", "r") as f:
            k_paths_raw = json.load(f)

        k_paths = {}
        for s_str, dst_dict in k_paths_raw.items():
            s = int(s_str) - 1              # src: 1-based → 0-based
            k_paths[s] = {}

            for d_str, paths in dst_dict.items():
                d = int(d_str) - 1          # dst: 1-based → 0-based

                k_paths[s][d] = [node - 1 for node in paths[0]]
                
        
        for src in range (0,self.numNodes):
            for dst in range (0,self.numNodes):
                if src!=dst:
                    bw_allocate = self.TM[src][dst]
                    if bw_allocate <= 0:
                        continue
                    currentPath = k_paths[src][dst]

                    for i in range(len(currentPath) - 1):
                        firstNode = currentPath[i]
                        secondNode = currentPath[i + 1]

                        self.graph[firstNode][secondNode][0]['utilization'] += bw_allocate  
                        self.graph[firstNode][secondNode][0]['crossing_paths'][str(src)+':'+str(dst)] = bw_allocate
                        position = self.edgesDict[f"{firstNode}:{secondNode}"]
                        self.edge_state[position][0] = self.graph[firstNode][secondNode][0]['utilization']

        # Advance queue by 1 step
        self._update_queues()

    # ===========================
    # MaskGIT Routing API
    # ===========================
    
    #=================================================
    #== preprocess for MaskGIT ==
    #=================================================
    def kpath_init(self, config=None):
        """
        設定 MaskGIT reward 參數與 baseline completion 策略
        """
        self.kpath_cfg = config
        self.kpath_cfg.setdefault("reward_def", "new")
        self.kpath_cfg.setdefault("metric_mode", "undirected")
        # 2026-05-07 v2 lu_est: when True, link_state ch1 (clamped util_ratio)
        # is REPLACED by lu_est_norm = lu_est / clamp_max ∈ [0, 1], and link
        # cols re-ordered to static-first layout. See get_link_features
        # docstring.
        self._link_state_with_lu_est = bool(config.get("link_state_with_lu_est", False))
        # 2026-05-09 lossdelay: when True, link_state schema becomes (E, 5) S-first
        # [cap_norm, betweenness, util_ratio, delay_norm, loss_norm] — drops
        # rem_ratio (redundant with util_ratio under clamp), adds raw delay/loss
        # per-link signals. path_state_init becomes (P, 6) with ch4=sum_delay,
        # ch5=compose_loss for e2e physical aggregation.
        self._link_state_with_lossdelay = bool(config.get("link_state_with_lossdelay", False))
        # 2026-05-23 ls2ic 3-ch alignment: when True, link_state becomes (E, 3) all-
        # dynamic [bwd_ratio, delay_norm, loss_norm] (drops cap_norm + betweenness
        # static). path_state becomes (P, 3) [min_bwd, sum_delay, compose_loss]
        # (drops hop_norm + min_cap static). All 3 channels recon-supervised. Use
        # with reward_def="ls2ic" for paper-clean ls2ic-style ablation.
        self._link_state_ls2ic_3ch = bool(config.get("link_state_ls2ic_3ch", False))
        # Three layouts mutually exclusive
        n_active = sum([self._link_state_with_lu_est,
                        self._link_state_with_lossdelay,
                        self._link_state_ls2ic_3ch])
        assert n_active <= 1, \
            "link_state_with_lu_est / link_state_with_lossdelay / link_state_ls2ic_3ch are mutually exclusive"
        # Cache topology-wide max queueing delay (seconds) for delay normalization.
        # D_max_topo = (queue_max_pkts × avg_pkt_bytes × 8) / (min_cap_kbps × 1000)
        # Per-topology constant (depends on min-cap link); recomputed per env instance
        # so cross-topology transfer keeps absolute physical scale.
        self._D_max_topo = None  # lazy-computed on first get_link_features call
        assert hasattr(self, "numPairs"), "Call build_pair_index() before kpath_init()."

    def build_pair_index(self):
        # 在 generate_environment() 裡面會被呼叫一次，建立 self.pairs 和 self.pair2id
        # 應該只有用到 pair
        """固定 token 順序：N = numNodes*(numNodes-1)"""
        self.pairs = []
        self.pair2id = {}
        idx = 0
        for s in range(self.numNodes):
            for d in range(self.numNodes):
                if s == d:
                    continue
                self.pairs.append((s, d))
                self.pair2id[(s, d)] = idx
                idx += 1
        self.numPairs = len(self.pairs)

    def precompute_path_structures_for_routenet(self):
        """
        依照 self.pairs + self.allPaths 建立 RouteNet bipartite 所需的 incidence arrays
        產物：
          self.numPaths = numPairs*K
          self.path_id      : (T,) 每個 incidence 屬於哪個 path
          self.path_link_id : (T,) 每個 incidence 對應的 link edgeId
          self.path_seq     : (T,) 在該 path 內的 hop index
          self.path_len     : (numPaths,) 每條 path 的 hop 長度
          self.path_pair_id : (numPaths,) 這條 path 屬於哪個 pair（方便你 pool K paths→pair token）
        """
        assert hasattr(self, "pairs") and hasattr(self, "numPairs"), "Call build_pair_index() first"
        assert hasattr(self, "K"), "Need self.K"
        assert hasattr(self, "allPaths"), "Need k paths loaded into self.allPaths"

        numPairs = self.numPairs
        K = int(self.K)
        numPaths = numPairs * K

        link_ids = []
        path_ids = []
        seqs = []
        path_len = np.zeros((numPaths,), dtype=np.int32)
        path_pair_id = np.zeros((numPaths,), dtype=np.int32)

        # helper: node path (n1,n2,n3) -> edge ids (e1,e2)
        def nodes_to_edge_ids(node_path):
            eids = []
            for t in range(len(node_path) - 1):
                u, v = node_path[t], node_path[t + 1]
                key = f"{u}:{v}"
                if key not in self.edgesDict:
                    raise RuntimeError(f"edge missing in edgesDict: {key}")
                eids.append(self.edgesDict[key])
            return eids

        # 一個pair一個pair處理
        for pair_id, (s, d) in enumerate(self.pairs):
            key = f"{s}:{d}"
            if key not in self.allPaths:
                raise RuntimeError(f"Missing allPaths for pair {key}")

            paths_k = self.allPaths[key]
            if len(paths_k) < K:
                raise RuntimeError(f"allPaths[{key}] has only {len(paths_k)} paths, need K={K}")

            for k in range(K):
                # pid=目前在處理的是哪一個pair的哪一個k path, 總共有 numPairs*K 個
                pid = pair_id * K + k
                path_pair_id[pid] = pair_id

                node_path = paths_k[k]
                eids = nodes_to_edge_ids(node_path)

                path_len[pid] = len(eids)

                for hop, eid in enumerate(eids):
                    link_ids.append(eid)
                    path_ids.append(pid)
                    seqs.append(hop)

        self.numPaths = numPaths
        self.path_link_id = np.asarray(link_ids, dtype=np.int64)
        self.path_id = np.asarray(path_ids, dtype=np.int64)
        self.path_seq = np.asarray(seqs, dtype=np.int64)
        self.path_len = path_len
        self.path_pair_id = path_pair_id

        self._build_undirected_edge_index()

    def _build_undirected_edge_index(self):
        """
        Build mapping from directed edge id (eid) to undirected edge id (uid),
        plus per-uid (eid_ab, eid_ba) to fetch two directions quickly.

        Assumes edgesDict contains BOTH "u:v" and "v:u".
        """
        E = int(self.numEdges)

        # eid -> (u,v)
        eid2uv = np.zeros((E, 2), dtype=np.int32)
        for key, eid in self.edgesDict.items():
            u, v = map(int, key.split(":"))
            eid2uv[eid, 0] = u
            eid2uv[eid, 1] = v
        self.eid2uv = eid2uv

        # build undirected ids
        undir_key2id = {}
        undir_edges = []
        for eid in range(E):
            u, v = int(eid2uv[eid, 0]), int(eid2uv[eid, 1])
            a, b = (u, v) if u < v else (v, u)
            k = (a, b)
            if k not in undir_key2id:
                undir_key2id[k] = len(undir_edges)
                undir_edges.append(k)

        U = len(undir_edges)
        eid2uid = np.zeros((E,), dtype=np.int32)
        for eid in range(E):
            u, v = int(eid2uv[eid, 0]), int(eid2uv[eid, 1])
            a, b = (u, v) if u < v else (v, u)
            eid2uid[eid] = undir_key2id[(a, b)]

        # for each uid store (eid_ab, eid_ba) to read directed cap/util easily
        uid_eids = np.full((U, 2), -1, dtype=np.int32)
        for (a, b), uid in undir_key2id.items():
            e_ab = self.edgesDict.get(f"{a}:{b}", None)
            e_ba = self.edgesDict.get(f"{b}:{a}", None)
            if e_ab is None or e_ba is None:
                raise RuntimeError(f"Missing directed edges for undirected pair {(a,b)}")
            uid_eids[uid, 0] = int(e_ab)
            uid_eids[uid, 1] = int(e_ba)

        self.undir_key2id = undir_key2id
        self.undir_edges = undir_edges
        self.eid2uid = eid2uid
        self.uid_eids = uid_eids  # (U,2): [eid_ab, eid_ba]
    
    #===================================================
    #== building RouteNet input regime : rollout full routing 
    #== 1) kpath reset 
    #== 2) rollout old action to bulid old regime 
    #===================================================

    def rollout_full(self, full_actions, track_crossing=False, assume_cleared=False):
        # 跟 kpath_init 配合使用 湊出 old regime 給 RouteNet 當 input
        """
        將 full_actions 對應的 routing 全部套用到空 topo 上（完整 rollout）
        full_actions: shape (N,) int64
        """
        full_actions = np.asarray(full_actions, dtype=np.int64)
        assert full_actions.shape[0] == self.numPairs
        if not assume_cleared:
            self.clear_utilization(clear_crossing=track_crossing)
        elif track_crossing:
            # 要 track crossing 的話就算 assume_cleared 也要清 crossing_paths
            for i in self.graph:
                for j in self.graph[i]:
                    if 'crossing_paths' in self.graph[i][j][0]:
                        self.graph[i][j][0]['crossing_paths'].clear()

        for idx, (s, d) in enumerate(self.pairs):
            bw = float(self.TM[s][d])
            if bw <= 0:
                continue

            kp = int(full_actions[idx])
            path = self.allPaths[str(s)+':'+str(d)][kp]
            for t in range(len(path) - 1):
                u, v = path[t], path[t + 1]
                self.graph[u][v][0]['utilization'] += bw
                if track_crossing:
                    self.graph[u][v][0]['crossing_paths'][f"{s}:{d}"] = bw

        # Advance queue by 1 step — writes delay/pkloss/queue_pkts to graph edges
        self._update_queues()

    def kpath_reset(self, tm_id, config, last_actions=None, track_crossing=False):
        """
        MaskGIT 統一 reset：
        1) 讀 TM + clear
        2) 立刻 rollout 出 old regime（last_actions or baseline）
        3) 計算 mean_bw_episode
        """
        if self.paths_metrics_minmax_dict is None:
            self.init_minmax_dic_like_trainloader(num_node_1based=config["num_node"])

        tm_id = int(tm_id)
        if getattr(self, "current_tm_id", None) == tm_id:
            # 只reset util
            self.clear_utilization(clear_crossing=True)
        else:
            # 還要多讀TM
            self.clean_and_generate_tm(tm_id)   

        if last_actions is None:
            # step1: no old action : SP
            old_actions = np.zeros((self.numPairs,), dtype=np.int64)
        else:
            # other steps : last action
            old_actions = np.asarray(last_actions, dtype=np.int64)

        # rollout 到 graph，讓 routenet input 有 regime
        # rollout_full 內建 _update_queues × 1，即 1 步 pre-fill。
        # 對齊 real env 實測：t=40→60 雖為 2 step 名義時間，但 iperf3/flow-install
        # ramp-up 使實際 excess 只累積 ~1.6 步 queue。1 步 pre-fill 誤差最小。
        self.rollout_full(old_actions, track_crossing=track_crossing, assume_cleared=True)

         # ---- build old reward table (optional) ----
        if self.kpath_cfg.get("reward_def", "new") in ("old", "old_sat"):
            rewards_dic, rewards_indicator, loss_value, delay_value = \
                self.compute_all_pair_action_rewards_like_trainloader(config)

            self._old_rewards_dic = rewards_dic
            self._old_rewards_indicator = rewards_indicator
            # 你要 log 的話留著

        # mean bw
        bw_list = [float(self.TM[s][d]) for (s, d) in self.pairs if float(self.TM[s][d]) > 0]
        self._mean_bw_episode = float(np.mean(bw_list)) if bw_list else 0.0
        
    #==========================
    #=== Routenet Inputs ======
    #==========================

    def _compute_lu_est_per_link(self):
        """Pure helper: compute unclamped LU estimate per directed link from
        current graph state (delay/loss/utilization).

        Returns:
            ratio_est: (numEdges,) float64 — unclamped LU clamped to
            [0, lu_est_clamp_max] (default 3.0). Reads tunable thresholds from
            self.kpath_cfg.

        Idempotent within a step (same graph state → same return value).
        Used by both:
          - get_link_features (state path, encoder input)
          - compute_metrics_fast (reward path)
        (The design rationale lived in a planning doc that is not part of this
        repository.)
        """
        cfg = self.kpath_cfg
        cap = self.edge_state[:, 1].astype(np.float64)
        dt = float(cfg.get("queue_step_duration", MONITOR_PERIOD))
        delay_thr_pkts = float(cfg.get("lu_est_delay_thr_pkts", 4))
        loss_thr = float(cfg.get("lu_est_loss_threshold", 0.002))
        combine = str(cfg.get("lu_est_combine", "max"))
        txutil_gate = float(cfg.get("lu_est_txutil_gate", 0.95))
        clamp_max = float(cfg.get("lu_est_clamp_max", 3.0))
        Q_max_pkts = float(cfg.get("queue_max_pkts", 1000))
        avg_pkt_bytes = float(cfg.get("queue_avg_pkt_bytes", 1488))
        ratio_est = np.zeros((self.numEdges,), dtype=np.float64)
        for key, e_idx in self.edgesDict.items():
            u_str, v_str = key.split(':')
            u, v = int(u_str), int(v_str)
            edge = self.graph[u][v][0]
            D_curr = float(edge.get('delay', 0.0))
            D_prev = float(self.prev_link_delay.get((u, v), 0.0))
            loss = float(edge.get('pkloss', 0.0))
            cap_kbps = float(cap[e_idx])
            tx_kbps = float(edge.get('utilization', 0.0))
            tx_util = min(1.0, tx_kbps / (cap_kbps + 1e-9)) if cap_kbps > 0 else 0.0
            cap_bps = max(cap_kbps * 1000.0, 1e-9)
            pkt_time_sec = (avg_pkt_bytes * 8.0) / cap_bps
            delay_thr_link = delay_thr_pkts * pkt_time_sec
            D_max_link = Q_max_pkts * pkt_time_sec
            ratio_est[e_idx] = estimate_lu_from_delay_loss(
                D_curr, D_prev, dt, loss, tx_util,
                delay_thr_link, loss_thr, D_max_link, combine, txutil_gate,
                clamp_max,
            )
        return ratio_est

    def _compute_D_max_topo(self):
        """Topology-wide max queueing delay (seconds), for delay normalization.

        D_max_topo = buffer_bits / min_cap_bps
                   = (queue_max_pkts × avg_pkt_bytes × 8) / (min_cap_kbps × 1000)

        Min-cap link in topology is the worst-case bottleneck — its full buffer
        drain time is the upper bound for queueing delay anywhere in this topo.
        Normalizing by this constant gives delay_norm ∈ [0, 1] with a
        topology-aware ceiling: same physical fraction-of-buffer-full across
        Geant (1.55Mbps) and 32node (different min-cap), so encoder learns a
        transfer-invariant signal.
        """
        cfg = getattr(self, 'kpath_cfg', {}) or {}
        max_queue_pkts = float(cfg.get("queue_max_pkts", 1000))
        avg_pkt_bytes  = float(cfg.get("queue_avg_pkt_bytes", 1488))
        min_cap_kbps = float('inf')
        for u in self.graph:
            for v in self.graph[u]:
                edge = self.graph[u][v][0]
                if 'capacity' in edge:
                    c = float(edge['capacity'])
                else:
                    try:
                        c = float(self.links_bw[u][v])
                    except (KeyError, IndexError, TypeError):
                        continue
                if 0.0 < c < min_cap_kbps:
                    min_cap_kbps = c
        if not (min_cap_kbps < float('inf')) or min_cap_kbps <= 0:
            raise RuntimeError("D_max_topo: no positive link capacity found")
        return (max_queue_pkts * avg_pkt_bytes * 8.0) / (min_cap_kbps * 1000.0)

    def get_link_features(self, use_undirected=False):
        """
        取得 link feature, dim 依 layout knob 決定:

        - **default** (4ch, back-compat interleaved):
            ch0 cap_norm (S)         — capacity / maxCapacity
            ch1 util_ratio (D)       — min(1, util/cap), CLAMPED
            ch2 rem_ratio (D)        — max(0, 1 - util/cap), clamped headroom
            ch3 betweenness (S)

        - **link_state_with_lu_est=True** (4ch v4 additive, static-first):
            ch0 cap_norm (S)
            ch1 betweenness (S)
            ch2 util_ratio (D)       — clamped, primary load signal
            ch3 lu_est_norm (D)      — lu_est / clamp_max ∈ [0, 1], derived overload sensor

          2026-05-17: redesigned from v2 (which replaced ch1 with lu_est_norm).
          Now ADDITIVE — keeps clamped util at ch2 (same position as lossdelay),
          adds lu_est at ch3 as derived overload sensor. Symmetric with lossdelay
          for clean raw-vs-derived comparison.

        - **link_state_with_lossdelay=True** (5ch v3, static-first, drop rem):
            ch0 cap_norm (S)
            ch1 betweenness (S)
            ch2 util_ratio (D)       — clamped, primary load signal
            ch3 delay_norm (D)       — graph['delay'] / D_max_topo ∈ [0, 1]
            ch4 loss_norm (D)        — graph['pkloss'] ∈ [0, 1] (already ratio)

          Drops rem_ratio (algebraically redundant with util_ratio under clamp);
          adds raw delay/loss for overload-regime discrimination that util_ratio
          clamps away. delay/loss in real env from controller telemetry, in sim
          from fluid queue _update_queues — same ['delay']/['pkloss'] keys.

        Recon convention (both lossdelay and lu_est): supervise link_state[:, 2:]
        — first 2 channels are static (cap, between), recon supervises only
        dynamic channels (util + overload sensors).

        預設用 directed edges (與 edgesDict / allPaths 一致).
        """
        eps = 1e-8
        E = self.numEdges

        use_lu_est    = bool(getattr(self, "_link_state_with_lu_est", False))
        use_lossdelay = bool(getattr(self, "_link_state_with_lossdelay", False))
        use_ls2ic     = bool(getattr(self, "_link_state_ls2ic_3ch",     False))
        # mutex enforced in kpath_init; keep redundant guard
        assert sum([use_lu_est, use_lossdelay, use_ls2ic]) <= 1

        # Both lossdelay and ls2ic_3ch need D_max_topo for delay_norm.
        # (2026-05-23 attempt: ls2ic *5 = /200ms literal alignment caused
        # path sum_delay magnitudes to break encoder recon balance — recon
        # loss 80× normal, encoder grad ratio critic/recon dropped to 1e-4
        # vs design 0.1. Reverted to /D_max_topo + clamp [0,1]: encoder
        # operating regime restored; reward path stays literal-aligned via
        # internal *1000 ms conversion in _ls2ic_reward.)
        if use_lossdelay or use_ls2ic:
            if self._D_max_topo is None:
                self._D_max_topo = self._compute_D_max_topo()
            D_max_topo = self._D_max_topo

        if use_lossdelay:
            F = 5
        elif use_ls2ic:
            F = 3                        # ls2ic strict: bwd_ratio + delay_norm + loss_norm
        else:
            F = 4
        feat = np.zeros((E, F), dtype=np.float32)

        if use_lu_est:
            clamp_max = float(self.kpath_cfg.get("lu_est_clamp_max", 3.0))
            ratio_est = self._compute_lu_est_per_link()  # (E,) unclamped LU

        for key, eid in self.edgesDict.items():
            u_str, v_str = key.split(":")
            u, v = int(u_str), int(v_str)

            cap = float(self.graph[u][v][0].get("capacity", self.links_bw[u][v]))
            util = float(self.graph[u][v][0].get("utilization", 0.0))

            cap_norm = cap / (float(self.maxCapacity) + eps)
            util_ratio = min(1.0, util / (cap + eps))     # clamped
            rem_ratio = max(0.0, (cap - util) / (cap + eps))
            betweenness = float(self.betweenness_centrality[eid])

            if use_lossdelay:
                # v3 layout (5ch S-first): [cap_norm, betweenness, util_ratio, delay_norm, loss_norm]
                delay_sec = float(self.graph[u][v][0].get('delay', 0.0))
                pkloss    = float(self.graph[u][v][0].get('pkloss', 0.0))
                delay_norm = min(1.0, max(0.0, delay_sec / max(D_max_topo, eps)))
                loss_norm  = min(1.0, max(0.0, pkloss))
                feat[eid, 0] = cap_norm
                feat[eid, 1] = betweenness
                feat[eid, 2] = util_ratio
                feat[eid, 3] = delay_norm
                feat[eid, 4] = loss_norm
            elif use_ls2ic:
                # ls2ic 3ch alignment (2026-05-23, revised v2):
                #   ch0 bwd_ratio  = max(0, (cap - util) / cap)   ∈ [0, 1]
                #        Directed equivalent of ls2ic ch0 (cur_bwd/total_cap),
                #        sign flipped to "free bw" direction per user request.
                #   ch1 delay_norm = clamp(delay_sec / D_max_topo, [0, 1])
                #        STRIDE-conventional norm (not ls2ic literal /200ms).
                #        Rationale (oz87dtld postmortem): *5 unclamped exploded
                #        path sum_delay → recon loss 80× normal → encoder grad
                #        ratio critic/recon dropped to 1e-4 vs design 0.1.
                #        /D_max_topo + clamp restores encoder operating regime.
                #   ch2 loss_norm  = clamp(pkloss, [0, 1])         ∈ [0, 1]
                #        STRIDE ratio convention (vs ls2ic percent literal).
                # ls2ic reward semantics (1.5 ms floor, 0.001 % floor, min-max
                # history, /100 scaling) stay fully literal in _ls2ic_reward —
                # state-side encoder input unit is decoupled from reward unit.
                delay_sec = float(self.graph[u][v][0].get('delay', 0.0))
                pkloss    = float(self.graph[u][v][0].get('pkloss', 0.0))
                feat[eid, 0] = rem_ratio
                feat[eid, 1] = min(1.0, max(0.0, delay_sec / max(D_max_topo, eps)))
                feat[eid, 2] = min(1.0, max(0.0, pkloss))
            elif use_lu_est:
                # v4 layout (additive static-first): [cap_norm, betweenness, util_ratio, lu_est_norm]
                lu_est_norm = float(min(ratio_est[eid] / max(clamp_max, eps), 1.0))
                feat[eid, 0] = cap_norm
                feat[eid, 1] = betweenness
                feat[eid, 2] = util_ratio
                feat[eid, 3] = lu_est_norm
            else:
                # Original layout (back-compat): [cap_norm, util_ratio, rem_ratio, betweenness]
                feat[eid, 0] = cap_norm
                feat[eid, 1] = util_ratio
                feat[eid, 2] = rem_ratio
                feat[eid, 3] = betweenness

        return feat  # (E, F_e=4 or 5)

    #=================================================
    #== actions ---rollout---> metrics -> reward   (old or new with COMA)==
    #=================================================
    
    def build_full_actions(self, decided_actions, decided_mask, baseline_actions):
        """
        decided_actions: shape (N,) int64
        decided_mask:    shape (N,) bool, True=已決策
        return full_actions: shape (N,) int64
        """
        decided_actions = np.asarray(decided_actions, dtype=np.int64)
        decided_mask = np.asarray(decided_mask, dtype=np.bool_)

        # 讀 baseline actions + error raise
        if baseline_actions is None:
            raise RuntimeError("[build_full_actions] baseline_actions is None.")
        baseline_actions = np.asarray(baseline_actions, dtype=np.int64)

        if baseline_actions.shape[0] != self.numPairs:
            raise RuntimeError(f"[build_full_actions] baseline_actions size mismatch: {baseline_actions.shape[0]} vs numPairs={self.numPairs}")

        # full = baseline + decided overwrite (目前應該會decide all, 用不到baseline)
        full = baseline_actions.copy()
        decided_actions = np.asarray(decided_actions, dtype=np.int64)
        decided_mask = np.asarray(decided_mask, dtype=np.bool_)
        full[decided_mask] = decided_actions[decided_mask]
        return full
    
    def _link_arrays_from_graph(self):
        """
        Return per-directed-edge arrays aligned with edgesDict:
        cap(E,), util(E,), delay(E,), loss(E,)
        delay: any unit you store (ms or normalized), loss: probability in [0,1]
        """
        eps = float(self.kpath_cfg.get("eps", 1e-8)) if hasattr(self, "kpath_cfg") else 1e-8

        E = self.numEdges
        cap  = np.zeros((E,), dtype=np.float64)
        util = np.zeros((E,), dtype=np.float64)
        delay = np.zeros((E,), dtype=np.float64)
        loss  = np.zeros((E,), dtype=np.float64)

        for key, eid in self.edgesDict.items():
            u_str, v_str = key.split(":")
            u, v = int(u_str), int(v_str)
            edge = self.graph[u][v][0]

            cap_k  = float(edge.get("capacity", self.links_bw[u][v]))
            util_k = float(edge.get("utilization", 0.0))

            # if you didn't fill these, keep stable defaults
            delay_k = float(edge.get("delay", 1e-6))
            loss_k  = float(edge.get("pkloss", 0.0))

            cap[eid] = cap_k
            util[eid] = util_k
            delay[eid] = delay_k
            loss[eid] = loss_k

        ratio = util / (cap + eps)
        return cap, util, ratio, delay, loss

    def compute_path_metrics_fast(
        self,
        full_actions=None,
        util=None,
        cap=None,
        need_delay=True,
        need_loss=True,
    ):
        """
        Compute per-path metrics using incidence arrays.

        Always computes:
        - bwd_path: min(cap-util) along path

        Optionally computes (if need_delay/need_loss):
        - delay_path: sum(delay_link)
        - loss_path : 1 - Π(1-loss_link)

        If full_actions provided, returns chosen_* per pair.
        """
        assert hasattr(self, "path_id") and hasattr(self, "path_link_id"), \
            "Call precompute_path_structures_for_routenet() first."
        assert hasattr(self, "numPairs") and hasattr(self, "K"), \
            "Need build_pair_index() and self.K set."
        assert hasattr(self, "numPaths"), "Need self.numPaths"

        eps = float(getattr(self, "kpath_cfg", {}).get("eps", 1e-8))
        E = int(self.numEdges)
        P = int(self.numPaths)

        # ------------- build per-edge arrays -------------
        if cap is None or util is None:
            cap_arr  = np.zeros((E,), dtype=np.float64)
            util_arr = np.zeros((E,), dtype=np.float64)

            delay_arr = np.zeros((E,), dtype=np.float64) if need_delay else None
            loss_arr  = np.zeros((E,), dtype=np.float64) if need_loss  else None

            for key, eid in self.edgesDict.items():
                u_str, v_str = key.split(":")
                u, v = int(u_str), int(v_str)
                edge = self.graph[u][v][0]

                cap_arr[eid]  = float(edge.get("capacity", self.links_bw[u][v]))
                util_arr[eid] = float(edge.get("utilization", 0.0))

                if need_delay:
                    delay_arr[eid] = float(edge.get("delay", 1e-6))
                if need_loss:
                    loss_arr[eid] = float(edge.get("pkloss", 0.0))

        else:
            cap_arr  = np.asarray(cap, dtype=np.float64)
            util_arr = np.asarray(util, dtype=np.float64)

            delay_arr = None
            loss_arr  = None

            # util/cap 已給 → 只有真的需要 delay/loss 才掃 graph
            if need_delay:
                delay_arr = np.zeros((E,), dtype=np.float64)
            if need_loss:
                loss_arr = np.zeros((E,), dtype=np.float64)

            if need_delay or need_loss:
                for key, eid in self.edgesDict.items():
                    u_str, v_str = key.split(":")
                    u, v = int(u_str), int(v_str)
                    edge = self.graph[u][v][0]
                    if need_delay:
                        delay_arr[eid] = float(edge.get("delay", 1e-6))
                    if need_loss:
                        loss_arr[eid] = float(edge.get("pkloss", 0.0))

        # ------------- aggregate per path -------------
        inc_path = self.path_id.astype(np.int64)        # (T,)
        inc_eid  = self.path_link_id.astype(np.int64)   # (T,)
        metric_mode = str(getattr(self, "kpath_cfg", {}).get("metric_mode", "directed"))
        
        if metric_mode == "directed":
            rem_dir = np.maximum(0.0, cap_arr - util_arr)  # (E,)
            bwd_path = np.full((P,), np.inf, dtype=np.float64)
            np.minimum.at(bwd_path, inc_path, rem_dir[inc_eid])
            bwd_path = np.where(np.isfinite(bwd_path), bwd_path, 0.0)
        elif metric_mode == "undirected":
            assert hasattr(self, "eid2uid") and hasattr(self, "uid_eids"), \
                "Call _build_undirected_edge_index() first"

            U = int(self.uid_eids.shape[0])
            uid = self.eid2uid  # (E,)

            e_ab = self.uid_eids[:, 0].astype(np.int64)
            e_ba = self.uid_eids[:, 1].astype(np.int64)
            cap_undir = cap_arr[e_ab] + cap_arr[e_ba]              # sum capacity

            # Per-direction clamp (matches real MN physics):
            # each direction's actual throughput can't exceed its capacity
            util_clamped = np.minimum(util_arr, cap_arr)           # (E,)
            util_undir = np.zeros((U,), dtype=np.float64)
            np.add.at(util_undir, uid, util_clamped)               # sum of clamped
            rem_undir = np.maximum(0.0, cap_undir - util_undir)    # shared remaining

            inc_uid = uid[inc_eid]  # (T,)
            bwd_path = np.full((P,), np.inf, dtype=np.float64)
            np.minimum.at(bwd_path, inc_path, rem_undir[inc_uid])
            bwd_path = np.where(np.isfinite(bwd_path), bwd_path, 0.0)
        else:
            raise ValueError(f"Unknown metric_mode={metric_mode}")

        out = {"bwd_path": bwd_path}

        if need_delay:
            delay_path = np.zeros((P,), dtype=np.float64)
            np.add.at(delay_path, inc_path, delay_arr[inc_eid])
            out["delay_path"] = delay_path

        if need_loss:
            loss_prob_link = np.clip(loss_arr, 0.0, 1.0)
            safe_term = np.clip(1.0 - loss_prob_link, eps, 1.0)
            sum_log = np.zeros((P,), dtype=np.float64)
            np.add.at(sum_log, inc_path, np.log(safe_term)[inc_eid])
            survive = np.exp(sum_log)
            loss_path = np.clip(1.0 - survive, 0.0, 1.0)
            out["loss_path"] = loss_path

        # ------------- chosen per pair -------------
        if full_actions is not None:
            full_actions = np.asarray(full_actions, dtype=np.int64)
            if full_actions.shape[0] != self.numPairs:
                raise RuntimeError(
                    f"[compute_path_metrics_fast] full_actions size mismatch: "
                    f"{full_actions.shape[0]} vs numPairs={self.numPairs}"
                )
            K = int(self.K)
            selected_pid = (np.arange(self.numPairs, dtype=np.int64) * K + full_actions)

            out["chosen_bwd"] = bwd_path[selected_pid]
            if need_delay:
                out["chosen_delay"] = out["delay_path"][selected_pid]
            if need_loss:
                out["chosen_loss"] = out["loss_path"][selected_pid]

        return out

    def compute_metrics_fast(self, full_actions, return_vectors=True):
        cfg = self.kpath_cfg
        metric_mode = str(cfg.get("metric_mode", "directed"))
        eps = float(cfg.get("eps", 1e-8))
        p = float(cfg.get("over_power", 2.0))

        ignore_zero_bw = bool(cfg.get("ignore_zero_bw", True))
        sat_w_mode = cfg.get("sat_weight_mode", "demand")

        over_lin = float(cfg.get("over_lin", 0.0))
        over_scale = float(cfg.get("over_scale", 1.0))
        over_agg = cfg.get("over_agg", "sum")

        reward_mode = cfg.get("reward_mode", "all")
        bwd_only = (reward_mode == "bwd_only")

        full_actions = np.asarray(full_actions, dtype=np.int64)
        assert full_actions.shape[0] == self.numPairs

        # ---- cap + bw per pair ----
        cap = self.edge_state[:, 1].astype(np.float64)
        bw_pair = np.zeros((self.numPairs,), dtype=np.float64)
        for pid, (s, d) in enumerate(self.pairs):
            bw_pair[pid] = float(self.TM[s][d])

        if ignore_zero_bw:
            valid_pair_mask = (bw_pair > 0.0)
            w_demand  = np.where(valid_pair_mask, bw_pair, 0.0)
            w_uniform = np.where(valid_pair_mask, 1.0,   0.0)
        else:
            valid_pair_mask = np.ones_like(bw_pair, dtype=bool)
            w_demand  = bw_pair
            w_uniform = np.ones_like(bw_pair)

        # ---- selected global path id ----
        K = int(self.K)
        selected_pid = (np.arange(self.numPairs, dtype=np.int64) * K + full_actions)

        # ---- incidence arrays ----
        inc_path = self.path_id
        inc_eid  = self.path_link_id
        inc_pair = self.path_pair_id[inc_path]
        active = (inc_path == selected_pid[inc_pair])

        # ---- util on directed edges ----
        util = np.zeros((self.numEdges,), dtype=np.float64)
        np.add.at(util, inc_eid[active], bw_pair[inc_pair[active]])

        # ---- per-link unclamped LU source ----
        # "offered"        : oracle ratio = util / cap (sim ground truth; TM-driven)
        # "delay_loss_est" : observable estimator; works in both sim and real
        lu_source = str(cfg.get("lu_source", "offered"))
        ratio_gt = util / (cap + eps)  # always compute GT for sim-side validation logging

        ratio_est = None
        if lu_source == "delay_loss_est":
            # 2026-05-07: extracted to env._compute_lu_est_per_link helper so that
            # state-path (get_link_features, v2 lu_est layout) and reward-path
            # share one implementation. Idempotent within a step.
            ratio_est = self._compute_lu_est_per_link()

        # =========================================================
        # metric_mode switch
        # =========================================================
        if metric_mode == "directed":
            ratio = ratio_est if ratio_est is not None else ratio_gt  # (E,)
            mlu = float(np.max(ratio)) if ratio.size > 0 else 0.0

            over = np.maximum(0.0, ratio - 1.0)
            over_term = (over_lin * over) + (np.power(over, p))
            over_term *= over_scale

            overload_sum  = float(np.sum(over_term))
            overload_mean = float(np.mean(over_term)) if over_term.size > 0 else 0.0
            overload_max  = float(np.max(over_term)) if over_term.size > 0 else 0.0
            if over_agg == "mean":
                overload = overload_mean
            elif over_agg == "max":
                overload = overload_max
            else:
                overload = overload_sum

            # path bottleneck ratio (max ratio along chosen path)
            P = self.numPairs * K
            path_max = np.zeros((P,), dtype=np.float64)
            np.maximum.at(path_max, inc_path[active], ratio[inc_eid[active]])
            chosen_path_max = path_max[selected_pid]
            chosen_path_max = np.where(chosen_path_max > 0.0, chosen_path_max, 1.0)

            # over share per pair
            over_share_pair = np.zeros((self.numPairs,), dtype=np.float64)
            np.add.at(over_share_pair, inc_pair[active], over_term[inc_eid[active]])

            # Directed mode: per-direction is already the natural unit, clamp = min(ratio, 1)
            mlu_clamped = min(mlu, 1.0)

            # [PATCH A] vectors follow current mode
            over_vec = over
            over_term_vec = over_term

        elif metric_mode == "undirected":
            assert hasattr(self, "eid2uid") and hasattr(self, "uid_eids")

            U = int(self.uid_eids.shape[0])
            uid = self.eid2uid  # (E,)

            e_ab = self.uid_eids[:, 0].astype(np.int64)
            e_ba = self.uid_eids[:, 1].astype(np.int64)
            cap_ab = cap[e_ab]
            cap_ba = cap[e_ba]
            cap_undir = cap_ab + cap_ba                      # (U,)

            # ---- Undirected util: 先 clamp 每個 directed 方向，再加總 ----
            # 物理上每方向 throughput 不超過該方向 capacity
            # (300%, 0%) → clamp → (100%, 0%) → sum → 100/200 = 50%
            util_clamped = np.minimum(util, cap)              # clamp each direction
            util_undir = np.zeros((U,), dtype=np.float64)
            np.add.at(util_undir, uid, util_clamped)          # sum of clamped

            ratio_u = util_undir / (cap_undir + eps)          # (U,)
            mlu = float(np.max(ratio_u)) if ratio_u.size > 0 else 0.0
            mlu_clamped = mlu  # undirected 本身就是 clamp 後算的

            # ---- Overload: 用 directed per-edge ratio ----
            # undirected clamp-first 後 ratio_u <= 1.0，overload 幾乎永遠 0
            # 改用 directed ratio 保留 overload 訊號；estimator 優先
            ratio = ratio_est if ratio_est is not None else (util / (cap + eps))  # directed (E,)
            over = np.maximum(0.0, ratio - 1.0)
            over_term = (over_lin * over) + (np.power(over, p))
            over_term *= over_scale

            overload_sum  = float(np.sum(over_term))
            overload_mean = float(np.mean(over_term)) if over_term.size > 0 else 0.0
            overload_max  = float(np.max(over_term)) if over_term.size > 0 else 0.0
            if over_agg == "mean":
                overload = overload_mean
            elif over_agg == "max":
                overload = overload_max
            else:
                overload = overload_sum

            # path bottleneck ratio: max undirected ratio along chosen path (for sat)
            P = self.numPairs * K
            path_max = np.zeros((P,), dtype=np.float64)
            inc_uid = uid[inc_eid]
            np.maximum.at(path_max, inc_path[active], ratio_u[inc_uid[active]])
            chosen_path_max = path_max[selected_pid]
            chosen_path_max = np.where(chosen_path_max > 0.0, chosen_path_max, 1.0)

            # over share per pair: 用 directed over_term，沿 chosen path 加總
            over_share_pair = np.zeros((self.numPairs,), dtype=np.float64)
            np.add.at(over_share_pair, inc_pair[active], over_term[inc_eid[active]])

            over_vec = over
            over_term_vec = over_term
        else:
            raise ValueError(f"Unknown metric_mode={metric_mode}")

        # ---- sat ----
        sat = np.minimum(1.0, 1.0 / (chosen_path_max + eps)).astype(np.float64)

        sat_w_demand  = float(np.sum(w_demand  * sat) / (np.sum(w_demand)  + eps))
        sat_w_uniform = float(np.sum(w_uniform * sat) / (np.sum(w_uniform) + eps))
        sat_weighted = sat_w_demand if sat_w_mode == "demand" else sat_w_uniform

        # ---- per-pair on-path util² sum (dense credit signal) ----
        # Replaces over_share_pair (which only fires when util>1). Every link on the
        # chosen path contributes util_l² regardless of threshold, giving the agent
        # gradient even in under-loaded regimes.
        util_sq_edge = np.power(ratio, 2.0)
        path_util_sq_sum = np.zeros((self.numPairs,), dtype=np.float64)
        np.add.at(path_util_sq_sum, inc_pair[active], util_sq_edge[inc_eid[active]])

        # ---- global soft-MLU aggregator (logsumexp / max / mean) ----
        # 2026-05-04: switched default to logsumexp after max(util^4) caused
        # training collapse (saturation at MLU≥2 in 50-99% of steps for the
        # 3 round-2 runs; reward stuck at ~-32 → no within-step variance →
        # critic can't learn → policy gradient = 0).
        # logsumexp: (1/τ) · log(mean exp(τ · util)) — interpolates between
        # mean (τ→0) and max (τ→∞). At τ=6 with Geant ~50 edges, top-3
        # of 50 dominate ~80% when u_top≈1.0, u_low≈0.3. Bounded by
        # max+log(N)/τ ≈ max+0.65 → at clamp=2, contribution ~-2.7 vs the
        # max(util^4) catastrophe of -32.
        agg = str(cfg.get("soft_mlu_aggregator", "logsumexp"))
        soft_mlu_p_exp = float(cfg.get("soft_mlu_p", 4.0))   # always defined (used in metrics dict)
        if ratio.size > 0:
            if agg == "logsumexp":
                tau = float(cfg.get("soft_mlu_tau", 6.0))
                # Numerical-safe LSE: shift by τ·max(ratio) to prevent overflow.
                m = float(np.max(ratio))
                shifted = tau * (ratio - m)
                soft_mlu_p = m + float(np.log(np.mean(np.exp(shifted)))) / max(tau, 1e-9)
            elif agg == "max":
                soft_mlu_p = float(np.max(np.power(ratio, soft_mlu_p_exp)))
            else:  # "mean" (legacy, dead signal)
                soft_mlu_p = float(np.mean(np.power(ratio, soft_mlu_p_exp)))
        else:
            soft_mlu_p = 0.0

        # ---- chosen path metrics ----
        pm = self.compute_path_metrics_fast(
            full_actions=full_actions,
            need_delay=(not bwd_only),
            need_loss=(not bwd_only),
        )
        chosen_bwd = pm["chosen_bwd"].astype(np.float64)

        metrics = {
            "mlu": mlu,
            "mlu_clamped": mlu_clamped,
            "overload": overload,
            "overload_sum": overload_sum,
            "overload_mean": overload_mean,
            "overload_max": overload_max,

            "sat_weighted": sat_weighted,
            "sat_w_demand": sat_w_demand,
            "sat_w_uniform": sat_w_uniform,

            "bwd_w_demand":  float(np.sum(w_demand  * chosen_bwd) / (np.sum(w_demand)  + eps)),
            "bwd_w_uniform": float(np.sum(w_uniform * chosen_bwd) / (np.sum(w_uniform) + eps)),

            "soft_mlu_p": soft_mlu_p,
            "soft_mlu_p_exp": soft_mlu_p_exp,
        }

        if not bwd_only:
            chosen_delay = pm["chosen_delay"].astype(np.float64)
            chosen_loss  = pm["chosen_loss"].astype(np.float64)
            metrics.update({
                "delay_w_demand":  float(np.sum(w_demand  * chosen_delay) / (np.sum(w_demand)  + eps)),
                "delay_w_uniform": float(np.sum(w_uniform * chosen_delay) / (np.sum(w_uniform) + eps)),
                "loss_w_demand":   float(np.sum(w_demand  * chosen_loss)  / (np.sum(w_demand)  + eps)),
                "loss_w_uniform":  float(np.sum(w_uniform * chosen_loss)  / (np.sum(w_uniform) + eps)),
            })

        if return_vectors:
            # use over_vec/over_term_vec (mode-specific)
            metrics.update({
                "util": util,
                "ratio": ratio,  # directed ratio kept for debug
                "over_vec": over_vec,
                "over_term_vec": over_term_vec,

                "sat_vec": sat.astype(np.float32),
                "bw_pair": bw_pair,
                "chosen_path_max": chosen_path_max,
                "valid_pair_mask": valid_pair_mask.astype(np.bool_),
                "over_share_pair": over_share_pair,
                "path_util_sq_sum": path_util_sq_sum,

                "chosen_bwd": chosen_bwd,
            })
            if not bwd_only:
                metrics.update({
                    "chosen_delay": chosen_delay,
                    "chosen_loss": chosen_loss,
                })

        # ---- sim-side validation logging: expose GT mlu alongside estimator mlu ----
        if ratio_est is not None and cfg.get("lu_log_gt_in_sim", True):
            mlu_gt = float(np.max(ratio_gt)) if ratio_gt.size > 0 else 0.0
            metrics["mlu_gt"] = mlu_gt
            metrics["mlu_abs_err"] = abs(mlu - mlu_gt)
            if return_vectors:
                metrics["ratio_gt"] = ratio_gt

        # ---- snapshot current delay → prev for next step (only when estimator active) ----
        if ratio_est is not None:
            for key in self.edgesDict:
                u_str, v_str = key.split(':')
                u, v = int(u_str), int(v_str)
                self.prev_link_delay[(u, v)] = float(self.graph[u][v][0].get('delay', 0.0))

        return metrics

    # -----------------
    # new reward 算法
    # ----------------

    def reward_vec_from_metrics(self, metrics):
        """
        Per-pair reward vector r_i.

        Default (backward-compatible):
        r_i = sat_vec[i] - lambda_over_share * over_share_pair[i]

        Optional shaping (set these in kpath_cfg):
        pair_beta_sat   (default 1.0)
        pair_beta_bwd   (default 0.0)  uses chosen_bwd  (higher better)
        pair_alpha_delay(default 0.0)  uses chosen_delay (lower better) -> subtract
        pair_alpha_loss (default 0.0)  uses chosen_loss  (lower better) -> subtract
        lambda_over_share (default 0.0) subtract overload share along chosen path
        """
        cfg = self.kpath_cfg
        eps = float(cfg.get("eps", 1e-8))

        beta_sat = float(cfg.get("pair_beta_sat", 1.0))
        beta_bwd = float(cfg.get("pair_beta_bwd", 0.0))
        alpha_delay = float(cfg.get("pair_alpha_delay", 0.0))
        alpha_loss  = float(cfg.get("pair_alpha_loss", 0.0))
        lam_over = float(cfg.get("lambda_over_share", 0.0))

        sat_vec = metrics.get("sat_vec", None)
        if sat_vec is None:
            raise RuntimeError("[reward_vec_from_metrics] metrics missing sat_vec. Use compute_metrics_fast(..., return_vectors=True).")

        r = beta_sat * sat_vec.astype(np.float32)

        # ---- optional: bwd/delay/loss shaping ----
        # We normalize them gently to keep scales stable across TMs.
        # - chosen_bwd: divide by (mean cap) or maxCapacity to get ~[0,1]
        # - chosen_delay: divide by delay_norm_div if provided
        # - chosen_loss: already prob in [0,1]
        if beta_bwd != 0.0:
            cb = metrics.get("chosen_bwd", None)
            if cb is None:
                raise RuntimeError("[reward_vec_from_metrics] missing chosen_bwd. compute_metrics_fast now provides it.")
            # normalize by maxCapacity as a stable scale (cap in same unit as util)
            denom = float(getattr(self, "maxCapacity", 1.0)) + eps
            r = r + (beta_bwd * (cb.astype(np.float32) / denom))

        # 先把new 裡面的 delay loss拔掉
        # if alpha_delay != 0.0:
        #     cd = metrics.get("chosen_delay", None)
        #     if cd is None:
        #         raise RuntimeError("[reward_vec_from_metrics] missing chosen_delay. compute_metrics_fast now provides it.")
        #     div = float(cfg.get("delay_norm_div", 200.0))
        #     r = r - (alpha_delay * (cd.astype(np.float32) / (div + eps)))

        # if alpha_loss != 0.0:
        #     cl = metrics.get("chosen_loss", None)
        #     if cl is None:
        #         raise RuntimeError("[reward_vec_from_metrics] missing chosen_loss. compute_metrics_fast now provides it.")
        #     r = r - (alpha_loss * cl.astype(np.float32))

        # ---- optional: overload share regularizer (legacy: only active when util>1) ----
        if lam_over > 0.0:
            over_share = metrics.get("over_share_pair", None)
            if over_share is None:
                raise RuntimeError("[reward_vec_from_metrics] metrics missing over_share_pair.")
            r = r - lam_over * over_share.astype(np.float32)

        # ---- new: per-pair on-path util² sum (dense credit signal) ----
        lam_path = float(cfg.get("lambda_path_util_sq", 0.0))
        if lam_path > 0.0:
            path_util_sq_sum = metrics.get("path_util_sq_sum", None)
            if path_util_sq_sum is None:
                raise RuntimeError("[reward_vec_from_metrics] metrics missing path_util_sq_sum.")
            r = r - lam_path * path_util_sq_sum.astype(np.float32)

        # ---- new: global soft-MLU (p-power mean); scalar broadcast to all pairs ----
        lam_mlu_soft = float(cfg.get("lambda_mlu_soft", 0.0))
        if lam_mlu_soft > 0.0:
            soft_mlu_p = float(metrics.get("soft_mlu_p", 0.0))
            r = r - np.float32(lam_mlu_soft * soft_mlu_p)

        # ---- zero-out invalid pairs (bw<=0) if requested ----
        valid = metrics.get("valid_pair_mask", None)
        if valid is not None and bool(cfg.get("ignore_zero_bw", True)):
            r = r.copy()
            r[~valid] = 0.0

        return r  # (numPairs,) float32

    def reward_vec_per_pair_ema_v1(self, metrics):
        """Per-pair EMA reward (DRL-OR-S style, but TE-direct indicators).

        Two terms:
          (a) per-pair: λ_util * path_util²_scaled[i] / EMA_path_util²[i]
              EMA-normalized so each pair compares against its own history;
              gives dense per-pair credit signal even in underload.
          (b) global:  λ_mlu * (Σ_l util^p_l) / N_edges
              Absolute scale (no EMA) — preserves headline MLU interpretability
              for the paper. Scalar broadcast to all pairs.

        r_i = -λ_util * path_util²_scaled[i]/EMA[i]  -  λ_mlu * soft_mlu_p  +  c

        Notes:
          - delay/loss are *not* separate terms. Under lu_source="offered" the
            ratio is the TM oracle (delay/loss are downstream effects); under
            lu_source="delay_loss_est" the ratio is reconstructed FROM delay/
            loss, so adding them as separate terms would double-count.
          - EMA state lives on env (process-level), inits from first observation
            so the initial scaled ratio starts near 1 (avoids critic TD blowup).
          - Reward sign: always negative; c shifts mean toward 0 (cosmetic, does
            not change optimal policy in contextual-bandit setting).
        """
        cfg = self.kpath_cfg
        N = int(self.numPairs)
        path_util_sq_sum = metrics.get("path_util_sq_sum", None)
        if path_util_sq_sum is None:
            raise RuntimeError("[reward_vec_per_pair_ema_v1] metrics missing path_util_sq_sum.")
        soft_mlu_p = float(metrics.get("soft_mlu_p", 0.0))

        beta      = float(cfg.get("per_pair_ema_beta", 0.99))
        clip_max  = float(cfg.get("per_pair_ema_clip", 10.0))
        util_clip = float(cfg.get("per_pair_ema_util_clip", 100.0))
        lam_util  = float(cfg.get("lambda_util_pair", 1.0))
        lam_mlu   = float(cfg.get("lambda_mlu_global", 2.0))
        c_shift   = float(cfg.get("reward_const_shift", 2.0))

        u_now = np.minimum(path_util_sq_sum.astype(np.float64), util_clip)

        state = self._per_pair_ema_state
        if state is None or state["util_sq_ema"].shape[0] != N:
            # Init from first observation so scaled ratio starts ~1.
            self._per_pair_ema_state = {
                "util_sq_ema": np.maximum(u_now, 1e-3).copy(),
                "beta": beta,
            }
        else:
            state["util_sq_ema"] = beta * state["util_sq_ema"] + (1.0 - beta) * u_now

        ema = self._per_pair_ema_state["util_sq_ema"]
        util_scaled = np.clip(u_now / (ema + 1e-9), 0.0, clip_max)

        r_per_pair = -lam_util * util_scaled
        r_global   = -lam_mlu * soft_mlu_p
        r = (r_per_pair + r_global + c_shift).astype(np.float32)

        # zero-out invalid pairs (bw<=0) if requested
        valid = metrics.get("valid_pair_mask", None)
        if valid is not None and bool(cfg.get("ignore_zero_bw", True)):
            r = r.copy()
            r[~valid] = 0.0

        return r  # (numPairs,) float32

    # - - - - - - - - - - - - - - - - - - - - - -
    # -  reward(evaluate_decisions) --> COMA - - 
    # - - - - - - - - - - - - - - - - - - - - - - -
    
    def counterfactual_advantages_per_pair(
        self,
        decided_actions,
        decided_mask,
        newly_idx,
        baseline_actions=None,
        full_actions=None,
        metrics_full=None,
        r_vec_full=None,
    ):
        """
        Per-pair COMA-like counterfactual:
        A_i = r_i(full) - r_i(full with i -> baseline)

        Now supports cache to avoid recomputing full regime.
        """
        newly_idx = list(newly_idx)

        if baseline_actions is None:
            raise RuntimeError("[counterfactual_advantages_per_pair] baseline_actions is None. Fallback is forbidden.")
        baseline_actions = np.asarray(baseline_actions, dtype=np.int64)

        # -------- full regime (cached if provided) --------
        if full_actions is None:
            full_actions = self.build_full_actions(decided_actions, decided_mask, baseline_actions)
        else:
            full_actions = np.asarray(full_actions, dtype=np.int64)

        if metrics_full is None:
            metrics_full = self.compute_metrics_fast(full_actions, return_vectors=True)

        if r_vec_full is None:
            r_vec_full = self.reward_vec_from_metrics(metrics_full)  # (N,)

        adv = np.zeros((len(newly_idx),), dtype=np.float32)

        # -------- counterfactual per newly committed --------
        for k, i in enumerate(newly_idx):
            cf_actions = full_actions.copy()
            cf_actions[i] = baseline_actions[i]
            cf_metrics = self.compute_metrics_fast(cf_actions, return_vectors=True)
            r_vec_cf = self.reward_vec_from_metrics(cf_metrics)
            adv[k] = float(r_vec_full[i] - r_vec_cf[i])

        info = {
            "mlu": float(metrics_full["mlu"]),
            "overload": float(metrics_full["overload"]),
            "sat_weighted": float(metrics_full["sat_weighted"]),
            "sat_w_demand": float(metrics_full.get("sat_w_demand", 0.0)),
            "sat_w_uniform": float(metrics_full.get("sat_w_uniform", 0.0)),
        }

        return adv, r_vec_full, metrics_full, info

    def _bw_of_pairs(self, idx_list):
        """
        Return bw only for idx with bw > 0
        """
        bw = []
        valid_idx = []

        for idx in idx_list:
            s, d = self.pairs[idx]
            bw_k = float(self.TM[s][d])
            if bw_k > 0:
                bw.append(bw_k)
                valid_idx.append(idx)

        return np.asarray(bw, dtype=np.float32), valid_idx

    def compute_commit_weights(self, newly_idx, mode="sqrt_bw", eps=1e-8):
        """
        w_i = sqrt(bw_i / mean_bw_episode)
        only for newly committed AND bw>0 tokens
        """
        newly_idx = list(newly_idx)
        if len(newly_idx) == 0:
            return np.zeros((0,), dtype=np.float32), []

        bw, valid_idx = self._bw_of_pairs(newly_idx)

        if len(valid_idx) == 0:
            return np.zeros((0,), dtype=np.float32), []

        mean_bw = float(getattr(self, "_mean_bw_episode", None))

        if mean_bw <= eps:
            w = np.ones_like(bw, dtype=np.float32)
            return w, valid_idx

        if mode == "sqrt_bw":
            w = np.sqrt(bw / (mean_bw + eps)).astype(np.float32)
        elif mode == "linear_bw":
            w = (bw / (mean_bw + eps)).astype(np.float32)
        elif mode == "uniform":
            w = np.ones_like(bw, dtype=np.float32)
        elif mode == "log1p_bw":
            w = (np.log1p(bw) / (np.log1p(mean_bw) + eps)).astype(np.float32)
            w = np.clip(w, 0.1, 10.0).astype(np.float32)
        else:
            raise ValueError(f"Unknown weight mode: {mode}")

        # optional stability
        w = w / (w.mean() + eps)

        return w, valid_idx

    def normalize_advantages(self, adv, eps=1e-8):
        """
        Per-batch normalization for newly committed adv only:
          A = (A - mean) / (std + eps)

        adv: np.ndarray shape (M,)
        return: np.ndarray shape (M,)
        """
        adv = np.asarray(adv, dtype=np.float32)
        if adv.size == 0:
            return adv

        mu = float(np.mean(adv))
        sigma = float(np.std(adv))

        if sigma <= eps:
            # 全部一樣 -> normalize 後全 0，避免爆梯度
            return np.zeros_like(adv, dtype=np.float32)

        return ((adv - mu) / (sigma + eps)).astype(np.float32)

    def counterfactual_advantages_with_fixes_per_pair(
        self,
        decided_actions,
        decided_mask,
        newly_idx,
        weight_mode="sqrt_bw",
        do_norm=True,
        baseline_actions=None,
        full_actions=None,
        metrics_full=None,
        r_vec_full=None,
    ):
        if baseline_actions is None:
            raise RuntimeError("[counterfactual_advantages_with_fixes_per_pair] baseline_actions is None. Fallback is forbidden.")
        baseline_actions = np.asarray(baseline_actions, dtype=np.int64)

        newly_idx = list(newly_idx)

        adv_raw_all, r_vec_full, metrics_full, info = self.counterfactual_advantages_per_pair(
            decided_actions=decided_actions,
            decided_mask=decided_mask,
            newly_idx=newly_idx,
            baseline_actions=baseline_actions,
            full_actions=full_actions,
            metrics_full=metrics_full,
            r_vec_full=r_vec_full,
        )

        # weights + valid_idx (bw>0 篩掉)
        weights, valid_idx = self.compute_commit_weights(newly_idx, mode=weight_mode)

        if len(valid_idx) == 0:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                r_vec_full,
                metrics_full,
                [],
                info
            )

        # align advantages to valid_idx order
        idx_map = {idx: k for k, idx in enumerate(newly_idx)}
        adv_raw = np.asarray([adv_raw_all[idx_map[i]] for i in valid_idx], dtype=np.float32)

        adv_norm = self.normalize_advantages(adv_raw) if do_norm else adv_raw

        return adv_raw, adv_norm, weights, r_vec_full, metrics_full, valid_idx, info

    # -----------------
    # old reward 算法
    # ----------------
    @staticmethod
    def normalize_like_trainloader(value, minD, maxD, min_val, max_val):
        if max_val == min_val:
            value_n = (maxD + minD) / 2
        else:
            value_n = (maxD - minD) * (value - min_val) / (max_val - min_val) + minD
        return round(value_n, 15)
        
    def reward_like_trainloader(self, src, dst, paths_metrics_dict, act, config):
        beta1=beta2=beta3=1
        if config.get("reward_mode","all") == "bwd_only":
            beta2=beta3=0
        m0,m1,m2 = self.metrics
        r = (beta1*paths_metrics_dict[str(src)][str(dst)][m0][1][act] +
            beta2*paths_metrics_dict[str(src)][str(dst)][m1][1][act] +
            beta3*paths_metrics_dict[str(src)][str(dst)][m2][1][act])
        return round(r, 15)

    def compute_all_pair_action_rewards_like_trainloader(self, config):
        """
        Legacy LS2IC/train_loader-style:
        - per (src,dst) compute K action rewards using historical min/max (paths_metrics_minmax_dict)
        - supports bwd/delay/loss
        Returns:
        rewards_dic, rewards_indicator, loss_value, delay_value
        """
        # 會一次把所有pair的所有20條kpath都計算出reward 所以才不用丟full action
        # 後面再取要的出來

        # accept dict or SimpleNamespace
        if config is None:
            raise ValueError("config is None in compute_all_pair_action_rewards_like_trainloader")
        if not isinstance(config, dict):
            # SimpleNamespace or other object with attributes
            config = vars(config)

        N = int(config["num_node"])

        assert self.paths_metrics_minmax_dict is not None, "call init_minmax_dic_like_trainloader first"
        N = int(config["num_node"])              # 1-based count
        K = int(config["action_dim"])
        size = N + 1

        # ---- 0) compute fast metrics for ALL global paths ----
        reward_def = self.kpath_cfg.get("reward_def", "old")
        bwd_only = (self.kpath_cfg.get("reward_mode","all") == "bwd_only")
        # old_sat swaps residual-bw for "satisfied throughput" = Π(1-link_loss) per
        # path, so we need per-link loss regardless of reward_mode.
        need_loss = (not bwd_only) or (reward_def == "old_sat")
        pm_all = self.compute_path_metrics_fast(
            full_actions=None,
            need_delay=(not bwd_only),
            need_loss=need_loss,
        )
        bwd_path = pm_all["bwd_path"]           # (numPairs*K,)
        if not bwd_only:
            delay_path = pm_all["delay_path"]   # (numPairs*K,)
            loss_path  = pm_all["loss_path"]    # (numPairs*K,) prob in [0,1]
        else:
            # legacy pipeline 後面仍會 reshape/建立 indicator，這裡補齊 shape
            # delay 用 0.0015 sec (→ 1.5 ms after *1000)，讓 legacy transform
            # 走 1/1.5 常數分支
            delay_path = np.full_like(bwd_path, 0.0015, dtype=np.float64)
            # loss 用 0 (prob)，轉 percent 後是 0，legacy transform 會走 else => 1/0.001 常數
            loss_path  = pm_all.get("loss_path", np.zeros_like(bwd_path, dtype=np.float64))

        # old_sat: replace bwd_path with satisfied-throughput proxy
        #   sat_path = Π(1 - link_loss) = 1 - loss_path  ∈ [0, 1]
        # Normalization pipeline below uses per-pair min/max and maps to [0, 100],
        # so the magnitude difference vs raw residual bw is absorbed.
        if reward_def == "old_sat":
            bwd_path = (1.0 - loss_path).astype(np.float64)

        # reshape to per-pair K
        P = self.numPairs
        bwd_pk   = bwd_path.reshape(P, K)

        # legacy thresholds (1.5 for delay, 0.001 for loss) were calibrated for
        # real-Mininet units: delay in ms, loss in percent. Sim env emits delay in
        # seconds and loss as probability [0,1], so convert to legacy units here.
        delay_pk = delay_path.reshape(P, K) * 1000.0    # sec -> ms
        loss_pk  = loss_path.reshape(P, K) * 100.0      # prob -> percent

        # ---- 1) build dict shells (train_loader-compatible) ----
        rewards_dic = {}
        rewards_indicator = {}
        loss_value = {}
        delay_value = {}
        paths_metrics_dict = {}

        for i in range(1, size):
            rewards_dic.setdefault(str(i), {})
            rewards_indicator.setdefault(str(i), {})
            loss_value.setdefault(str(i), {})
            delay_value.setdefault(str(i), {})
            paths_metrics_dict.setdefault(str(i), {})
            for j in range(1, size):
                if i == j:
                    continue
                rewards_dic[str(i)].setdefault(str(j), {})
                rewards_indicator[str(i)].setdefault(str(j), {})
                loss_value[str(i)].setdefault(str(j), {})
                delay_value[str(i)].setdefault(str(j), {})
                paths_metrics_dict[str(i)].setdefault(str(j), {
                    "bwd_paths": [],
                    "delay_paths": [],
                    "loss_paths": [],
                })

        # ---- 2) fill per (src,dst) raw lists from bwd_pk/delay_pk/loss_pk ----
        # mapping: env pair order = self.pairs = (0-based s,d)
        # legacy dict expects 1-based keys: src=1..N, dst=1..N
        for pair_id, (s0, d0) in enumerate(self.pairs):
            s = s0 + 1
            d = d0 + 1
            if s == d:
                continue
            if s < 1 or s >= size or d < 1 or d >= size:
                continue

            bwd_list   = [round(float(x), 6) for x in bwd_pk[pair_id]]
            delay_list = [round(float(x), 6) for x in delay_pk[pair_id]]
            loss_list  = [round(float(x), 6) for x in loss_pk[pair_id]]

            # keep indicator raw (before transform) aligned with your previous style
            loss_value[str(s)][str(d)]  = loss_list
            delay_value[str(s)][str(d)] = delay_list

            paths_metrics_dict[str(s)][str(d)]["bwd_paths"].append(bwd_list)
            paths_metrics_dict[str(s)][str(d)]["delay_paths"].append(delay_list)
            paths_metrics_dict[str(s)][str(d)]["loss_paths"].append(loss_list)

        # ---- 3) apply EXACT train_loader transforms + update historical min/max + normalize ----
        for i in paths_metrics_dict:
            for j in paths_metrics_dict[i]:
                for m in self.metrics:
                    raw = paths_metrics_dict[i][j][m][0]  # K list

                    if m == "bwd_paths":
                        cost = [round(float(val), 15) for val in raw]
                        paths_metrics_dict[i][j][m][0] = cost

                        mm = self.paths_metrics_minmax_dict[i][j][m]
                        mm["max"] = max(mm["max"], max(cost))
                        mm["min"] = min(mm["min"], min(cost))

                        # match your comment: max_val uses current step max (train_loader behavior)
                        cur_max = max(cost)
                        met_norm = [self.normalize_like_trainloader(x, 0, 100, mm["min"], cur_max) for x in cost]

                    elif m == "delay_paths":
                        # train_loader transform: val>1.5 -> 1/val else 1/1.5
                        cost = []
                        for val in raw:
                            v = float(val)
                            if v > 1.5:
                                cost.append(round(1.0 / v, 15))
                            else:
                                cost.append(round(1.0 / 1.5, 15))
                        paths_metrics_dict[i][j][m][0] = cost

                        mm = self.paths_metrics_minmax_dict[i][j][m]
                        mm["max"] = max(mm["max"], max(cost))
                        mm["min"] = min(mm["min"], min(cost))
                        met_norm = [self.normalize_like_trainloader(x, 0, 100, mm["min"], mm["max"]) for x in cost]

                    else:  # loss_paths
                        # legacy uses val in "percent": if >0.001 => 1/val else 1/0.001
                        cost = []
                        for val in raw:
                            v = float(val)
                            if v > 0.001:
                                cost.append(round(1.0 / v, 15))
                            else:
                                cost.append(1.0 / 0.001)
                        paths_metrics_dict[i][j][m][0] = cost

                        mm = self.paths_metrics_minmax_dict[i][j][m]
                        mm["max"] = max(mm["max"], max(cost))
                        mm["min"] = min(mm["min"], min(cost))
                        met_norm = [self.normalize_like_trainloader(x, 0, 100, mm["min"], mm["max"]) for x in cost]

                    # append normalized at index [1]
                    paths_metrics_dict[i][j][m].append(met_norm)

        # ---- 4) compose per-action reward + indicator tuples ----
        for i in paths_metrics_dict:
            for j in paths_metrics_dict[i]:
                rewards_actions = []
                rewards_actions_indicator = []
                for act in range(K):
                    r = self.reward_like_trainloader(i, j, paths_metrics_dict, act, config)
                    rewards_actions.append(r)
                    rewards_actions_indicator.append((
                        paths_metrics_dict[i][j]["bwd_paths"][1][act],
                        paths_metrics_dict[i][j]["delay_paths"][1][act],
                        paths_metrics_dict[i][j]["loss_paths"][1][act],
                    ))
                rewards_dic[i][j] = rewards_actions
                rewards_indicator[i][j] = rewards_actions_indicator

        return rewards_dic, rewards_indicator, loss_value, delay_value

    # ===================================
    # Unified reward interface
    # ==================================
    def evaluate_routing(
        self,
        decided_actions,
        decided_mask,
        baseline_actions,
        newly_idx=None,
        mode=None,
        weight_mode="sqrt_bw",
        do_norm=True,
        config=None,
    ):
        """
        Unified reward interface.

        Returns:
        out = {
            "full_actions": (N,),
            "r_vec": (N,),                 # per-pair reward (new or legacy-derived)
            "metrics": dict or None,       # only meaningful for new
            "adv_raw": (M,) or None,       # only if newly_idx is not None and mode uses COMA
            "adv_norm": (M,) or None,
            "weights": (M,) or None,
            "valid_idx": list or None,
            "info": dict,
        }
        """
        if mode is None:
            raise RuntimeError("reward mode undefined")

        full_actions = self.build_full_actions(decided_actions, decided_mask, baseline_actions)

        # -------------------------
        # NEW reward branch
        # -------------------------
        if mode == "new":
            metrics = self.compute_metrics_fast(full_actions, return_vectors=True)
            r_vec = self.reward_vec_from_metrics(metrics)
                    
            # --- extra logging: uniform sum over ALL pairs (including zero_bw) ---
            cfg = self.kpath_cfg
            old_izb = bool(cfg.get("ignore_zero_bw", True))
            cfg["ignore_zero_bw"] = False
            try:
                r_vec_all = self.reward_vec_from_metrics(metrics)   # includes bw==0 pairs
            finally:
                cfg["ignore_zero_bw"] = old_izb

            reward_sum_uniform_all = float(np.sum(r_vec_all))
            reward_mean_uniform_all = float(np.mean(r_vec_all)) if r_vec_all.size > 0 else 0.0

            out = {
                "full_actions": full_actions,
                "r_vec": r_vec,
                "metrics": metrics,
                "info": {
                    "mlu": float(metrics["mlu"]),
                    "overload": float(metrics["overload"]),
                    "sat_weighted": float(metrics["sat_weighted"]),
                },
                "reward_sum_uniform_all": reward_sum_uniform_all,
                "reward_mean_uniform_all": reward_mean_uniform_all,  # optional
                "num_pairs_all": int(self.numPairs),
            }

            if newly_idx is not None:
                adv_raw, adv_norm, weights, r_vec_full, metrics_full, valid_idx, info = \
                    self.counterfactual_advantages_with_fixes_per_pair(
                        decided_actions=decided_actions,
                        decided_mask=decided_mask,
                        newly_idx=newly_idx,
                        weight_mode=weight_mode,
                        do_norm=do_norm,
                        baseline_actions=baseline_actions,
                        # ---- pass caches (the cleanup) ----
                        full_actions=full_actions,
                        metrics_full=metrics,
                        r_vec_full=r_vec,
                    )
                out.update({
                    "adv_raw": adv_raw,
                    "adv_norm": adv_norm,
                    "weights": weights,
                    "valid_idx": valid_idx,
                    "info": {**out["info"], **info},
                })
            return out

        # -------------------------
        # LEGACY reward branch
        # -------------------------
        elif mode == "old":
            # 你 legacy reward 需要一份 rewards_dic[src][dst][k]
            # 這裡直接用你已經有的函數：compute_all_pair_action_rewards_like_trainloader
            # NOTE: 這會用 self.paths_metrics_minmax_dict，所以要先 init + 會更新歷史 minmax
            if config is None:
                raise RuntimeError("[evaluate_routing] legacy mode needs `config` for num_node/action_dim/delay_norm_div etc.")

            rewards_dic, rewards_indicator, loss_value, delay_value = \
                self.compute_all_pair_action_rewards_like_trainloader(config)

            # 把 full_actions 轉成 r_vec (per pair)
            r_vec = np.zeros((self.numPairs,), dtype=np.float32)
            for pid, (s, d) in enumerate(self.pairs):
                s1 = str(s + 1); d1 = str(d + 1)
                act = int(full_actions[pid])
                r_vec[pid] = float(rewards_dic[s1][d1][act])

            reward_sum_uniform_all = float(np.sum(r_vec))
            reward_mean_uniform_all = float(np.mean(r_vec)) if r_vec.size > 0 else 0.0

            out = {
                "full_actions": full_actions,
                "r_vec": r_vec,
                "metrics": None,
                "info": {"old": 1},
                "reward_sum_uniform_all": reward_sum_uniform_all,
                "reward_mean_uniform_all": reward_mean_uniform_all,  # optional
                "num_pairs_all": int(self.numPairs),
            }

            # legacy + COMA：概念上也可以做 counterfactual（同樣 expensive）
            # 但你現階段可以先不做，或用同樣方式 i->baseline 重算 rewards_dic（很慢）。
            # 建議：legacy 模式先只給 r_vec，不做 COMA baseline。
            
            # mlu 等指標也順便補上
            metrics = self.compute_metrics_fast(full_actions, return_vectors=True)
            out["info"].update({
                "mlu": float(metrics["mlu"]),
                "overload": float(metrics["overload"]),
                "sat_weighted": float(metrics["sat_weighted"]),
            })
            out["metrics_fast"] = metrics   # optional
            
            return out

        else:
            raise ValueError(f"Unknown mode={mode}, expected 'new' or 'old'")

    #=============================
    #== write back function
    #== used in real env inference
    #=============================

    def apply_netinfo_directed_to_graph(self, csv_path):
        # 用在 real testing 
        # 把 real 填回 graph 取得 regime 才能抓 path feature
        """
        把 net_info_directed.csv 的量測回填到 self.graph 的 edge attr
        - node1/node2: 1-based in csv -> 0-based in graph
        - bwd: free bw (Kbps)
        - utilization: used (Kbps)  = cap_kbps - free_kbps
        """
        df = pd.read_csv(csv_path)

        # 先清掉舊的 utilization（但不要動 crossing_paths 以免你 debug 用）
        for u in self.graph:
            for v in self.graph[u]:
                self.graph[u][v][0]["utilization"] = 0.0
        if self.edge_state is not None:
            self.edge_state[:, 0] = 0.0

        # 回填 directed edges
        for _, row in df.iterrows():
            u1 = int(row["node1"])
            v1 = int(row["node2"])
            u0 = u1 - 1
            v0 = v1 - 1

            free_kbps = float(row["bwd"])  # Kbps
            # Controller writes 'delay_tc_ms' (tc backlog ground truth, ms);
            # legacy CSVs may only have 'delay' (sec). Prefer tc, fallback to
            # lldp_fwd_ms / legacy 'delay'. See utils/manager.py header.
            if "delay_tc_ms" in row and pd.notna(row["delay_tc_ms"]):
                delay = float(row["delay_tc_ms"]) / 1000.0
            elif "lldp_fwd_ms" in row and pd.notna(row["lldp_fwd_ms"]):
                delay = float(row["lldp_fwd_ms"]) / 1000.0
            else:
                delay = float(row.get("delay", 0.0))
            # CSV pkloss column from Ryu (manager.py:113 + simple_tc_loss.py:114)
            # is in PERCENT (already ×100); env contract is RATIO [0,1] (docstring
            # at line 1097 / 1634). Convert here.
            # Bug surfaced 2026-05-18: get_link_features (line 1685) clamps
            # to [0,1] under lossdelay layout — without this /100, any loss > 1%
            # got clipped to 1.0 → loss_norm channel became binary for the RL agent
            # in real Mininet training. Affected all v4-era (_BASE_A2C_v2) EXPs
            # since 2026-05-17. Sim training unaffected (pkloss computed as
            # overflow_bits/arrival_bits in [0,1] at line 1144).
            pkloss = float(row.get("pkloss", 0.0)) / 100.0

            if u0 not in self.graph or v0 not in self.graph[u0]:
                continue

            edge = self.graph[u0][v0][0]

            # capacity (Kbps)
            cap_kbps = float(edge.get("capacity", None))

            # used = cap - free
            used_kbps = cap_kbps - free_kbps
            if used_kbps < 0:
                used_kbps = 0.0
            if used_kbps > cap_kbps:
                used_kbps = cap_kbps

            edge["capacity"] = cap_kbps           # Kbps
            edge["utilization"] = used_kbps       # Kbps
            edge["delay"] = delay
            edge["pkloss"] = pkloss               # ratio [0, 1]

            # Backfill queue state from real measurement so sim queue
            # tracks real env across steps (used in transient mode).
            # Real netem packets are ~1430 bytes (iperf3 UDP), sim uses 1500B.
            # Convert: real total bytes = queue_pkts * 1430, then to bits.
            queue_pkts = int(row.get("queue_pkts", 0))
            cfg = getattr(self, 'kpath_cfg', {}) or {}
            real_pkt_bytes = float(cfg.get("queue_real_pkt_bytes", 1430))
            if hasattr(self, 'queue_bits'):
                self.queue_bits[(u0, v0)] = queue_pkts * real_pkt_bytes * 8
            edge["queue_pkts"] = queue_pkts

