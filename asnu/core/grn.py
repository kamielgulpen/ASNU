import numpy as np
import math
import random

def establish_links(G, src_id, dst_id,
                    target_link_count, fraction, reciprocity_p, transitivity_p,
                    valid_communities=None, pa_scope="local",
                    bridge_probability=0, number_of_communities=1):

    num_links = G.existing_num_links.get((src_id, dst_id), 0)
    if num_links >= target_link_count:
        return True

    possible_communities = valid_communities
    if not possible_communities:
        return True

    # ── Phase A: community-based edge creation ────────────────────────────
    comm_order = list(dict.fromkeys(possible_communities))
    random.shuffle(comm_order)
    n_comms = len(comm_order)
    src_node_cache = {}
    max_passes = 3

    for _pass in range(max_passes):
        if num_links >= target_link_count:
            break

        for community_id in comm_order:
            if num_links >= target_link_count:
                break

            remaining = target_link_count - num_links
            quota = max(1, math.ceil(remaining / n_comms))

            if community_id not in src_node_cache:
                src_node_cache[community_id] = G.communities_to_nodes.get(
                    (community_id, src_id), []
                )
            src_nodes = src_node_cache[community_id]
            if not src_nodes:
                continue

            if (bridge_probability > 0 and number_of_communities > 1
                    and random.random() < bridge_probability):
                direction = random.choice([-1, 1])
                dst_community = (community_id + direction) % number_of_communities
            else:
                dst_community = community_id

            pool_key = (dst_community, dst_id)
            if pool_key not in G.popularity_pool:
                dst_nodes = G.communities_to_nodes.get((dst_community, dst_id), [])
                if dst_nodes:
                    sample_size = min(len(dst_nodes), math.ceil(len(dst_nodes) * fraction))
                    sampled = list(dst_nodes)
                    random.shuffle(sampled)
                    G.popularity_pool[pool_key] = sampled[:sample_size]
                else:
                    G.popularity_pool[pool_key] = []

            if not G.popularity_pool[pool_key]:
                continue

            created = 0
            local_attempts = 0
            max_local = quota * 3

            while created < quota and local_attempts < max_local and num_links < target_link_count:
                local_attempts += 1
                s = random.choice(src_nodes)
                d = random.choice(G.popularity_pool[pool_key])

                if s == d or G.graph.has_edge(s, d):
                    continue

                G.graph.add_edge(s, d)
                num_links += 1
                G.existing_num_links[(src_id, dst_id)] = num_links
                created += 1

                if random.random() < reciprocity_p:
                    self_pair = (dst_id == src_id)
                    if not (self_pair and num_links >= target_link_count):
                        rev_existing = G.existing_num_links.get((dst_id, src_id), 0)
                        rev_max = G.maximum_num_links.get((dst_id, src_id), 0)
                        if rev_existing < rev_max and not G.graph.has_edge(d, s):
                            G.graph.add_edge(d, s)
                            G.existing_num_links[(dst_id, src_id)] = rev_existing + 1
                            if self_pair:
                                num_links += 1
                                G.existing_num_links[(src_id, dst_id)] = num_links

                if random.random() > fraction and fraction != 1.0:
                    if pa_scope == "global":
                        for comm_id in range(G.number_of_communities):
                            if random.random() < (1.0 / G.number_of_communities) * fraction:
                                global_key = (comm_id, dst_id)
                                if global_key in G.popularity_pool:
                                    G.popularity_pool[global_key].append(d)
                    else:
                        if random.random() > fraction:
                            G.popularity_pool[pool_key].append(d)
                            dst_community_nodes = G.communities_to_nodes.get(
                                (dst_community, dst_id), []
                            )
                            if dst_community_nodes:
                                G.popularity_pool[pool_key].append(
                                    random.choice(dst_community_nodes)
                                )

                if random.random() >= transitivity_p:
                    continue
                for n in G.graph.neighbors(d):
                    if num_links >= target_link_count:
                        break
                    if n == s:
                        continue
                    n_id = G.nodes_to_group.get(n)
                    if n_id is None:
                        continue
                    pair = (src_id, n_id)
                    if pair not in G.maximum_num_links:
                        continue
                    existing = G.existing_num_links.get(pair, 0)
                    if existing < G.maximum_num_links[pair] and not G.graph.has_edge(s, n):
                        G.graph.add_edge(s, n)
                        G.existing_num_links[pair] = existing + 1
                        if n_id == dst_id:
                            num_links += 1
                            G.existing_num_links[(src_id, dst_id)] = num_links

                        if random.random() < reciprocity_p:
                            self_pair = (n_id == src_id and src_id == dst_id)
                            if not (self_pair and num_links >= target_link_count):
                                rev_pair = (n_id, src_id)
                                rev_existing = G.existing_num_links.get(rev_pair, 0)
                                rev_max = G.maximum_num_links.get(rev_pair, 0)
                                if not G.graph.has_edge(n, s) and rev_existing < rev_max:
                                    G.graph.add_edge(n, s)
                                    G.existing_num_links[rev_pair] = rev_existing + 1
                                    if self_pair:
                                        num_links += 1
                                        G.existing_num_links[(src_id, dst_id)] = num_links

    # ── Phase B: ring search across communities ───────────────────────────
    if num_links >= target_link_count:
        return True

    # Phase B requires node_coordinates on G
    node_coordinates = getattr(G, 'node_coordinates', None)
    if node_coordinates is None:
        return num_links <= target_link_count

    # Build sorted arrays if not already cached on G
    if not hasattr(G, '_phase_b_src_sorted'):
        G._phase_b_src_sorted = {}   # group_id -> sorted [(theta, node), ...]
        G._phase_b_dst_comm_sorted = {}  # group_id -> sorted [(centroid, comm_id), ...]

    if src_id not in G._phase_b_src_sorted:
        all_src_nodes = []
        for (comm_id, gid), nodes in G.communities_to_nodes.items():
            if gid == src_id:
                all_src_nodes.extend(nodes)
        G._phase_b_src_sorted[src_id] = sorted(
            ((node_coordinates.get(n, 0.5), n) for n in all_src_nodes)
        )

    if dst_id not in G._phase_b_dst_comm_sorted:
        comm_centroids = []
        for (comm_id, gid), nodes in G.communities_to_nodes.items():
            if gid == dst_id and nodes:
                centroid = sum(node_coordinates.get(n, 0.5) for n in nodes) / len(nodes)
                comm_centroids.append((centroid, comm_id))
        G._phase_b_dst_comm_sorted[dst_id] = sorted(comm_centroids)

    src_sorted = G._phase_b_src_sorted[src_id]
    dst_comm_sorted = G._phase_b_dst_comm_sorted[dst_id]

    if not src_sorted or not dst_comm_sorted:
        return num_links <= target_link_count

    PHASE_B_COMM_WINDOW = 200
    n_dst_comm = len(dst_comm_sorted)
    win = min(PHASE_B_COMM_WINDOW, n_dst_comm)
    n_src = len(src_sorted)
    dst_centroids = [c for c, _ in dst_comm_sorted]

    import bisect

    while num_links < target_link_count:
        src_indices = list(range(n_src))
        random.shuffle(src_indices)
        remaining = target_link_count - num_links
        edges_per_src = max(1, min(math.ceil(remaining / n_src), win))
        made_progress = False

        for si in src_indices:
            if num_links >= target_link_count:
                break
            theta_s, s = src_sorted[si]

            # Find nearest community by coordinate
            center = bisect.bisect_left(dst_centroids, theta_s)
            found = 0

            for delta in range(win):
                if found >= edges_per_src or num_links >= target_link_count:
                    break
                for j in [(center + delta) % n_dst_comm,
                           (center + n_dst_comm - delta - 1) % n_dst_comm]:
                    if found >= edges_per_src or num_links >= target_link_count:
                        break
                    _, comm_id = dst_comm_sorted[j]
                    pool_key = (comm_id, dst_id)
                    dst_nodes = G.communities_to_nodes.get(pool_key, [])
                    if not dst_nodes:
                        continue

                    d = random.choice(dst_nodes)
                    if s == d or G.graph.has_edge(s, d):
                        continue

                    G.graph.add_edge(s, d)
                    num_links += 1
                    G.existing_num_links[(src_id, dst_id)] = num_links
                    found += 1
                    made_progress = True

                    # Reciprocity
                    if random.random() < reciprocity_p:
                        self_pair = (dst_id == src_id)
                        if not (self_pair and num_links >= target_link_count):
                            rev_existing = G.existing_num_links.get((dst_id, src_id), 0)
                            rev_max = G.maximum_num_links.get((dst_id, src_id), 0)
                            if rev_existing < rev_max and not G.graph.has_edge(d, s):
                                G.graph.add_edge(d, s)
                                G.existing_num_links[(dst_id, src_id)] = rev_existing + 1
                                if self_pair:
                                    num_links += 1
                                    G.existing_num_links[(src_id, dst_id)] = num_links

                    # Transitivity
                    if transitivity_p > 0 and random.random() < transitivity_p:
                        for n in G.graph.neighbors(d):
                            if num_links >= target_link_count:
                                break
                            if n == s:
                                continue
                            n_id = G.nodes_to_group.get(n)
                            if n_id is None:
                                continue
                            pair = (src_id, n_id)
                            if pair not in G.maximum_num_links:
                                continue
                            existing = G.existing_num_links.get(pair, 0)
                            if existing < G.maximum_num_links[pair] and not G.graph.has_edge(s, n):
                                G.graph.add_edge(s, n)
                                G.existing_num_links[pair] = existing + 1
                                if n_id == dst_id:
                                    num_links += 1
                                    G.existing_num_links[(src_id, dst_id)] = num_links

                                if random.random() < reciprocity_p:
                                    self_pair = (n_id == src_id and src_id == dst_id)
                                    if not (self_pair and num_links >= target_link_count):
                                        rev_pair = (n_id, src_id)
                                        rev_existing = G.existing_num_links.get(rev_pair, 0)
                                        rev_max = G.maximum_num_links.get(rev_pair, 0)
                                        if not G.graph.has_edge(n, s) and rev_existing < rev_max:
                                            G.graph.add_edge(n, s)
                                            G.existing_num_links[rev_pair] = rev_existing + 1
                                            if self_pair:
                                                num_links += 1
                                                G.existing_num_links[(src_id, dst_id)] = num_links

        if not made_progress:
            break

    return num_links <= target_link_count